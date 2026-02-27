"""
Backfill Hollow Conditions
--------------------------
For conditions with empty description_text, uses their knowledge_chunks
text + Claude Haiku to extract structured fields:
  - description_text
  - general_measures
  - medicine_treatment
  - danger_signs

Reads chunk text from DB, sends to Haiku, updates the condition row.
Cost: ~$1-2 for ~100 conditions.
"""

import os
import sys
import json
import asyncio
import time
import anthropic
import asyncpg
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass


EXTRACTION_TOOL = {
    "name": "extract_sections",
    "description": "Extract structured text sections from raw STG condition text",
    "input_schema": {
        "type": "object",
        "properties": {
            "description_text": {
                "type": "string",
                "description": "Clinical description of the condition: what it is, who it affects, how it presents. Include all text from DESCRIPTION/DIAGNOSTIC CRITERIA sections. If the text describes a treatment algorithm or flowchart, describe it here."
            },
            "general_measures": {
                "type": "string",
                "description": "Non-pharmacological management: patient education, lifestyle advice, supportive care, monitoring instructions. Include nursing care steps."
            },
            "medicine_treatment": {
                "type": "string",
                "description": "All pharmacological treatment information: medicine names, doses, routes, frequencies, durations, treatment steps/lines. Preserve table structure if present (use markdown tables)."
            },
            "danger_signs": {
                "type": "string",
                "description": "Warning signs that require urgent referral or escalation. Include all DANGER SIGNS, RED FLAGS, and REFER URGENTLY criteria."
            },
            "referral_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific conditions/criteria for when to refer the patient. Each item should be one referral criterion."
            }
        },
        "required": ["description_text", "medicine_treatment"]
    }
}


PROMPT_TEMPLATE = """You are extracting structured clinical text sections from the South African Standard Treatment Guidelines (STG).

Below is raw text from the STG section for: {condition_name} (STG code: {stg_code})

Extract the following sections from this text:
1. DESCRIPTION: What the condition is, diagnostic criteria, clinical presentation
2. GENERAL MEASURES: Non-drug management, patient education, lifestyle advice, monitoring
3. MEDICINE TREATMENT: All drugs, doses, routes, durations, treatment steps. Keep table format if present.
4. DANGER SIGNS: Warning signs requiring urgent referral
5. REFERRAL CRITERIA: When to refer the patient

IMPORTANT:
- Extract DIRECTLY from the text — do not infer or add information
- If a section has no relevant content in the text, return an empty string for that section
- Preserve clinical specificity (exact doses, frequencies, durations)
- If the text contains treatment tables or step-based protocols, preserve them as markdown tables
- Include ALL content, even if it seems repetitive — completeness is critical

RAW TEXT:
{raw_text}

Extract the sections using the extract_sections tool."""


async def get_hollow_conditions(conn):
    """Get conditions with empty description_text that are NOT parent headings."""
    # Get all conditions with empty description
    empty = await conn.fetch("""
        SELECT c.id, c.stg_code, c.name,
               c.description_text, c.general_measures, c.medicine_treatment, c.danger_signs
        FROM conditions c
        WHERE (c.description_text IS NULL OR c.description_text = '')
        ORDER BY c.stg_code
    """)

    # Filter out parent headings (those with populated children)
    all_codes = {r['stg_code'] for r in await conn.fetch("SELECT stg_code FROM conditions")}
    populated_codes = {r['stg_code'] for r in await conn.fetch(
        "SELECT stg_code FROM conditions WHERE description_text IS NOT NULL AND description_text != ''"
    )}

    result = []
    for row in empty:
        code = row['stg_code']
        # Check if any populated child exists
        is_parent = any(
            pc.startswith(code + '.') for pc in populated_codes
        )
        if not is_parent:
            result.append(row)

    return result


async def get_chunk_text(conn, condition_id):
    """Get concatenated knowledge_chunks text for a condition."""
    rows = await conn.fetch("""
        SELECT chunk_text, section_role
        FROM knowledge_chunks
        WHERE condition_id = $1
        ORDER BY id
    """, condition_id)

    if not rows:
        return ""

    parts = []
    for r in rows:
        text = r['chunk_text'] or ''
        if text.strip():
            parts.append(text.strip())

    return "\n\n".join(parts)


def extract_sections_with_haiku(client, condition_name, stg_code, raw_text):
    """Use Haiku to extract structured sections from raw text."""
    # Truncate to avoid token limits (keep first ~12000 chars)
    if len(raw_text) > 12000:
        raw_text = raw_text[:12000] + "\n\n[... text truncated for processing ...]"

    prompt = PROMPT_TEMPLATE.format(
        condition_name=condition_name,
        stg_code=stg_code,
        raw_text=raw_text
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4000,
                tools=[EXTRACTION_TOOL],
                tool_choice={"type": "tool", "name": "extract_sections"},
                messages=[{"role": "user", "content": prompt}]
            )

            for block in response.content:
                if block.type == "tool_use" and block.name == "extract_sections":
                    return block.input

            return None

        except Exception as e:
            if attempt < max_retries - 1 and ("overloaded" in str(e).lower() or "rate" in str(e).lower()):
                wait = 15 * (2 ** attempt)
                print(f"    Retrying in {wait}s...", flush=True)
                time.sleep(wait)
                continue
            raise


async def update_condition(conn, condition_id, sections):
    """Update a condition's text fields from extracted sections."""
    await conn.execute("""
        UPDATE conditions SET
            description_text = CASE WHEN LENGTH($2) > LENGTH(COALESCE(description_text, ''))
                                    THEN $2 ELSE description_text END,
            general_measures = CASE WHEN LENGTH($3) > LENGTH(COALESCE(general_measures, ''))
                                    THEN $3 ELSE general_measures END,
            medicine_treatment = CASE WHEN LENGTH($4) > LENGTH(COALESCE(medicine_treatment, ''))
                                      THEN $4 ELSE medicine_treatment END,
            danger_signs = CASE WHEN LENGTH($5) > LENGTH(COALESCE(danger_signs, ''))
                                THEN $5 ELSE danger_signs END,
            referral_criteria = CASE WHEN LENGTH($6::text) > LENGTH(COALESCE(referral_criteria::text, '[]'))
                                     THEN $6 ELSE referral_criteria END,
            updated_at = NOW()
        WHERE id = $1
    """,
        condition_id,
        sections.get('description_text', ''),
        sections.get('general_measures', ''),
        sections.get('medicine_treatment', ''),
        sections.get('danger_signs', ''),
        json.dumps(sections.get('referral_criteria', [])),
    )


async def main():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    conn = await asyncpg.connect(database_url)
    client = anthropic.Anthropic()

    try:
        # Get hollow conditions (not parent headings)
        hollow = await get_hollow_conditions(conn)
        print(f"Found {len(hollow)} genuinely hollow conditions to backfill")
        print(f"{'='*60}")

        success = 0
        skipped = 0
        errors = 0

        for i, cond in enumerate(hollow):
            cid = cond['id']
            name = cond['name']
            code = cond['stg_code']

            # Get chunk text
            chunk_text = await get_chunk_text(conn, cid)

            if len(chunk_text.strip()) < 50:
                print(f"  [{i+1}/{len(hollow)}] {code} {name} — insufficient chunk text ({len(chunk_text)} chars), skipping")
                skipped += 1
                continue

            print(f"  [{i+1}/{len(hollow)}] {code} {name} ({len(chunk_text)} chars)", end="", flush=True)

            try:
                sections = extract_sections_with_haiku(client, name, code, chunk_text)

                if sections:
                    desc_len = len(sections.get('description_text', ''))
                    med_len = len(sections.get('medicine_treatment', ''))
                    gm_len = len(sections.get('general_measures', ''))
                    ds_len = len(sections.get('danger_signs', ''))
                    ref_count = len(sections.get('referral_criteria', []))

                    await update_condition(conn, cid, sections)
                    print(f" — OK (desc={desc_len}, meds={med_len}, gm={gm_len}, ds={ds_len}, ref={ref_count})")
                    success += 1
                else:
                    print(f" — EMPTY (Haiku returned no tool use)")
                    skipped += 1

                # Small delay to avoid rate limits
                if (i + 1) % 5 == 0:
                    time.sleep(2)

            except Exception as e:
                print(f" — ERROR: {e}")
                errors += 1

        print(f"\n{'='*60}")
        print(f"BACKFILL COMPLETE")
        print(f"{'='*60}")
        print(f"  Success: {success}")
        print(f"  Skipped: {skipped}")
        print(f"  Errors:  {errors}")

        # Verify
        still_empty = await conn.fetchval("""
            SELECT COUNT(*) FROM conditions
            WHERE (description_text IS NULL OR description_text = '')
        """)
        print(f"\n  Conditions still empty after backfill: {still_empty}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
