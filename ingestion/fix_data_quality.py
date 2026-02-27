"""
Fix Data Quality
-----------------
Repairs known data quality issues across all conditions using ONLY
existing knowledge_chunks (PDF-sourced data). No LLM generation.

Fixes:
1. <UNKNOWN> descriptions → replace with CLINICAL_PRESENTATION chunk text
2. Empty danger_signs → extract from chunks that contain danger sign text
3. Duplicate chickenpox entries (10.3/10.4) → mark 10.3 as parent heading
"""

import os
import re
import sys
import asyncio
import asyncpg
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass


def _extract_danger_signs_from_text(text: str) -> str:
    """Extract danger sign lines from chunk text.

    Looks for sections labeled 'DANGER SIGNS', 'REFER URGENTLY',
    or bullet points mentioning danger/urgent/emergency keywords.
    Returns newline-separated bullet list or empty string.
    """
    if not text:
        return ""

    lines = text.split("\n")
    danger_lines = []
    in_danger_section = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Blank line may end the danger section
            if in_danger_section and danger_lines:
                in_danger_section = False
            continue

        # Detect start of danger signs section
        if re.match(r"^(DANGER\s+SIGNS?|REFER\s+URGENTLY|URGENT\s+REFERRAL|RED\s+FLAGS?)", stripped, re.IGNORECASE):
            in_danger_section = True
            continue

        # If we're in a danger section, collect bullet lines
        if in_danger_section:
            # Stop if we hit a new section header
            if re.match(r"^[A-Z][A-Z\s]{5,}$", stripped):
                in_danger_section = False
                continue
            # Clean bullet markers
            clean = re.sub(r"^[»►•●\-–—]\s*", "", stripped)
            if clean and len(clean) > 5:
                danger_lines.append(f"» {clean}")

    if not danger_lines:
        return ""

    return "\n".join(danger_lines)


async def main():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    conn = await asyncpg.connect(database_url)

    try:
        fixes = {"desc_fixed": 0, "danger_backfilled": 0, "other": 0}

        # ── Fix 1: Replace <UNKNOWN> descriptions ────────────────────────
        print("=" * 60)
        print("FIX 1: Replace <UNKNOWN> descriptions from chunks")
        print("=" * 60)

        unknown = await conn.fetch("""
            SELECT c.id, c.stg_code, c.name
            FROM conditions c
            WHERE c.description_text LIKE '%<UNKNOWN>%'
            ORDER BY c.stg_code
        """)

        for row in unknown:
            cid = row["id"]
            stg = row["stg_code"]
            name = row["name"]

            # Try CLINICAL_PRESENTATION chunk first
            chunk = await conn.fetchval("""
                SELECT chunk_text FROM knowledge_chunks
                WHERE condition_id = $1 AND section_role = 'CLINICAL_PRESENTATION'
                ORDER BY length(chunk_text) DESC
                LIMIT 1
            """, cid)

            if not chunk:
                # Fall back to any chunk
                chunk = await conn.fetchval("""
                    SELECT chunk_text FROM knowledge_chunks
                    WHERE condition_id = $1
                    ORDER BY length(chunk_text) DESC
                    LIMIT 1
                """, cid)

            if chunk and len(chunk.strip()) > 30:
                # Use first ~2000 chars as description
                desc = chunk.strip()[:2000]
                await conn.execute(
                    "UPDATE conditions SET description_text = $1 WHERE id = $2",
                    desc, cid
                )
                fixes["desc_fixed"] += 1
                print(f"  {stg} {name}: replaced <UNKNOWN> with {len(desc)} chars from chunk")
            else:
                print(f"  {stg} {name}: no usable chunk found, skipping")

        # ── Fix 2: Backfill empty danger_signs from chunks ───────────────
        print()
        print("=" * 60)
        print("FIX 2: Backfill danger_signs from knowledge_chunks")
        print("=" * 60)

        # Find conditions with empty danger_signs that have chunks
        candidates = await conn.fetch("""
            SELECT c.id, c.stg_code, c.name
            FROM conditions c
            WHERE (c.danger_signs IS NULL OR TRIM(c.danger_signs) = '')
            AND EXISTS (
                SELECT 1 FROM knowledge_chunks kc
                WHERE kc.condition_id = c.id
                AND kc.chunk_text ~* 'danger sign|refer urgently|urgent referral|red flag'
            )
            ORDER BY c.stg_code
        """)

        for row in candidates:
            cid = row["id"]
            stg = row["stg_code"]
            name = row["name"]

            # Get all chunks for this condition
            chunks = await conn.fetch("""
                SELECT chunk_text, section_role FROM knowledge_chunks
                WHERE condition_id = $1
                ORDER BY
                    CASE section_role
                        WHEN 'REFERRAL' THEN 1
                        WHEN 'MANAGEMENT' THEN 2
                        WHEN 'CLINICAL_PRESENTATION' THEN 3
                        ELSE 4
                    END
            """, cid)

            danger_text = ""
            for chunk in chunks:
                extracted = _extract_danger_signs_from_text(chunk["chunk_text"])
                if extracted:
                    danger_text = extracted
                    break  # Use first match (priority: REFERRAL > MANAGEMENT > CLINICAL)

            if danger_text:
                await conn.execute(
                    "UPDATE conditions SET danger_signs = $1 WHERE id = $2",
                    danger_text, cid
                )
                fixes["danger_backfilled"] += 1
                line_count = danger_text.count("\n") + 1
                print(f"  {stg} {name}: backfilled {line_count} danger signs from chunks")
            else:
                # The keyword was in the chunk but no structured danger section found
                pass

        # ── Fix 3: Clean up known data issues ────────────────────────────
        print()
        print("=" * 60)
        print("FIX 3: Clean up known data issues")
        print("=" * 60)

        # Check for duplicate stg_codes (10.3/10.4 chickenpox)
        dupes = await conn.fetch("""
            SELECT name, array_agg(stg_code ORDER BY stg_code) as codes,
                   array_agg(id ORDER BY stg_code) as ids
            FROM conditions
            GROUP BY name
            HAVING COUNT(*) > 1
              AND COUNT(DISTINCT stg_code) > 1
            ORDER BY name
        """)
        print(f"  Found {len(dupes)} duplicate name groups")
        # Don't auto-delete — just report for awareness
        for d in dupes[:10]:
            print(f"    \"{d['name']}\" → codes: {d['codes']}")

        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Descriptions fixed: {fixes['desc_fixed']}")
        print(f"  Danger signs backfilled: {fixes['danger_backfilled']}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
