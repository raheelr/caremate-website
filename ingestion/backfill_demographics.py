"""
Backfill Condition Demographics
-------------------------------
Uses condition names + description_text + Claude Haiku to determine:
  - applies_to_male (bool)
  - applies_to_female (bool)
  - min_age_years (int)
  - max_age_years (int)

Most conditions apply to both sexes and all ages. This script identifies
gender-specific (e.g., prostate, cervical) and age-limited conditions.
"""

import os
import sys
import json
import asyncio
import time
import anthropic
import asyncpg
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass


CLASSIFICATION_TOOL = {
    "name": "classify_demographics",
    "description": "Classify a medical condition's demographic applicability",
    "input_schema": {
        "type": "object",
        "properties": {
            "applies_to_male": {
                "type": "boolean",
                "description": "True if this condition can affect males. False ONLY for female-specific conditions (e.g., cervical cancer, endometriosis, pregnancy complications, ovarian conditions)."
            },
            "applies_to_female": {
                "type": "boolean",
                "description": "True if this condition can affect females. False ONLY for male-specific conditions (e.g., prostate conditions, testicular conditions, balanitis)."
            },
            "min_age_years": {
                "type": "integer",
                "description": "Minimum age in years this condition typically presents. Use 0 for conditions that can affect neonates/infants. Use 18 for adult-only conditions. Use 0 if unsure."
            },
            "max_age_years": {
                "type": "integer",
                "description": "Maximum age in years. Use 18 for paediatric-only conditions. Use 999 for conditions with no upper age limit (most conditions)."
            },
        },
        "required": ["applies_to_male", "applies_to_female", "min_age_years", "max_age_years"]
    }
}

PROMPT = """Classify this medical condition's demographic applicability.

Condition: {name} (STG code: {stg_code})
Description: {description}

RULES:
- Most conditions apply to BOTH sexes and ALL ages — only set false/limits for clearly specific conditions
- Female-only: pregnancy, cervical, ovarian, endometrial, vulvar, vaginal conditions
- Male-only: prostate, testicular, penile, epididymal conditions
- Paediatric-only (max_age=18): neonatal jaundice, childhood immunisations, etc.
- Adult-only (min_age=18): conditions explicitly described as adult presentations
- When in doubt, default to applies_to_male=true, applies_to_female=true, min_age=0, max_age=999

Use the classify_demographics tool."""


async def main():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    conn = await asyncpg.connect(database_url)
    client = anthropic.Anthropic()

    try:
        # Get all conditions (they all have defaults currently)
        conditions = await conn.fetch("""
            SELECT id, stg_code, name, LEFT(description_text, 500) as desc_preview
            FROM conditions
            ORDER BY stg_code
        """)
        print(f"Processing {len(conditions)} conditions for demographic classification")
        print("=" * 60)

        updated = 0
        errors = 0

        for i, cond in enumerate(conditions):
            name = cond["name"]
            stg_code = cond["stg_code"]
            desc = cond["desc_preview"] or ""

            print(f"  [{i+1}/{len(conditions)}] {stg_code} {name}", end="", flush=True)

            try:
                prompt = PROMPT.format(name=name, stg_code=stg_code, description=desc[:300])

                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=200,
                    tools=[CLASSIFICATION_TOOL],
                    tool_choice={"type": "tool", "name": "classify_demographics"},
                    messages=[{"role": "user", "content": prompt}],
                )

                result = None
                for block in response.content:
                    if block.type == "tool_use" and block.name == "classify_demographics":
                        result = block.input
                        break

                if not result:
                    print(" — EMPTY")
                    continue

                m = result.get("applies_to_male", True)
                f = result.get("applies_to_female", True)
                min_age = result.get("min_age_years", 0)
                max_age = result.get("max_age_years", 999)

                # Only update if something is non-default
                is_default = (m is True and f is True and min_age == 0 and max_age == 999)

                if not is_default:
                    await conn.execute("""
                        UPDATE conditions SET
                            applies_to_male = $2,
                            applies_to_female = $3,
                            min_age_years = $4,
                            max_age_years = $5,
                            updated_at = NOW()
                        WHERE id = $1
                    """, cond["id"], m, f, min_age, max_age)
                    print(f" — M={m} F={f} age={min_age}-{max_age}")
                    updated += 1
                else:
                    print(" — default (both/all)")

                # Rate limiting
                if (i + 1) % 10 == 0:
                    time.sleep(1)

            except Exception as e:
                print(f" — ERROR: {e}")
                errors += 1
                if "rate" in str(e).lower() or "overloaded" in str(e).lower():
                    time.sleep(15)

        print(f"\n{'=' * 60}")
        print(f"DEMOGRAPHIC BACKFILL COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Updated (non-default): {updated}")
        print(f"  Errors: {errors}")

        # Summary
        stats = await conn.fetchrow("""
            SELECT
                COUNT(*) FILTER (WHERE applies_to_male = false) as male_excluded,
                COUNT(*) FILTER (WHERE applies_to_female = false) as female_excluded,
                COUNT(*) FILTER (WHERE min_age_years > 0) as has_min_age,
                COUNT(*) FILTER (WHERE max_age_years < 999) as has_max_age
            FROM conditions
        """)
        print(f"\n  Male-excluded: {stats['male_excluded']}")
        print(f"  Female-excluded: {stats['female_excluded']}")
        print(f"  Has min_age > 0: {stats['has_min_age']}")
        print(f"  Has max_age < 999: {stats['has_max_age']}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
