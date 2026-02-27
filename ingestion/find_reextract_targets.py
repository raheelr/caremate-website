"""
Find conditions that need re-extraction from the STG PDF.

Identifies:
1. Conditions with zero knowledge chunks (empty shells)
2. Conditions with very few original features (thin extraction)
3. Outputs a comma-separated list of STG codes ready for --stg-codes flag

Usage:
    python3 ingestion/find_reextract_targets.py
"""

import os
import sys
import asyncio
import asyncpg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


async def main():
    url = os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    conn = await asyncpg.connect(url)

    # Category 1: Zero knowledge chunks
    zero_chunks = await conn.fetch("""
        SELECT c.stg_code, c.name, LENGTH(c.description_text) as desc_len,
               (SELECT COUNT(*) FROM clinical_relationships cr
                WHERE cr.condition_id = c.id
                AND (cr.source_section IS NULL OR cr.source_section != 'ENRICHMENT')) as orig_features
        FROM conditions c
        WHERE NOT EXISTS (SELECT 1 FROM knowledge_chunks kc WHERE kc.condition_id = c.id)
        ORDER BY c.stg_code
    """)

    # Category 2: Very few original features (1-3), even if they have chunks
    thin_extraction = await conn.fetch("""
        SELECT c.stg_code, c.name,
               (SELECT COUNT(*) FROM knowledge_chunks kc WHERE kc.condition_id = c.id) as chunk_count,
               COUNT(cr.id) as orig_features
        FROM conditions c
        LEFT JOIN clinical_relationships cr ON cr.condition_id = c.id
            AND (cr.source_section IS NULL OR cr.source_section != 'ENRICHMENT')
        GROUP BY c.id, c.stg_code, c.name
        HAVING COUNT(cr.id) BETWEEN 1 AND 3
        AND EXISTS (SELECT 1 FROM knowledge_chunks kc WHERE kc.condition_id = c.id)
        ORDER BY c.stg_code
    """)

    print("=" * 70)
    print("CONDITIONS NEEDING RE-EXTRACTION")
    print("=" * 70)

    print(f"\n--- CATEGORY 1: Zero knowledge chunks ({len(zero_chunks)} conditions) ---")
    for r in zero_chunks:
        print(f"  {r['stg_code']:12s} {r['name'][:50]:50s} desc={r['desc_len'] or 0:5d}  orig_feats={r['orig_features']}")

    print(f"\n--- CATEGORY 2: Thin extraction, 1-3 original features ({len(thin_extraction)} conditions) ---")
    for r in thin_extraction:
        print(f"  {r['stg_code']:12s} {r['name'][:50]:50s} chunks={r['chunk_count']}  orig_feats={r['orig_features']}")

    # Combine all codes
    all_codes = set()
    for r in zero_chunks:
        all_codes.add(r['stg_code'])
    for r in thin_extraction:
        all_codes.add(r['stg_code'])

    # Sort codes naturally
    def sort_key(code):
        parts = code.split(".")
        return tuple(int(p) if p.isdigit() else p for p in parts)

    sorted_codes = sorted(all_codes, key=sort_key)

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {len(sorted_codes)} conditions need re-extraction")
    print(f"{'=' * 70}")
    print(f"\nRun command:")
    print(f"  python3 ingestion/pipeline.py --pdf stg.pdf --stg-codes '{','.join(sorted_codes)}' --force")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
