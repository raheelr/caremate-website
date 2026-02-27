"""
Re-extraction Verification
--------------------------
Automated quality checks after a full re-extraction.
Compares against the targets from the extraction plan.

Run: python3 ingestion/verify_reextraction.py

Targets:
  - Conditions with 0 knowledge chunks: 96 -> target 0
  - Conditions with 0 original features: 117 -> target <10
  - Total conditions: >= 410
  - Total knowledge chunks: 975 -> target >1500
  - Chapter 12 conditions with content: 0 -> target all
"""

import os
import sys
import asyncio
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent / 'db'))
from database import get_connection


async def verify():
    conn = await get_connection()

    print("=" * 60)
    print("RE-EXTRACTION VERIFICATION")
    print("=" * 60)

    checks_passed = 0
    checks_failed = 0

    def check(name, actual, target, comparator=">="):
        nonlocal checks_passed, checks_failed
        if comparator == ">=":
            passed = actual >= target
        elif comparator == "<=":
            passed = actual <= target
        elif comparator == "==":
            passed = actual == target
        elif comparator == ">":
            passed = actual > target
        elif comparator == "<":
            passed = actual < target
        else:
            passed = False

        status = "PASS" if passed else "FAIL"
        if passed:
            checks_passed += 1
        else:
            checks_failed += 1
        print(f"  [{status}] {name}: {actual} (target: {comparator} {target})")

    # ── 1. Total conditions ──────────────────────────────────────
    print("\n1. CONDITIONS")
    total_conditions = await conn.fetchval("SELECT COUNT(*) FROM conditions")
    check("Total conditions", total_conditions, 410, ">=")

    # ── 2. Conditions with 0 knowledge chunks ────────────────────
    print("\n2. KNOWLEDGE CHUNKS")
    total_chunks = await conn.fetchval("SELECT COUNT(*) FROM knowledge_chunks")
    check("Total knowledge chunks", total_chunks, 1500, ">")

    zero_chunks = await conn.fetchval("""
        SELECT COUNT(*) FROM conditions c
        WHERE NOT EXISTS (
            SELECT 1 FROM knowledge_chunks kc WHERE kc.condition_id = c.id
        )
    """)
    check("Conditions with 0 chunks", zero_chunks, 0, "==")

    # Table chunks
    table_chunks = await conn.fetchval(
        "SELECT COUNT(*) FROM knowledge_chunks WHERE is_table = TRUE"
    )
    print(f"  [INFO] Table chunks (Docling): {table_chunks}")

    # Algorithm chunks
    algo_chunks = await conn.fetchval(
        "SELECT COUNT(*) FROM knowledge_chunks WHERE is_algorithm = TRUE"
    )
    print(f"  [INFO] Algorithm chunks (Vision): {algo_chunks}")

    # ── 3. Clinical features ─────────────────────────────────────
    print("\n3. CLINICAL FEATURES (KNOWLEDGE GRAPH)")
    total_relationships = await conn.fetchval("SELECT COUNT(*) FROM clinical_relationships")
    print(f"  [INFO] Total relationships: {total_relationships}")

    # Original (non-enrichment) features
    original_features = await conn.fetchval("""
        SELECT COUNT(*) FROM clinical_relationships
        WHERE source_section IS NULL OR source_section != 'ENRICHMENT'
    """)
    enrichment_features = await conn.fetchval("""
        SELECT COUNT(*) FROM clinical_relationships
        WHERE source_section = 'ENRICHMENT'
    """)
    print(f"  [INFO] Original features: {original_features}")
    print(f"  [INFO] Enrichment features: {enrichment_features}")

    if total_relationships > 0:
        enrichment_ratio = enrichment_features / total_relationships
        print(f"  [INFO] Enrichment ratio: {enrichment_ratio:.2f} (target < 0.50)")
        check("Enrichment ratio", enrichment_ratio, 0.50, "<")

    # Conditions with 0 original features
    zero_original = await conn.fetchval("""
        SELECT COUNT(*) FROM conditions c
        WHERE NOT EXISTS (
            SELECT 1 FROM clinical_relationships cr
            WHERE cr.condition_id = c.id
            AND (cr.source_section IS NULL OR cr.source_section != 'ENRICHMENT')
        )
    """)
    check("Conditions with 0 original features", zero_original, 10, "<")

    # ── 4. Medicines ─────────────────────────────────────────────
    print("\n4. MEDICINES")
    total_medicines = await conn.fetchval("SELECT COUNT(*) FROM medicines")
    print(f"  [INFO] Total medicines: {total_medicines}")

    total_condition_meds = await conn.fetchval("SELECT COUNT(*) FROM condition_medicines")
    print(f"  [INFO] Condition-medicine links: {total_condition_meds}")

    # ── 5. Chapter 12 (STIs) ────────────────────────────────────
    print("\n5. CHAPTER 12 (STIs)")
    ch12_total = await conn.fetchval("SELECT COUNT(*) FROM conditions WHERE chapter = 12")
    ch12_with_chunks = await conn.fetchval("""
        SELECT COUNT(DISTINCT c.id) FROM conditions c
        JOIN knowledge_chunks kc ON kc.condition_id = c.id
        WHERE c.chapter = 12
    """)
    ch12_with_features = await conn.fetchval("""
        SELECT COUNT(DISTINCT c.id) FROM conditions c
        JOIN clinical_relationships cr ON cr.condition_id = c.id
        WHERE c.chapter = 12
    """)
    print(f"  [INFO] Chapter 12 conditions: {ch12_total}")
    print(f"  [INFO] With knowledge chunks: {ch12_with_chunks}")
    print(f"  [INFO] With clinical features: {ch12_with_features}")
    if ch12_total > 0:
        check("Ch12 conditions with chunks", ch12_with_chunks, ch12_total, "==")

    # ── 6. Spot checks ──────────────────────────────────────────
    print("\n6. SPOT CHECKS")
    spot_checks = [
        ("12.1", "VDS (STI flowchart)"),
        ("4.7.1", "Hypertension in Adults"),
        ("1.2", "Oral Thrush"),
        ("2.9.1", "Diarrhoea in Children"),
        ("16.4.1", "Depression"),
    ]
    for stg_code, label in spot_checks:
        row = await conn.fetchrow("""
            SELECT c.id, c.name,
                (SELECT COUNT(*) FROM knowledge_chunks kc WHERE kc.condition_id = c.id) as chunks,
                (SELECT COUNT(*) FROM clinical_relationships cr WHERE cr.condition_id = c.id
                 AND (cr.source_section IS NULL OR cr.source_section != 'ENRICHMENT')) as features,
                (SELECT COUNT(*) FROM condition_medicines cm WHERE cm.condition_id = c.id) as meds
            FROM conditions c WHERE c.stg_code = $1
        """, stg_code)
        if row:
            print(f"  {stg_code} {label}: chunks={row['chunks']}, "
                  f"features={row['features']}, meds={row['meds']}")
            if row['chunks'] == 0:
                checks_failed += 1
                print(f"    [FAIL] No knowledge chunks!")
            else:
                checks_passed += 1
        else:
            print(f"  {stg_code} {label}: NOT FOUND")
            checks_failed += 1

    # ── 7. Per-chapter summary ───────────────────────────────────
    print("\n7. PER-CHAPTER SUMMARY")
    chapters = await conn.fetch("""
        SELECT c.chapter, c.chapter_name, COUNT(*) as count,
            SUM(CASE WHEN EXISTS (
                SELECT 1 FROM knowledge_chunks kc WHERE kc.condition_id = c.id
            ) THEN 1 ELSE 0 END) as with_chunks,
            SUM(CASE WHEN EXISTS (
                SELECT 1 FROM clinical_relationships cr WHERE cr.condition_id = c.id
                AND (cr.source_section IS NULL OR cr.source_section != 'ENRICHMENT')
            ) THEN 1 ELSE 0 END) as with_features
        FROM conditions c
        GROUP BY c.chapter, c.chapter_name
        ORDER BY c.chapter
    """)
    print(f"  {'Ch':>3} {'Name':<35} {'Total':>5} {'Chunks':>7} {'Feats':>7}")
    print(f"  {'-'*3} {'-'*35} {'-'*5} {'-'*7} {'-'*7}")
    for ch in chapters:
        print(f"  {ch['chapter']:>3} {(ch['chapter_name'] or ''):<35} "
              f"{ch['count']:>5} {ch['with_chunks']:>7} {ch['with_features']:>7}")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"VERIFICATION COMPLETE: {checks_passed} passed, {checks_failed} failed")
    print(f"{'='*60}")

    await conn.close()
    return checks_failed == 0


if __name__ == "__main__":
    success = asyncio.run(verify())
    sys.exit(0 if success else 1)
