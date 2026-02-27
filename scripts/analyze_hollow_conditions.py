"""
Analyze hollow conditions to distinguish:
1. Parent headings with populated children (can be left as-is or merged)
2. Genuinely hollow — no populated child, but knowledge_chunks have text

A condition is "hollow" if it has no description_text (empty or NULL).
A hollow condition is a "parent heading" if there exists another condition
with a longer stg_code that starts with the same prefix + ".",
AND that child has non-empty description_text.
"""

import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATABASE_URL = os.environ['DATABASE_URL']


async def main():
    conn = await asyncpg.connect(DATABASE_URL)

    try:
        # 1. Get all hollow conditions (no description_text)
        hollow = await conn.fetch("""
            SELECT id, stg_code, name, 
                   COALESCE(description_text, '') as description_text
            FROM conditions
            WHERE description_text IS NULL 
               OR TRIM(description_text) = ''
            ORDER BY stg_code
        """)

        print(f"Total hollow conditions: {len(hollow)}")

        # 2. Get all conditions with populated description_text
        populated = await conn.fetch("""
            SELECT id, stg_code, name
            FROM conditions
            WHERE description_text IS NOT NULL 
              AND TRIM(description_text) != ''
            ORDER BY stg_code
        """)

        populated_codes = {r['stg_code'] for r in populated}

        # 3. For each hollow condition, check if it's a parent heading
        parent_headings = []
        genuinely_hollow = []

        for row in hollow:
            code = row['stg_code']
            # Check if any populated condition has a code starting with this code + "."
            children = [pc for pc in populated_codes if pc.startswith(code + '.')]

            if children:
                parent_headings.append({
                    'id': row['id'],
                    'stg_code': code,
                    'name': row['name'],
                    'populated_children': sorted(children),
                    'child_count': len(children),
                })
            else:
                genuinely_hollow.append({
                    'id': row['id'],
                    'stg_code': code,
                    'name': row['name'],
                })

        # 4. For genuinely hollow, get knowledge_chunks info + edge counts in batch
        if genuinely_hollow:
            gh_ids = [gh['id'] for gh in genuinely_hollow]

            # Batch query for chunks
            chunk_data = await conn.fetch("""
                SELECT condition_id, COUNT(*) as chunk_count, 
                       SUM(LENGTH(chunk_text)) as total_chars
                FROM knowledge_chunks
                WHERE condition_id = ANY($1::int[])
                GROUP BY condition_id
            """, gh_ids)
            chunk_map = {r['condition_id']: (r['chunk_count'], r['total_chars']) for r in chunk_data}

            # Batch query for edges
            edge_data = await conn.fetch("""
                SELECT condition_id, COUNT(*) as edge_count
                FROM clinical_relationships
                WHERE condition_id = ANY($1::int[])
                GROUP BY condition_id
            """, gh_ids)
            edge_map = {r['condition_id']: r['edge_count'] for r in edge_data}

            for gh in genuinely_hollow:
                cid = gh['id']
                gh['chunk_count'] = chunk_map.get(cid, (0, 0))[0]
                gh['total_chars'] = chunk_map.get(cid, (0, 0))[1] or 0
                gh['edge_count'] = edge_map.get(cid, 0)

        # Also get chunk/edge info for parent headings (for context)
        if parent_headings:
            ph_ids = [ph['id'] for ph in parent_headings]
            ph_chunk_data = await conn.fetch("""
                SELECT condition_id, COUNT(*) as chunk_count,
                       SUM(LENGTH(chunk_text)) as total_chars
                FROM knowledge_chunks
                WHERE condition_id = ANY($1::int[])
                GROUP BY condition_id
            """, ph_ids)
            ph_chunk_map = {r['condition_id']: (r['chunk_count'], r['total_chars']) for r in ph_chunk_data}

            for ph in parent_headings:
                cid = ph['id']
                ph['chunk_count'] = ph_chunk_map.get(cid, (0, 0))[0]
                ph['total_chars'] = ph_chunk_map.get(cid, (0, 0))[1] or 0

        # ---- PRINT RESULTS ----

        print(f"\n{'=' * 90}")
        print(f"CATEGORY 1: PARENT HEADINGS WITH POPULATED CHILDREN ({len(parent_headings)})")
        print(f"{'=' * 90}")
        print(f"{'STG Code':<12} {'Name':<50} {'Children':<10} {'Chunks':<8} {'Chars':<10}")
        print(f"{'-'*12} {'-'*50} {'-'*10} {'-'*8} {'-'*10}")
        for ph in sorted(parent_headings, key=lambda x: x['stg_code']):
            print(f"{ph['stg_code']:<12} {ph['name'][:50]:<50} {ph['child_count']:<10} {ph['chunk_count']:<8} {ph['total_chars']:<10}")
            for child in ph['populated_children'][:5]:
                print(f"  -> {child}")
            if len(ph['populated_children']) > 5:
                print(f"  -> ... and {len(ph['populated_children']) - 5} more")

        print(f"\n{'=' * 90}")
        print(f"CATEGORY 2: GENUINELY HOLLOW ({len(genuinely_hollow)})")
        print(f"{'=' * 90}")

        # Split genuinely hollow into those with chunks and those without
        with_chunks = [g for g in genuinely_hollow if g['chunk_count'] > 0]
        without_chunks = [g for g in genuinely_hollow if g['chunk_count'] == 0]

        print(f"\n--- 2a: Genuinely hollow WITH knowledge_chunks ({len(with_chunks)}) ---")
        print(f"These have raw text that could be re-processed to fill description_text")
        print(f"{'STG Code':<12} {'Name':<45} {'Chunks':<8} {'Chars':<10} {'Edges':<8}")
        print(f"{'-'*12} {'-'*45} {'-'*8} {'-'*10} {'-'*8}")
        for g in sorted(with_chunks, key=lambda x: x['total_chars'], reverse=True):
            print(f"{g['stg_code']:<12} {g['name'][:45]:<45} {g['chunk_count']:<8} {g['total_chars']:<10} {g['edge_count']:<8}")

        print(f"\n--- 2b: Genuinely hollow WITHOUT knowledge_chunks ({len(without_chunks)}) ---")
        print(f"These are truly empty — no text, no chunks")
        print(f"{'STG Code':<12} {'Name':<55} {'Edges':<8}")
        print(f"{'-'*12} {'-'*55} {'-'*8}")
        for g in sorted(without_chunks, key=lambda x: x['stg_code']):
            print(f"{g['stg_code']:<12} {g['name'][:55]:<55} {g['edge_count']:<8}")

        # Summary
        print(f"\n{'=' * 90}")
        print("SUMMARY")
        print(f"{'=' * 90}")
        print(f"Total hollow conditions:              {len(hollow)}")
        print(f"  Parent headings (have children):    {len(parent_headings)}")
        print(f"  Genuinely hollow:                   {len(genuinely_hollow)}")
        print(f"    - With knowledge_chunks:          {len(with_chunks)}")
        print(f"    - Without knowledge_chunks:       {len(without_chunks)}")
        total_recoverable_chars = sum(g['total_chars'] for g in with_chunks)
        print(f"  Total recoverable text (chars):     {total_recoverable_chars:,}")
        with_edges = [g for g in genuinely_hollow if g['edge_count'] > 0]
        print(f"  Genuinely hollow with edges > 0:    {len(with_edges)} (partially populated)")
        zero_everything = [g for g in genuinely_hollow if g['edge_count'] == 0 and g['chunk_count'] == 0]
        print(f"  Completely empty (no chunks/edges): {len(zero_everything)}")

    finally:
        await conn.close()


if __name__ == '__main__':
    asyncio.run(main())
