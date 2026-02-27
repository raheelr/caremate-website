"""Check which of the 54 filtered conditions are real parents vs incorrectly filtered."""
import asyncio, asyncpg, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

FILTERED = [
    '1.3','10.15','10.16','10.3','10.4','10.6','10.9','11.10','11.2','11.3',
    '11.3.13','11.3.14','11.3.16','11.3.6','11.3.8','11.3.9','11.8','11.8.3','11.8.4',
    '11.8.6','12.1','17.3.4.2','2.11','2.5','2.9','3.5','4.5','4.6','4.8','5.10','5.10.6',
    '5.14','5.15','5.18','5.4','5.5','5.6','5.7','5.8','6.1','6.1.1','6.14','6.4','6.4.5',
    '6.4.7','6.6','6.6.4','6.7','7.2','9.4','9.5','9.5.3','9.6','9.7'
]

async def main():
    conn = await asyncpg.connect(os.getenv('DATABASE_URL'))

    parents_with_content_children = []
    parents_without_content_children = []
    leaf_nodes = []

    for code in sorted(FILTERED):
        children = await conn.fetch(
            "SELECT stg_code, name FROM conditions WHERE stg_code LIKE $1 AND stg_code != $2 ORDER BY stg_code",
            code + '.%', code
        )

        if children:
            child_with_chunks = 0
            for child in children:
                has = await conn.fetchval(
                    'SELECT COUNT(*) FROM knowledge_chunks WHERE condition_id = (SELECT id FROM conditions WHERE stg_code = $1)',
                    child['stg_code']
                )
                if has > 0:
                    child_with_chunks += 1

            child_codes = [r['stg_code'] for r in children]
            if child_with_chunks > 0:
                parents_with_content_children.append((code, len(children), child_with_chunks, child_codes[:5]))
            else:
                parents_without_content_children.append((code, len(children), child_with_chunks, child_codes[:5]))
        else:
            leaf_nodes.append(code)

    print(f"\n=== LEAF NODES (no children — should NOT have been filtered) ===")
    print(f"Count: {len(leaf_nodes)}")
    for code in leaf_nodes:
        name = await conn.fetchval("SELECT name FROM conditions WHERE stg_code = $1", code)
        print(f"  {code:12s} {name}")

    print(f"\n=== PARENTS where children ALSO have no content ===")
    print(f"Count: {len(parents_without_content_children)}")
    for code, n_children, n_with, child_codes in parents_without_content_children:
        name = await conn.fetchval("SELECT name FROM conditions WHERE stg_code = $1", code)
        print(f"  {code:12s} {name[:50]:50s} children={n_children} w/content={n_with}")

    print(f"\n=== PARENTS where children DO have content (OK to skip) ===")
    print(f"Count: {len(parents_with_content_children)}")
    for code, n_children, n_with, child_codes in parents_with_content_children:
        name = await conn.fetchval("SELECT name FROM conditions WHERE stg_code = $1", code)
        print(f"  {code:12s} {name[:50]:50s} children={n_children} w/content={n_with}")

    await conn.close()

asyncio.run(main())
