"""
Audit enrichment coverage of the knowledge graph.
Reports conditions with zero enrichment edges and low total edge counts.
"""
import os
import sys
import asyncio
import asyncpg
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

FAILING_STGS = {
    "3.2": "Diarrhoea",
    "2.14": "Scabies",
    "10.4": "Malaria",
    "15.1": "Epilepsy",
    "20.2": "Pain",
    "15.4": "Headache",
    "12.1": "Vaginal Discharge",
    "6.15": "Vaginal Discharge (alt)",
    "21.3.7": "Soft Tissue",
}


async def main():
    pool = await asyncpg.create_pool(os.getenv("DATABASE_URL"), min_size=1, max_size=3)

    async with pool.acquire() as conn:
        # Total conditions
        total = await conn.fetchval("SELECT COUNT(*) FROM conditions")
        print(f"Total conditions: {total}")

        # Total relationships
        total_rels = await conn.fetchval("SELECT COUNT(*) FROM clinical_relationships")
        print(f"Total relationships: {total_rels}")

        # Enrichment edges
        enrich_count = await conn.fetchval(
            "SELECT COUNT(*) FROM clinical_relationships WHERE source_section = 'ENRICHMENT'"
        )
        print(f"Enrichment edges: {enrich_count}")

        # Conditions with enrichment edges
        enriched = await conn.fetchval(
            "SELECT COUNT(DISTINCT condition_id) FROM clinical_relationships WHERE source_section = 'ENRICHMENT'"
        )
        print(f"Conditions with enrichment: {enriched}/{total}")
        print(f"Conditions WITHOUT enrichment: {total - enriched}")

        # Conditions with < 3 total edges
        sparse = await conn.fetch("""
            SELECT c.stg_code, c.name, COUNT(cr.id) as edge_count
            FROM conditions c
            LEFT JOIN clinical_relationships cr ON cr.condition_id = c.id
            GROUP BY c.id, c.stg_code, c.name
            HAVING COUNT(cr.id) < 3
            ORDER BY COUNT(cr.id), c.stg_code
        """)
        print(f"\nConditions with < 3 edges: {len(sparse)}")
        for row in sparse[:20]:
            print(f"  {row['stg_code']}: {row['name']} ({row['edge_count']} edges)")
        if len(sparse) > 20:
            print(f"  ... and {len(sparse) - 20} more")

        # Check the 8 failing conditions specifically
        print(f"\n{'='*60}")
        print("FAILING CONDITIONS — Graph Coverage")
        print(f"{'='*60}")

        for stg_prefix, label in FAILING_STGS.items():
            rows = await conn.fetch("""
                SELECT c.stg_code, c.name,
                    COUNT(cr.id) as total_edges,
                    COUNT(cr.id) FILTER (WHERE cr.source_section = 'ENRICHMENT') as enrichment_edges,
                    COUNT(cr.id) FILTER (WHERE cr.relationship_type = 'INDICATES') as indicates,
                    COUNT(cr.id) FILTER (WHERE cr.relationship_type = 'RED_FLAG') as red_flags
                FROM conditions c
                LEFT JOIN clinical_relationships cr ON cr.condition_id = c.id
                WHERE c.stg_code LIKE $1
                GROUP BY c.id, c.stg_code, c.name
                ORDER BY c.stg_code
            """, f"{stg_prefix}%")

            for r in rows:
                enriched_mark = "✅" if r["enrichment_edges"] > 0 else "❌"
                print(
                    f"  {enriched_mark} {r['stg_code']}: {r['name']} — "
                    f"{r['total_edges']} edges ({r['enrichment_edges']} enrichment, "
                    f"{r['indicates']} INDICATES, {r['red_flags']} RED_FLAG)"
                )

                # Show actual features for failing conditions
                features = await conn.fetch("""
                    SELECT e.canonical_name, cr.feature_type, cr.source_section
                    FROM clinical_relationships cr
                    JOIN clinical_entities e ON e.id = cr.source_entity_id
                    WHERE cr.condition_id = (SELECT id FROM conditions WHERE stg_code = $1)
                    AND cr.relationship_type IN ('INDICATES', 'RED_FLAG')
                    ORDER BY cr.feature_type, e.canonical_name
                """, r["stg_code"])
                for f in features[:10]:
                    print(f"      {f['feature_type']:20s} | {f['canonical_name']} [{f['source_section']}]")
                if len(features) > 10:
                    print(f"      ... +{len(features) - 10} more")

        # Synonym rings stats
        syn_count = await conn.fetchval("SELECT COUNT(*) FROM synonym_rings")
        print(f"\nSynonym rings total: {syn_count}")

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
