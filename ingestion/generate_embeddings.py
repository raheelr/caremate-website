"""
Generate Voyage AI embeddings for all knowledge chunks.

Usage:
    python3 ingestion/generate_embeddings.py

Requires VOYAGE_API_KEY in .env.
Processes in batches of 64, skips chunks that already have embeddings.
"""

import os
import sys
import asyncio
import asyncpg
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from agents.embeddings import get_embeddings_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

import time

BATCH_SIZE = 8  # Small batches — Voyage free tier is 3 RPM / 10K TPM
DELAY_BETWEEN_BATCHES = 22  # seconds between batches to stay under 3 RPM


async def main():
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print("ERROR: VOYAGE_API_KEY not set in .env")
        print("Get a key at https://dash.voyageai.com/")
        sys.exit(1)

    pool = await asyncpg.create_pool(os.getenv("DATABASE_URL"), min_size=1, max_size=3)

    async with pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM knowledge_chunks")
        existing = await conn.fetchval(
            "SELECT COUNT(*) FROM knowledge_chunks WHERE embedding IS NOT NULL"
        )
        remaining = total - existing
        print(f"Knowledge chunks: {total} total, {existing} with embeddings, {remaining} to process")

        if remaining == 0:
            print("All chunks already have embeddings!")
            await pool.close()
            return

        # Fetch chunks without embeddings
        rows = await conn.fetch("""
            SELECT id, chunk_text, section_role, condition_id
            FROM knowledge_chunks
            WHERE embedding IS NULL
            ORDER BY id
        """)

    processed = 0
    errors = 0

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        texts = [
            f"[{r['section_role']}] {r['chunk_text'][:2000]}"
            for r in batch
        ]

        embeddings = await get_embeddings_batch(texts, input_type="document")
        if embeddings is None:
            logger.error(f"Batch {i // BATCH_SIZE + 1} failed — stopping")
            errors += 1
            break

        # Write embeddings to DB
        async with pool.acquire() as conn:
            for j, (row, emb) in enumerate(zip(batch, embeddings)):
                # Convert to pgvector format
                vec_str = "[" + ",".join(str(x) for x in emb) + "]"
                await conn.execute(
                    "UPDATE knowledge_chunks SET embedding = $1 WHERE id = $2",
                    vec_str, row["id"]
                )
                processed += 1

        total_batches = (remaining + BATCH_SIZE - 1) // BATCH_SIZE
        batch_num = i // BATCH_SIZE + 1
        logger.info(f"Batch {batch_num}/{total_batches}: {processed}/{remaining} processed")

        # Rate limit delay (skip after last batch)
        if batch_num < total_batches:
            time.sleep(DELAY_BETWEEN_BATCHES)

    print(f"\n{'='*60}")
    print(f"EMBEDDING GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Processed: {processed}")
    print(f"Errors: {errors}")
    print(f"{'='*60}")

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
