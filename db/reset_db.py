"""
Reset Database
--------------
Drops all CareMate tables and recreates them fresh.
Run this when you need a clean slate.

Usage:
    python3 db/reset_db.py
"""
import asyncio, asyncpg, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DROP_ALL = """
DROP TABLE IF EXISTS session_answers CASCADE;
DROP TABLE IF EXISTS clinical_sessions CASCADE;
DROP TABLE IF EXISTS complaint_condition_routes CASCADE;
DROP TABLE IF EXISTS chief_complaints CASCADE;
DROP TABLE IF EXISTS knowledge_chunks CASCADE;
DROP TABLE IF EXISTS condition_medicines CASCADE;
DROP TABLE IF EXISTS medicines CASCADE;
DROP TABLE IF EXISTS clinical_relationships CASCADE;
DROP TABLE IF EXISTS clinical_entities CASCADE;
DROP TABLE IF EXISTS synonym_rings CASCADE;
DROP TABLE IF EXISTS condition_prerequisites CASCADE;
DROP TABLE IF EXISTS conditions CASCADE;
DROP TABLE IF EXISTS triage_events CASCADE;
DROP TABLE IF EXISTS ingestion_progress CASCADE;
DROP TABLE IF EXISTS ingestion_runs CASCADE;
"""

async def reset():
    url = os.getenv("DATABASE_URL")
    if not url:
        print("❌ DATABASE_URL not set")
        return
    
    print("Connecting to database...")
    conn = await asyncpg.connect(url)
    
    print("Dropping all tables...")
    await conn.execute(DROP_ALL)
    print("✅ All tables dropped")
    
    print("Recreating schema...")
    schema_path = Path(__file__).parent / 'schema.sql'
    with open(schema_path) as f:
        schema = f.read()
    await conn.execute(schema)
    print("✅ Schema recreated with correct constraints")
    
    await conn.close()
    print("\nDatabase is ready. Run the pipeline:")
    print("  python3 ingestion/pipeline.py --test --pdf stg.pdf")

asyncio.run(reset())
