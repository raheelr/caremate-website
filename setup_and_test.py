#!/usr/bin/env python3
"""
CareMate Setup & Test Script
-----------------------------
Run this first to verify your environment is ready.
Then run the pipeline.

Usage:
    python setup_and_test.py
"""

import os
import sys
import subprocess
import asyncio


def check_python():
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print(f"❌ Python 3.11+ required. You have {version.major}.{version.minor}")
        print("   Download from: https://www.python.org/downloads/")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_env_file():
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            import shutil
            shutil.copy(".env.example", ".env")
            print("⚠️  Created .env from template — please fill in your values")
            print("   Edit .env and add your ANTHROPIC_API_KEY and DATABASE_URL")
            return False
        else:
            print("❌ No .env file found")
            return False
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    db_url = os.getenv("DATABASE_URL", "")
    
    if not api_key or api_key == "sk-ant-your-key-here":
        print("❌ ANTHROPIC_API_KEY not set in .env")
        return False
    print(f"✅ Anthropic API key: {api_key[:12]}...")
    
    if not db_url or db_url == "postgresql://your-connection-string-here":
        print("❌ DATABASE_URL not set in .env")
        return False
    print(f"✅ Database URL: {db_url[:30]}...")
    
    return True


def install_packages():
    print("\nInstalling packages...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"❌ Package install failed:\n{result.stderr}")
        return False
    print("✅ All packages installed")
    return True


async def test_database():
    print("\nTesting database connection...")
    try:
        import asyncpg
        from dotenv import load_dotenv
        load_dotenv()
        
        conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
        
        # Check pgvector
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            print("✅ Database connected, pgvector enabled")
        except Exception as e:
            print(f"⚠️  Database connected but pgvector issue: {e}")
            print("   This is fine for Railway — pgvector is pre-installed")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\n   Make sure your DATABASE_URL in .env is correct")
        print("   It should look like:")
        print("   postgresql://postgres:password@host.railway.app:5432/railway")
        return False


def test_anthropic():
    print("\nTesting Anthropic API...")
    try:
        import anthropic
        from dotenv import load_dotenv
        load_dotenv()
        
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            messages=[{"role": "user", "content": "Reply with just: OK"}]
        )
        print(f"✅ Anthropic API working: {response.content[0].text.strip()}")
        return True
    except Exception as e:
        print(f"❌ Anthropic API failed: {e}")
        return False


def test_pdf():
    print("\nChecking STG PDF...")
    pdf_paths = [
        # Common locations
        "Primary-Healthcare-Standard-Treatment-Guidelines-and-Essential-Medicines-List-8th-Edition-2024-Updated-December-2025__1_.pdf",
        "./stg.pdf",
        os.path.expanduser("~/Downloads/Primary-Healthcare-Standard-Treatment-Guidelines-and-Essential-Medicines-List-8th-Edition-2024-Updated-December-2025__1_.pdf"),
    ]
    
    for path in pdf_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✅ PDF found: {path} ({size_mb:.1f} MB)")
            return path
    
    print("❌ STG PDF not found")
    print("   Place the PDF in the same folder as this script")
    print("   Or update the pdf_path in pipeline.py")
    return None


async def run_all():
    print("=" * 55)
    print("CAREMATE SETUP CHECK")
    print("=" * 55)
    
    checks = []
    
    checks.append(check_python())
    
    if not install_packages():
        print("\n❌ Setup incomplete — fix package installation first")
        return
    
    if not check_env_file():
        print("\n❌ Setup incomplete — fill in your .env file first")
        print("\nYou need:")
        print("  1. Anthropic API key from https://console.anthropic.com/settings/keys")
        print("  2. Database URL from Railway (instructions in SETUP_GUIDE.md)")
        return
    
    api_ok = test_anthropic()
    db_ok = await test_database()
    pdf_path = test_pdf()
    
    print("\n" + "=" * 55)
    print("RESULTS")
    print("=" * 55)
    
    all_good = api_ok and db_ok and pdf_path
    
    if all_good:
        print("✅ Everything ready! Run the pipeline:")
        print()
        print("  Test mode first (5 conditions, ~2 min):")
        print("  python ingestion/pipeline.py --test")
        print()
        print("  Then full ingestion (444 conditions, ~45 min):")
        print("  python ingestion/pipeline.py")
    else:
        print("❌ Fix the issues above, then run this script again")
        if not api_ok:
            print("\n  → Get API key: https://console.anthropic.com/settings/keys")
        if not db_ok:
            print("\n  → Set up database: follow SETUP_GUIDE.md")
        if not pdf_path:
            print("\n  → Put the STG PDF in this folder")


if __name__ == "__main__":
    asyncio.run(run_all())
