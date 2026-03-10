"""
CareMate Ingestion Pipeline — Multi-Tool Orchestrator
-----------------------------------------------------
TWO-PHASE architecture for resilience:

  PHASE A: Extract (Claude API — expensive, ~$10-15)
    Stage 1: Page Classification (pdfplumber, all 747 pages)
    Stage 2: Docling Full Parse (AI tables, cached)
    Stage 3: Claude Vision (image-heavy pages)
    Stage 4: pdfplumber Segmentation (all 444 conditions)
    Stage 5: Multi-Source Merge (best-of-three per condition)
    Stage 6: Claude Extraction → save each to JSON on disk
             Resumable: skips conditions that already have a JSON file.

  PHASE B: Load to DB (free, fast, retryable)
    Stage 7: Wipe DB + batch-insert all JSON extractions

Usage:
  # Full run (both phases)
  python pipeline.py --pdf stg.pdf --full-reset

  # Resume extraction after interruption (skips existing JSONs)
  python pipeline.py --pdf stg.pdf --full-reset --resume

  # Re-run ONLY Phase B (all JSONs already on disk, just reload DB)
  python pipeline.py --pdf stg.pdf --db-load-only

  # Test on specific conditions
  python pipeline.py --pdf stg.pdf --stg-codes '1.2,4.7.1,12.1'
"""

import os
import sys
import json
import asyncio
import argparse
import time
import traceback
from datetime import datetime
from pathlib import Path

# Force unbuffered output so we can monitor progress
os.environ['PYTHONUNBUFFERED'] = '1'

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'db'))

from segmenter import STGSegmenter, ConditionSegment
from extractor import ConditionExtractor
from page_classifier import PageClassifier
from docling_extractor import DoclingExtractor
from vision_extractor import VisionExtractor
from multi_source_merger import MultiSourceMerger, MergedConditionInput
from database import (
    get_connection, create_schema,
    save_condition, start_ingestion_run, complete_ingestion_run
)


# Where extracted JSONs are saved (survives crashes)
EXTRACTIONS_DIR = "extractions"


def log(msg: str):
    """Print with flush to ensure output is visible immediately."""
    print(msg, flush=True)


# ── Quality report ───────────────────────────────────────────────────────────

class QualityReport:
    """Generates a quality report from saved extraction JSONs."""

    def __init__(self):
        self.results = []

    def add(self, stg_code: str, name: str, extraction: dict, error: str = None):
        ambiguity = extraction.get('ambiguity_flags', {}) if extraction else {}
        self.results.append({
            'stg_code': stg_code,
            'name': name,
            'status': 'error' if error else 'ok',
            'error': error,
            'ambiguity_score': ambiguity.get('ambiguity_score', 0),
            'ambiguity_notes': ambiguity.get('ambiguity_notes', ''),
            'needs_review': extraction.get('needs_review', False) if extraction else True,
            'feature_count': len(extraction.get('clinical_features', [])) if extraction else 0,
            'medicine_count': len(extraction.get('medicines', [])) if extraction else 0,
            'has_danger_signs': bool(extraction.get('danger_signs')) if extraction else False,
            'resolved_by_pass2': ambiguity.get('resolved_by_pass2', False),
            'primary_source': extraction.get('_primary_source', 'pdfplumber') if extraction else '',
            'has_tables': extraction.get('_has_tables', False) if extraction else False,
            'has_vision': extraction.get('_has_vision', False) if extraction else False,
        })

    def save(self, path: str = "ingestion_quality_report.json"):
        total = len(self.results)
        if total == 0:
            log("  No results to report.")
            return {}

        errors = sum(1 for r in self.results if r['status'] == 'error')
        needs_review = sum(1 for r in self.results if r['needs_review'])
        pass2_used = sum(1 for r in self.results if r['resolved_by_pass2'])
        no_features = sum(1 for r in self.results if r['feature_count'] == 0)
        no_medicines = sum(1 for r in self.results if r['medicine_count'] == 0)
        docling_primary = sum(1 for r in self.results if r.get('primary_source') == 'docling')
        with_tables = sum(1 for r in self.results if r.get('has_tables'))
        with_vision = sum(1 for r in self.results if r.get('has_vision'))

        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_conditions': total,
                'extraction_errors': errors,
                'needs_human_review': needs_review,
                'pass2_extended_thinking_used': pass2_used,
                'zero_clinical_features': no_features,
                'zero_medicines': no_medicines,
                'docling_primary_source': docling_primary,
                'with_table_content': with_tables,
                'with_vision_content': with_vision,
                'success_rate': f"{((total - errors) / total * 100):.1f}%"
            },
            'needs_review': [r for r in self.results if r['needs_review'] or r['status'] == 'error'],
            'all_results': self.results
        }

        with open(path, 'w') as f:
            json.dump(report, f, indent=2)

        summary_path = path.replace('.json', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("CAREMATE INGESTION QUALITY REPORT (MULTI-TOOL)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {report['generated_at']}\n\n")
            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            for k, v in report['summary'].items():
                f.write(f"  {k}: {v}\n")

            f.write(f"\n\nCONDITIONS NEEDING REVIEW ({len(report['needs_review'])})\n")
            f.write("-" * 40 + "\n")
            for r in report['needs_review']:
                f.write(f"\n  {r['stg_code']} {r['name']}\n")
                if r['status'] == 'error':
                    f.write(f"    ERROR: {r['error']}\n")
                else:
                    f.write(f"    Ambiguity score: {r['ambiguity_score']:.1f}\n")
                    if r['ambiguity_notes']:
                        f.write(f"    Notes: {r['ambiguity_notes']}\n")
                    f.write(f"    Features: {r['feature_count']}, Medicines: {r['medicine_count']}\n")

        log(f"  Quality report saved: {path}")
        return report


# ── Extraction file helpers ──────────────────────────────────────────────────

def extraction_path(stg_code: str) -> str:
    """Path to the JSON file for a condition's extraction."""
    safe_name = stg_code.replace('.', '_')
    return os.path.join(EXTRACTIONS_DIR, f"{safe_name}.json")


def extraction_exists(stg_code: str) -> bool:
    """Check if an extraction JSON already exists on disk and is not a failed extraction."""
    path = extraction_path(stg_code)
    if not os.path.exists(path):
        return False
    # Also verify it's valid JSON, not empty, and not a failed extraction
    try:
        with open(path) as f:
            data = json.load(f)
        return bool(data.get('stg_code')) and not data.get('_extraction_failed')
    except (json.JSONDecodeError, KeyError):
        return False


def save_extraction_to_disk(extraction: dict):
    """Save an extraction result to a JSON file."""
    os.makedirs(EXTRACTIONS_DIR, exist_ok=True)
    path = extraction_path(extraction['stg_code'])
    with open(path, 'w') as f:
        json.dump(extraction, f, indent=2)


def load_extraction_from_disk(stg_code: str) -> dict:
    """Load an extraction result from a JSON file."""
    path = extraction_path(stg_code)
    with open(path) as f:
        return json.load(f)


def list_all_extractions() -> list[str]:
    """List all STG codes that have extraction JSONs on disk."""
    if not os.path.exists(EXTRACTIONS_DIR):
        return []
    codes = []
    for fname in os.listdir(EXTRACTIONS_DIR):
        if fname.endswith('.json'):
            stg_code = fname[:-5].replace('_', '.')
            codes.append(stg_code)
    return sorted(codes)


# ── DB operations ────────────────────────────────────────────────────────────

TRUNCATE_TABLES = [
    "knowledge_chunks",
    "condition_medicines",
    "clinical_relationships",
    "condition_prerequisites",
    "synonym_rings",
    "complaint_condition_routes",
    "conditions",
    "clinical_entities",
    "medicines",
]


async def full_db_wipe(conn):
    """TRUNCATE all content tables for a complete clean slate."""
    log("\n  FULL DATABASE WIPE")
    log("  " + "-" * 40)
    for table in TRUNCATE_TABLES:
        try:
            await conn.execute(f"TRUNCATE {table} CASCADE")
            log(f"    Truncated: {table}")
        except Exception as e:
            if "does not exist" in str(e):
                log(f"    Skipped (not found): {table}")
            else:
                log(f"    Warning truncating {table}: {e}")
    log("  Database wiped — clean slate ready\n")


async def get_fresh_connection():
    """Get a new DB connection. Call this per-operation, not once at startup."""
    return await get_connection()


# ── PHASE A: Extract all conditions to disk ──────────────────────────────────

def run_phase_a(
    pdf_path: str,
    resume: bool = False,
    stg_codes: list[str] = None,
    full_reset: bool = False,
    test_mode: bool = False,
    skip_docling: bool = False,
    skip_vision: bool = False,
    docling_cache: str = "docling_cache.json",
):
    """
    Phase A: Extract all conditions using Claude and save to JSON files.
    No database needed. Fully resumable.
    """
    log("\n" + "=" * 60)
    log("PHASE A: EXTRACT TO DISK")
    log("=" * 60)

    # ── Stage 1: Page Classification ─────────────────────────────
    log("\nSTAGE 1: Classifying all pages...")
    classifier = PageClassifier(pdf_path)
    classifications = classifier.classify_all()
    log(f"  Initial vision candidates: {len(classifier.get_vision_pages())} pages\n")

    # ── Stage 2: Docling Full Parse ──────────────────────────────
    docling_pages = {}
    if not skip_docling:
        log("STAGE 2: Running Docling on full PDF...")
        docling = DoclingExtractor(pdf_path, cache_path=docling_cache)
        docling_pages = docling.extract_all_pages()
        log(f"  Docling produced {len(docling_pages)} pages with content\n")
        classifier.refine_with_docling(docling_pages)
    else:
        log("STAGE 2: Docling SKIPPED\n")

    # ── Stage 3: Claude Vision ───────────────────────────────────
    vision_pages = {}
    if not skip_vision:
        vision_page_nums = classifier.get_vision_pages()
        if vision_page_nums:
            log(f"STAGE 3: Running Claude Vision on {len(vision_page_nums)} pages...")
            vision_ext = VisionExtractor()
            vision_pages = vision_ext.extract_pages(pdf_path, vision_page_nums)
            vision_ext.print_cost_summary()
            log("")
        else:
            log("STAGE 3: No pages need Vision extraction\n")
    else:
        log("STAGE 3: Vision SKIPPED\n")

    # ── Stage 4: pdfplumber Segmentation ─────────────────────────
    log("STAGE 4: Segmenting PDF with pdfplumber...")
    segmenter = STGSegmenter(pdf_path)

    if full_reset:
        segments = segmenter.segment(no_filter=True)
        log(f"  All segments (no filter): {len(segments)} conditions")
    elif stg_codes:
        include_set = set(stg_codes)
        segments = segmenter.segment(include_codes=include_set)
        segments = [s for s in segments if s.stg_code in include_set]
        missing = include_set - {s.stg_code for s in segments}
        if missing:
            log(f"  WARNING: {len(missing)} codes not found: {sorted(missing)}")
        log(f"  TARGETED: {len(segments)} conditions")
    elif test_mode:
        test_codes = ["1.2", "2.9.1", "4.7.1", "17.3.4.2", "16.4.1"]
        segments = segmenter.segment(include_codes=set(test_codes))
        segments = [s for s in segments if s.stg_code in test_codes]
        log(f"  TEST MODE: {len(segments)} conditions")
    else:
        segments = segmenter.segment()
        log(f"  Filtered segments: {len(segments)} conditions")
    log("")

    # ── Stage 5: Multi-Source Merge ──────────────────────────────
    log("STAGE 5: Merging extraction sources...")
    merger = MultiSourceMerger(
        segments=segments,
        docling_pages=docling_pages,
        vision_pages=vision_pages,
    )
    merged_inputs = merger.merge_all()
    log("")

    # ── Stage 6: Claude Extraction → JSON on disk ────────────────
    # Figure out what needs extracting
    if resume:
        already_done = [m for m in merged_inputs if extraction_exists(m.stg_code)]
        to_extract = [m for m in merged_inputs if not extraction_exists(m.stg_code)]
        log(f"STAGE 6: Extracting conditions with Claude...")
        log(f"  Already on disk: {len(already_done)}")
        log(f"  To extract: {len(to_extract)}")
    else:
        # Clear old extractions for a fresh start
        if full_reset and os.path.exists(EXTRACTIONS_DIR):
            import shutil
            shutil.rmtree(EXTRACTIONS_DIR)
            log("  Cleared old extraction files")
        to_extract = merged_inputs
        log(f"STAGE 6: Extracting {len(to_extract)} conditions with Claude...")

    log(f"  Saving to: {EXTRACTIONS_DIR}/\n")

    extractor = ConditionExtractor()
    success_count = 0
    error_count = 0

    for i, merged in enumerate(to_extract):
        # Double-check (for resume safety)
        if extraction_exists(merged.stg_code):
            log(f"  [{i+1}/{len(to_extract)}] {merged.stg_code} — already on disk, skipping")
            success_count += 1
            continue

        src_tag = f"[{merged.primary_source}]"
        extra = []
        if merged.has_tables:
            extra.append("tables")
        if merged.has_vision:
            extra.append("vision")
        tag_str = f" +{'+'.join(extra)}" if extra else ""

        log(f"  [{i+1}/{len(to_extract)}] {merged.display_name} {src_tag}{tag_str}")

        try:
            extraction = extractor.extract(merged)

            # Attach metadata
            extraction['sections'] = merged.sections
            if merged.has_tables and merged.tables_markdown:
                extraction['sections']['_tables'] = merged.tables_markdown
            if merged.has_vision and merged.vision_content:
                extraction['sections']['_vision'] = merged.vision_content

            # Check ambiguity
            ambiguity_score = extraction.get('ambiguity_flags', {}).get('ambiguity_score', 0)
            if ambiguity_score > 0.7 and not extraction.get('ambiguity_flags', {}).get('resolved_by_pass2'):
                extraction['needs_review'] = True

            # Save to disk — this is the critical save that protects our $$$
            save_extraction_to_disk(extraction)

            features = len(extraction.get('clinical_features', []))
            meds = len(extraction.get('medicines', []))
            review_tag = ", REVIEW" if extraction.get('needs_review') else ""
            log(f"    Saved to disk (features={features}, meds={meds}{review_tag})")

            success_count += 1

        except Exception as e:
            error_msg = str(e)
            log(f"    FAILED: {error_msg}")
            if "overloaded" not in error_msg.lower():
                traceback.print_exc()
            error_count += 1

            # For API overload, wait and continue (don't give up)
            if "overloaded" in error_msg.lower():
                log(f"    Waiting 30s before continuing...")
                time.sleep(30)

    extractor.print_cost_summary()

    log(f"\n{'='*60}")
    log(f"PHASE A COMPLETE")
    log(f"{'='*60}")
    log(f"  Extracted: {success_count}/{len(to_extract)}")
    log(f"  Failed:    {error_count}")
    total_on_disk = len(list_all_extractions())
    log(f"  Total JSONs on disk: {total_on_disk}")

    if error_count > 0:
        log(f"\n  To retry failed conditions, run again with --resume")

    # ── Manifest verification: check every segmented condition has an extraction ──
    expected_codes = {m.stg_code for m in merged_inputs}
    on_disk_codes = set(list_all_extractions())
    # Also filter out failed extractions from on_disk_codes
    valid_codes = set()
    for code in on_disk_codes:
        if extraction_exists(code):
            valid_codes.add(code)
    missing_codes = expected_codes - valid_codes
    if missing_codes:
        log(f"\n  WARNING: {len(missing_codes)} conditions segmented but NOT extracted:")
        for code in sorted(missing_codes):
            log(f"    - {code}")
        log(f"\n  To extract just these, run:")
        log(f"    python3 ingestion/pipeline.py --pdf {pdf_path} --stg-codes '{','.join(sorted(missing_codes))}' --skip-docling --skip-vision")
    else:
        log(f"  Manifest check: ALL {len(expected_codes)} segmented conditions have valid extractions")

    return total_on_disk


# ── PHASE B: Load all JSONs into database ────────────────────────────────────

async def run_phase_b(full_reset: bool = False, stg_codes: list[str] = None):
    """
    Phase B: Read all extraction JSONs from disk and load into DB.
    Fast (~2 min), free, infinitely retryable.
    """
    log("\n" + "=" * 60)
    log("PHASE B: LOAD TO DATABASE")
    log("=" * 60)

    # Find all extraction JSONs
    all_codes = list_all_extractions()
    if stg_codes:
        stg_set = set(stg_codes)
        all_codes = [c for c in all_codes if c in stg_set]

    log(f"  Found {len(all_codes)} extraction JSONs on disk")

    if not all_codes:
        log("  Nothing to load. Run Phase A first.")
        return

    # Connect to DB
    log("  Connecting to database...")
    conn = await get_fresh_connection()
    await create_schema(conn)
    log("  Database ready")

    # Wipe if full reset
    if full_reset:
        await full_db_wipe(conn)

    run_id = await start_ingestion_run(conn, "phase_b_load")

    # Load each extraction
    success_count = 0
    error_count = 0
    quality = QualityReport()

    for i, stg_code in enumerate(all_codes):
        try:
            extraction = load_extraction_from_disk(stg_code)
            name = extraction.get('condition_name_normalised', stg_code)

            # Reconnect if needed (fresh connection every 50 conditions)
            if i > 0 and i % 50 == 0:
                log(f"  [{i}/{len(all_codes)}] Refreshing DB connection...")
                try:
                    await conn.close()
                except Exception:
                    pass
                conn = await get_fresh_connection()

            condition_id = await save_condition(conn, extraction)
            success_count += 1
            quality.add(stg_code, name, extraction)

            if i % 20 == 0 or i == len(all_codes) - 1:
                log(f"  [{i+1}/{len(all_codes)}] Loaded {stg_code} {name} (id={condition_id})")

        except Exception as e:
            error_msg = str(e)
            log(f"  [{i+1}/{len(all_codes)}] FAILED {stg_code}: {error_msg}")
            error_count += 1
            quality.add(stg_code, stg_code, None, error=error_msg)

            # If connection died, reconnect
            if "closed" in error_msg.lower() or "connection" in error_msg.lower():
                log(f"    Reconnecting...")
                try:
                    conn = await get_fresh_connection()
                    # Retry this one
                    extraction = load_extraction_from_disk(stg_code)
                    condition_id = await save_condition(conn, extraction)
                    success_count += 1
                    error_count -= 1  # Undo the error count
                    log(f"    Retry succeeded (id={condition_id})")
                except Exception as retry_err:
                    log(f"    Retry also failed: {retry_err}")

    # Finalise
    try:
        await complete_ingestion_run(conn, run_id, success_count, 0)
    except Exception:
        pass
    try:
        await conn.close()
    except Exception:
        pass

    # Quality report
    log("\nGenerating quality report...")
    quality.save("ingestion_quality_report.json")

    log(f"\n{'='*60}")
    log(f"PHASE B COMPLETE")
    log(f"{'='*60}")
    log(f"  Loaded: {success_count}/{len(all_codes)} conditions")
    log(f"  Failed: {error_count}")

    if error_count > 0:
        log(f"\n  To retry, just run Phase B again (--db-load-only)")


# ── Main orchestrator ────────────────────────────────────────────────────────

async def run_pipeline(
    pdf_path: str,
    resume: bool = False,
    test_mode: bool = False,
    stg_codes: list[str] = None,
    force: bool = False,
    full_reset: bool = False,
    skip_docling: bool = False,
    skip_vision: bool = False,
    docling_cache: str = "docling_cache.json",
    db_load_only: bool = False,
):
    log("\n" + "=" * 60)
    log("CAREMATE STG INGESTION PIPELINE (MULTI-TOOL)")
    log("=" * 60)
    log(f"PDF: {pdf_path}")
    if db_load_only:
        log("Mode: DB LOAD ONLY (Phase B — load JSONs to DB)")
    elif full_reset:
        log("Mode: FULL RESET (Phase A + B)")
    elif stg_codes:
        log(f"Mode: TARGETED ({len(stg_codes)} conditions)")
    elif test_mode:
        log(f"Mode: TEST")
    else:
        log("Mode: FULL")
    log(f"Resume: {resume}")
    if not db_load_only:
        log(f"Docling: {'SKIP' if skip_docling else 'enabled'}")
        log(f"Vision: {'SKIP' if skip_vision else 'enabled'}")
    log("")

    if not db_load_only:
        # PHASE A: Extract to disk (no DB needed)
        total_on_disk = run_phase_a(
            pdf_path=pdf_path,
            resume=resume,
            stg_codes=stg_codes,
            full_reset=full_reset,
            test_mode=test_mode,
            skip_docling=skip_docling,
            skip_vision=skip_vision,
            docling_cache=docling_cache,
        )

        if total_on_disk == 0:
            log("\nNo extractions to load. Exiting.")
            return

    # PHASE B: Load to DB
    await run_phase_b(
        full_reset=full_reset,
        stg_codes=stg_codes,
    )

    log(f"\n{'='*60}")
    log("ALL DONE")
    log(f"{'='*60}")
    log(f"Next steps:")
    log(f"  1. python3 ingestion/verify_reextraction.py  # check quality")
    log(f"  2. python3 ingestion/enrich.py --apply        # enrichment")
    log(f"  3. python3 ingestion/generate_embeddings.py   # vector search")
    log(f"  4. python3 ingestion/qa_validator.py           # QA vignettes")


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CareMate STG Ingestion Pipeline (Multi-Tool)")
    parser.add_argument("--pdf", default="stg.pdf", help="Path to the STG PDF")
    parser.add_argument("--resume", action="store_true",
                        help="Resume: skip conditions that already have JSON on disk")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: only process 5 sample conditions")
    parser.add_argument("--stg-codes", type=str, default=None,
                        help="Comma-separated STG codes (e.g. '1.2,4.7.1,12.1')")
    parser.add_argument("--force", action="store_true",
                        help="Force re-extraction of targeted conditions")
    parser.add_argument("--full-reset", action="store_true",
                        help="FULL RESET: extract everything + wipe DB + reload")
    parser.add_argument("--skip-docling", action="store_true",
                        help="Skip Docling extraction (debug only)")
    parser.add_argument("--skip-vision", action="store_true",
                        help="Skip Claude Vision extraction (debug only)")
    parser.add_argument("--docling-cache", type=str, default="docling_cache.json",
                        help="Docling cache file path")
    parser.add_argument("--db-load-only", action="store_true",
                        help="Skip extraction, only load existing JSONs to DB (Phase B only)")

    args = parser.parse_args()

    stg_code_list = None
    if args.stg_codes:
        stg_code_list = [c.strip() for c in args.stg_codes.split(",") if c.strip()]

    asyncio.run(run_pipeline(
        pdf_path=args.pdf,
        resume=args.resume,
        test_mode=args.test,
        stg_codes=stg_code_list,
        force=args.force,
        full_reset=args.full_reset,
        skip_docling=args.skip_docling,
        skip_vision=args.skip_vision,
        docling_cache=args.docling_cache,
        db_load_only=args.db_load_only,
    ))
