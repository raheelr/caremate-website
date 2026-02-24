"""
CareMate Ingestion Pipeline — Orchestrator
------------------------------------------
Runs the complete STG digitisation pipeline:

  1. Segment PDF → 444 condition blocks
  2. Extract each condition (Claude Haiku, fast pass)
  3. Re-extract ambiguous conditions (Claude Sonnet, extended thinking)
  4. Save everything to PostgreSQL
  5. Generate quality report

This is a ONE-TIME (or per-edition) operation.
Run it when you have a new STG edition.

Usage:
  python pipeline.py --pdf /path/to/stg.pdf
  python pipeline.py --pdf /path/to/stg.pdf --resume  (skip already-done conditions)
  python pipeline.py --pdf /path/to/stg.pdf --test     (only process 5 conditions)
"""

import os
import sys
import json
import asyncio
import argparse
import traceback
from datetime import datetime
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # If no dotenv, environment variables must be set manually

sys.path.insert(0, str(Path(__file__).parent))

from segmenter import STGSegmenter, ConditionSegment
from extractor import ConditionExtractor
sys.path.insert(0, str(Path(__file__).parent.parent / 'db'))
from database import (
    get_connection, create_schema,
    save_condition, start_ingestion_run, complete_ingestion_run
)


# ── Progress tracking ────────────────────────────────────────────────────────

class ProgressTracker:
    """Tracks progress so we can resume if something goes wrong."""
    
    def __init__(self, resume_file: str = "ingestion_progress.json"):
        self.resume_file = resume_file
        self.completed = set()
        self.failed = {}
        
        if os.path.exists(resume_file):
            with open(resume_file) as f:
                data = json.load(f)
                self.completed = set(data.get('completed', []))
                self.failed = data.get('failed', {})
            print(f"Resuming: {len(self.completed)} already done, {len(self.failed)} failed")
    
    def mark_complete(self, stg_code: str):
        self.completed.add(stg_code)
        self._save()
    
    def mark_failed(self, stg_code: str, error: str):
        self.failed[stg_code] = error
        self._save()
    
    def is_done(self, stg_code: str) -> bool:
        return stg_code in self.completed
    
    def _save(self):
        with open(self.resume_file, 'w') as f:
            json.dump({
                'completed': list(self.completed),
                'failed': self.failed,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)


# ── Quality report ───────────────────────────────────────────────────────────

class QualityReport:
    """Generates a human-readable quality report after ingestion."""
    
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
        })
    
    def save(self, path: str = "ingestion_quality_report.json"):
        # Summary stats
        total = len(self.results)
        errors = sum(1 for r in self.results if r['status'] == 'error')
        needs_review = sum(1 for r in self.results if r['needs_review'])
        pass2_used = sum(1 for r in self.results if r['resolved_by_pass2'])
        no_features = sum(1 for r in self.results if r['feature_count'] == 0)
        no_medicines = sum(1 for r in self.results if r['medicine_count'] == 0)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_conditions': total,
                'extraction_errors': errors,
                'needs_human_review': needs_review,
                'pass2_extended_thinking_used': pass2_used,
                'zero_clinical_features': no_features,
                'zero_medicines': no_medicines,
                'success_rate': f"{((total - errors) / total * 100):.1f}%" if total else "0%"
            },
            'needs_review': [r for r in self.results if r['needs_review'] or r['status'] == 'error'],
            'all_results': self.results
        }
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also write a human-readable summary
        summary_path = path.replace('.json', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("CAREMATE INGESTION QUALITY REPORT\n")
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
                    f.write(f"    ❌ ERROR: {r['error']}\n")
                else:
                    f.write(f"    ⚠️  Ambiguity score: {r['ambiguity_score']:.1f}\n")
                    if r['ambiguity_notes']:
                        f.write(f"    Notes: {r['ambiguity_notes']}\n")
                    f.write(f"    Features: {r['feature_count']}, Medicines: {r['medicine_count']}\n")
        
        print(f"\n📋 Quality report saved:")
        print(f"   {path}")
        print(f"   {summary_path}")
        
        return report


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run_pipeline(
    pdf_path: str,
    resume: bool = False,
    test_mode: bool = False,
    test_limit: int = 5
):
    """
    Full ingestion pipeline.
    
    Args:
        pdf_path: Path to the STG PDF
        resume: Skip already-processed conditions
        test_mode: Only process a few conditions (for testing)
        test_limit: Number of conditions to process in test mode
    """
    
    print("\n" + "=" * 60)
    print("CAREMATE STG INGESTION PIPELINE")
    print("=" * 60)
    print(f"PDF: {pdf_path}")
    print(f"Mode: {'TEST (' + str(test_limit) + ' conditions)' if test_mode else 'FULL'}")
    print(f"Resume: {resume}")
    print()
    
    # ── Step 1: Segment PDF ───────────────────────────────────────
    print("STEP 1: Segmenting PDF...")
    segmenter = STGSegmenter(pdf_path)
    segments = segmenter.segment()
    print(f"  Found {len(segments)} conditions\n")
    
    if test_mode:
        # Pick a diverse sample for testing
        test_codes = ["1.2", "2.9.1", "4.7.1", "17.3.4.2", "16.4.1"]
        segments = [s for s in segments if s.stg_code in test_codes]
        if not segments:
            segments = segments[:test_limit]
        print(f"  TEST MODE: Processing {len(segments)} conditions\n")
    
    # ── Step 2: Connect to database ───────────────────────────────
    print("STEP 2: Connecting to database...")
    try:
        conn = await get_connection()
        await create_schema(conn)
        print("  ✅ Database ready\n")
    except Exception as e:
        print(f"  ❌ Database connection failed: {e}")
        print("\n  To run the pipeline, you need a PostgreSQL database.")
        print("  Add DATABASE_URL to your .env file and try again.")
        print("\n  For now, we'll run in DRY RUN mode (extraction only, no DB save)")
        conn = None
    
    # ── Step 3: Set up tracking ───────────────────────────────────
    tracker = ProgressTracker() if resume else ProgressTracker.__new__(ProgressTracker)
    if not resume:
        tracker.resume_file = "ingestion_progress.json"
        tracker.completed = set()
        tracker.failed = {}
    
    quality = QualityReport()
    extractor = ConditionExtractor()
    
    run_id = None
    if conn:
        run_id = await start_ingestion_run(conn, pdf_path)
    
    # ── Step 4: Extract each condition ────────────────────────────
    print(f"STEP 3: Extracting {len(segments)} conditions...")
    print("  (This will take several minutes)\n")
    
    success_count = 0
    review_count = 0
    
    for i, segment in enumerate(segments):
        
        # Skip if already done (resume mode)
        if resume and tracker.is_done(segment.stg_code):
            print(f"  [{i+1}/{len(segments)}] ⏭️  Skipping {segment.stg_code} (already done)")
            continue
        
        print(f"  [{i+1}/{len(segments)}] {segment.display_name}")
        
        try:
            # Extract
            extraction = extractor.extract(segment)
            
            # Attach section text for saving
            extraction['sections'] = segment.sections
            
            needs_review = extraction.get('needs_review', False)
            ambiguity_score = extraction.get('ambiguity_flags', {}).get('ambiguity_score', 0)
            
            # Flag for review if high ambiguity remained after Pass 2
            if ambiguity_score > 0.7 and not extraction.get('ambiguity_flags', {}).get('resolved_by_pass2'):
                needs_review = True
                extraction['needs_review'] = True
            
            # Save to database
            if conn:
                condition_id = await save_condition(conn, extraction)
                print(f"    ✅ Saved (id={condition_id}, "
                      f"features={len(extraction.get('clinical_features', []))}, "
                      f"meds={len(extraction.get('medicines', []))}"
                      f"{', ⚠️ REVIEW' if needs_review else ''})")
            else:
                print(f"    📝 Extracted (dry run, "
                      f"features={len(extraction.get('clinical_features', []))}, "
                      f"meds={len(extraction.get('medicines', []))}"
                      f"{', ⚠️ REVIEW' if needs_review else ''})")
                
                # Save to JSON in dry run mode
                dry_run_path = f"dry_run_extractions/{segment.stg_code.replace('.', '_')}.json"
                os.makedirs("dry_run_extractions", exist_ok=True)
                with open(dry_run_path, 'w') as f:
                    # Don't save raw_text to keep files manageable
                    out = {k: v for k, v in extraction.items() if k != 'raw_text'}
                    json.dump(out, f, indent=2)
            
            quality.add(segment.stg_code, segment.name, extraction)
            tracker.mark_complete(segment.stg_code)
            
            success_count += 1
            if needs_review:
                review_count += 1
                
        except Exception as e:
            error_msg = str(e)
            print(f"    ❌ FAILED: {error_msg}")
            traceback.print_exc()
            quality.add(segment.stg_code, segment.name, None, error=error_msg)
            tracker.mark_failed(segment.stg_code, error_msg)
    
    # ── Step 5: Finalise ──────────────────────────────────────────
    print(f"\nSTEP 4: Finalising...")
    
    if conn and run_id:
        await complete_ingestion_run(conn, run_id, success_count, review_count)
        await conn.close()
    
    # ── Step 6: Quality report ────────────────────────────────────
    print("\nSTEP 5: Generating quality report...")
    report = quality.save("ingestion_quality_report.json")
    
    extractor.print_cost_summary()
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Extracted:    {success_count}/{len(segments)} conditions")
    print(f"⚠️  Need review: {review_count} conditions")
    print(f"❌ Failed:       {len(extractor.pass1_count and [] or [])} conditions")
    
    if review_count > 0:
        print(f"\nNext step: Review flagged conditions in ingestion_quality_report_summary.txt")
    
    return report


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CareMate STG Ingestion Pipeline")
    parser.add_argument(
        "--pdf",
        default="/mnt/user-data/uploads/Primary-Healthcare-Standard-Treatment-Guidelines-and-Essential-Medicines-List-8th-Edition-2024-Updated-December-2025__1_.pdf",
        help="Path to the STG PDF"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-processed conditions"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only process 5 sample conditions"
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_pipeline(
        pdf_path=args.pdf,
        resume=args.resume,
        test_mode=args.test,
    ))
