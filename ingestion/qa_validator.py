"""
Automated QA Validator
-----------------------
Runs after ingestion to automatically verify extraction quality.
Two types of checks:

1. DATA QUALITY CHECKS — runs on every condition in the DB:
   - No <UNKNOWN> doses
   - No duplicate features
   - Minimum feature counts
   - Referral criteria present
   - CAUTION box content captured for high-risk conditions

2. CLINICAL VIGNETTES — 10 test cases with known correct outputs:
   - Known presentation → should match expected condition
   - Known RED FLAG symptoms → should be flagged correctly
   - Known medicines → should be present with correct doses

Usage:
    python3 ingestion/qa_validator.py
    python3 ingestion/qa_validator.py --vignettes-only
    python3 ingestion/qa_validator.py --quality-only
"""

import asyncio
import asyncpg
import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────
# CLINICAL VIGNETTES
# Each vignette is a known clinical scenario with expected outputs.
# These act as unit tests for the knowledge graph.
# ─────────────────────────────────────────────────────────────

CLINICAL_VIGNETTES = [
    {
        "id": "V01",
        "name": "Oral Thrush — straightforward presentation",
        "stg_code": "1.2",
        "condition_name_contains": "candidiasis",
        "presenting_symptoms": ["white patches", "painful mouth"],
        "expected_medicine": "nystatin",
        "expected_medicine_line": "first_line",
        "expected_dose_contains": "100",        # 100,000 IU/mL
        "expected_referral_trigger": None,       # No emergency referral
        "red_flag_symptom": "pain and difficulty in swallowing in hiv",
        "notes": "PLHIV symptom should be RED_FLAG, not standard feature"
    },
    {
        "id": "V02",
        "name": "Depression — first-line medicine",
        "stg_code": "16.4.1",
        "condition_name_contains": "depress",
        "presenting_symptoms": ["low mood", "loss of interest"],
        "expected_medicine": "fluoxetine",
        "expected_medicine_line": "first_line",
        "expected_dose_contains": "20",         # 20mg
        "expected_referral_trigger": "suicidal",
        "red_flag_symptom": "suicidal ideation",
        "notes": "Suicidal ideation must appear as RED_FLAG or referral criterion"
    },
    {
        "id": "V03",
        "name": "Depression — second-line if fluoxetine fails",
        "stg_code": "16.4.1",
        "condition_name_contains": "depress",
        "presenting_symptoms": ["insomnia", "fatigue"],
        "expected_medicine": "citalopram",
        "expected_medicine_line": "second_line",
        "expected_dose_contains": "10",         # 10mg initial dose
        "expected_referral_trigger": None,
        "red_flag_symptom": None,
        "notes": "Citalopram must be second_line with 10mg starting dose"
    },
    {
        "id": "V04",
        "name": "Hypertension — diagnostic criteria",
        "stg_code": "4.7.1",
        "condition_name_contains": "hypertension",
        "presenting_symptoms": ["elevated blood pressure"],
        "expected_medicine": "hydrochlorothiazide",
        "expected_medicine_line": "first_line",
        "expected_dose_contains": "12.5",       # 12.5mg starting dose
        "expected_referral_trigger": None,
        "red_flag_symptom": "severe headache",
        "notes": "Systolic ≥140 on 3 occasions must be diagnostic_feature"
    },
    {
        "id": "V05",
        "name": "Hypertension — amlodipine in regimen",
        "stg_code": "4.7.1",
        "condition_name_contains": "hypertension",
        "presenting_symptoms": ["high blood pressure"],
        "expected_medicine": "amlodipine",
        "expected_medicine_line": "first_line",
        "expected_dose_contains": "5",          # 5mg starting dose
        "expected_referral_trigger": None,
        "red_flag_symptom": None,
        "notes": "Amlodipine must be present with 5mg dose"
    },
    {
        "id": "V06",
        "name": "Diarrhoea in Children — ORS first",
        "stg_code": "2.9.1",
        "condition_name_contains": "diarrhoea",
        "presenting_symptoms": ["diarrhoea", "dehydration"],
        "expected_medicine": "oral rehydration solution",
        "expected_medicine_line": "first_line",
        "expected_dose_contains": None,
        "expected_referral_trigger": "dysentery",  # STG uses "dysentery" not "blood in stool"
        "red_flag_symptom": "blood in stool",
        "notes": "ORS must be first-line; dysentery must trigger referral (blood in stool = RED_FLAG)"
    },
    {
        "id": "V07",
        "name": "Diarrhoea in Children — zinc supplementation",
        "stg_code": "2.9.1",
        "condition_name_contains": "diarrhoea",
        "presenting_symptoms": ["loose stools"],
        "expected_medicine": "zinc",
        "expected_medicine_line": "first_line",
        "expected_dose_contains": None,
        "expected_referral_trigger": None,
        "red_flag_symptom": "severe dehydration",
        "notes": "Zinc must be present as adjunct/first-line treatment"
    },
    {
        "id": "V08",
        "name": "Oral Thrush — risk factor captured",
        "stg_code": "1.2",
        "condition_name_contains": "candidiasis",
        "presenting_symptoms": ["immunosuppression"],
        "expected_medicine": "nystatin",
        "expected_medicine_line": "first_line",
        "expected_dose_contains": None,
        "expected_referral_trigger": None,
        "red_flag_symptom": None,
        "check_risk_factor": "immunosuppression",
        "notes": "Immunosuppression should be captured as associated_feature risk factor"
    },
    {
        "id": "V09",
        "name": "Depression — amitriptyline caution",
        "stg_code": "16.4.1",
        "condition_name_contains": "depress",
        "presenting_symptoms": ["depression"],
        "expected_medicine": "amitriptyline",
        "expected_medicine_line": "second_line",
        "expected_dose_contains": "25",         # 25mg initial dose
        "expected_referral_trigger": None,
        "red_flag_symptom": None,
        "check_medicine_note": "fatal",         # "fatal in overdose" must be in special_notes
        "notes": "Amitriptyline fatal-in-overdose warning must be captured"
    },
    {
        "id": "V10",
        "name": "Hypertension — enalapril present",
        "stg_code": "4.7.1",
        "condition_name_contains": "hypertension",
        "presenting_symptoms": ["high blood pressure"],
        "expected_medicine": "enalapril",
        "expected_medicine_line": "first_line",
        "expected_dose_contains": "10",         # 10mg
        "expected_referral_trigger": None,
        "red_flag_symptom": None,
        "notes": "Enalapril (ACE inhibitor) must be first_line with 10mg dose"
    },
]


# ─────────────────────────────────────────────────────────────
# DATA QUALITY CHECKS
# ─────────────────────────────────────────────────────────────

async def run_quality_checks(conn: asyncpg.Connection) -> list[dict]:
    """Run automated data quality checks on all ingested conditions."""
    issues = []
    
    conditions = await conn.fetch(
        "SELECT id, stg_code, name, extraction_confidence FROM conditions ORDER BY stg_code"
    )
    
    if not conditions:
        return [{"severity": "CRITICAL", "check": "database", 
                 "message": "No conditions found in database — nothing was ingested"}]
    
    for c in conditions:
        cid = c['id']
        code = c['stg_code']
        name = c['name']
        
        # ── Check 1: Unknown doses ──────────────────────────────
        unknown_doses = await conn.fetch("""
            SELECT m.name, cm.dose_context
            FROM condition_medicines cm
            JOIN medicines m ON m.id = cm.medicine_id
            WHERE cm.condition_id = $1
              AND (cm.dose_context ILIKE '%<UNKNOWN>%' OR cm.dose_context IS NULL)
        """, cid)
        
        for row in unknown_doses:
            issues.append({
                "severity": "WARNING",
                "check": "unknown_dose",
                "stg_code": code,
                "condition": name,
                "message": f"Medicine '{row['name']}' has unknown/missing dose",
                "fix": "Likely in a treatment table — needs table extraction"
            })
        
        # ── Check 2: Duplicate features ─────────────────────────
        duplicates = await conn.fetch("""
            SELECT e.canonical_name, cr.relationship_type, COUNT(*) as cnt
            FROM clinical_relationships cr
            JOIN clinical_entities e ON e.id = cr.source_entity_id
            WHERE cr.condition_id = $1
            GROUP BY e.canonical_name, cr.relationship_type
            HAVING COUNT(*) > 1
        """, cid)
        
        for row in duplicates:
            issues.append({
                "severity": "WARNING",
                "check": "duplicate_feature",
                "stg_code": code,
                "condition": name,
                "message": f"Feature '{row['canonical_name']}' appears {row['cnt']}x as {row['relationship_type']}",
                "fix": "Schema UNIQUE index should prevent this after next ingestion"
            })
        
        # ── Check 3: Minimum feature count ──────────────────────
        feature_count = await conn.fetchval(
            "SELECT COUNT(*) FROM clinical_relationships WHERE condition_id = $1", cid
        )
        
        if feature_count == 0:
            issues.append({
                "severity": "ERROR",
                "check": "no_features",
                "stg_code": code,
                "condition": name,
                "message": "No clinical features extracted — condition is empty",
                "fix": "Likely a heading-only section (check if sub-sections exist)"
            })
        elif feature_count < 3:
            issues.append({
                "severity": "WARNING",
                "check": "low_feature_count",
                "stg_code": code,
                "condition": name,
                "message": f"Only {feature_count} features — may be incomplete",
                "fix": "Review source PDF for this condition"
            })
        
        # ── Check 4: No medicines ────────────────────────────────
        med_count = await conn.fetchval(
            "SELECT COUNT(*) FROM condition_medicines WHERE condition_id = $1", cid
        )
        
        if med_count == 0:
            issues.append({
                "severity": "INFO",
                "check": "no_medicines",
                "stg_code": code,
                "condition": name,
                "message": "No medicines — may be normal for some conditions (e.g. prevention/education sections)",
                "fix": "Verify against STG — some sections are management-only"
            })
        
        # ── Check 5: No referral criteria ───────────────────────
        referral = await conn.fetchval(
            "SELECT referral_criteria FROM conditions WHERE id = $1", cid
        )
        
        referral_list = json.loads(referral) if referral else []
        if not referral_list:
            issues.append({
                "severity": "INFO",
                "check": "no_referral_criteria",
                "stg_code": code,
                "condition": name,
                "message": "No referral criteria extracted",
                "fix": "Check if STG has REFERRAL section for this condition"
            })
        
        # ── Check 6: Low confidence ──────────────────────────────
        if c['extraction_confidence'] < 0.5:
            issues.append({
                "severity": "WARNING",
                "check": "low_confidence",
                "stg_code": code,
                "condition": name,
                "message": f"Low extraction confidence: {int(c['extraction_confidence']*100)}%",
                "fix": "Manually review this condition's extraction"
            })
    
    return issues


# ─────────────────────────────────────────────────────────────
# CLINICAL VIGNETTE TESTS  
# ─────────────────────────────────────────────────────────────

async def run_vignette_tests(conn: asyncpg.Connection) -> list[dict]:
    """Run all clinical vignettes as pass/fail tests."""
    results = []
    
    for vignette in CLINICAL_VIGNETTES:
        result = await _run_single_vignette(conn, vignette)
        results.append(result)
    
    return results


async def _run_single_vignette(conn: asyncpg.Connection, v: dict) -> dict:
    """Run a single vignette test. Returns pass/fail with details."""
    
    stg_code = v['stg_code']
    checks = []
    passed = 0
    failed = 0
    skipped = 0
    
    # Find the condition
    condition = await conn.fetchrow(
        "SELECT id, name FROM conditions WHERE stg_code = $1", stg_code
    )
    
    if not condition:
        return {
            "vignette_id": v['id'],
            "name": v['name'],
            "stg_code": stg_code,
            "status": "SKIPPED",
            "reason": f"Condition {stg_code} not yet ingested",
            "checks": []
        }
    
    cid = condition['id']
    
    # ── Check: Expected medicine exists ─────────────────────────
    if v.get('expected_medicine'):
        med = await conn.fetchrow("""
            SELECT m.name, cm.treatment_line, cm.dose_context, cm.special_notes
            FROM condition_medicines cm
            JOIN medicines m ON m.id = cm.medicine_id
            WHERE cm.condition_id = $1 AND m.name ILIKE $2
        """, cid, f"%{v['expected_medicine']}%")
        
        if med:
            check = {"check": f"medicine '{v['expected_medicine']}' exists", "status": "PASS"}
            passed += 1
            
            # ── Check: Treatment line ──────────────────────────
            if v.get('expected_medicine_line'):
                if med['treatment_line'] == v['expected_medicine_line']:
                    checks.append({"check": f"  └─ treatment line = {v['expected_medicine_line']}", "status": "PASS"})
                    passed += 1
                else:
                    checks.append({"check": f"  └─ treatment line = {v['expected_medicine_line']}", 
                                   "status": "FAIL",
                                   "detail": f"Got: {med['treatment_line']}"})
                    failed += 1
            
            # ── Check: Dose contains expected value ───────────
            if v.get('expected_dose_contains'):
                dose = med['dose_context'] or ''
                if v['expected_dose_contains'] in dose:
                    checks.append({"check": f"  └─ dose contains '{v['expected_dose_contains']}'", "status": "PASS"})
                    passed += 1
                else:
                    checks.append({"check": f"  └─ dose contains '{v['expected_dose_contains']}'",
                                   "status": "FAIL",
                                   "detail": f"Got dose: '{dose[:80]}'"})
                    failed += 1
            
            # ── Check: Medicine special notes ─────────────────
            if v.get('check_medicine_note'):
                notes = (med['special_notes'] or '').lower()
                if v['check_medicine_note'].lower() in notes:
                    checks.append({"check": f"  └─ special note contains '{v['check_medicine_note']}'", "status": "PASS"})
                    passed += 1
                else:
                    checks.append({"check": f"  └─ special note contains '{v['check_medicine_note']}'",
                                   "status": "FAIL",
                                   "detail": f"Notes: '{notes[:80]}' — CAUTION box may not have been extracted"})
                    failed += 1
            
            checks.insert(0, check)
        else:
            checks.append({"check": f"medicine '{v['expected_medicine']}' exists", 
                           "status": "FAIL",
                           "detail": "Not found in database"})
            failed += 1
    
    # ── Check: RED FLAG symptom ──────────────────────────────────
    if v.get('red_flag_symptom'):
        red_flag = await conn.fetchrow("""
            SELECT e.canonical_name, cr.relationship_type
            FROM clinical_relationships cr
            JOIN clinical_entities e ON e.id = cr.source_entity_id
            WHERE cr.condition_id = $1 
              AND cr.relationship_type = 'RED_FLAG'
              AND e.canonical_name ILIKE $2
        """, cid, f"%{v['red_flag_symptom']}%")
        
        if red_flag:
            checks.append({"check": f"RED FLAG: '{v['red_flag_symptom']}'", "status": "PASS"})
            passed += 1
        else:
            # Check if it exists but as wrong type
            any_flag = await conn.fetchrow("""
                SELECT e.canonical_name, cr.relationship_type
                FROM clinical_relationships cr
                JOIN clinical_entities e ON e.id = cr.source_entity_id
                WHERE cr.condition_id = $1 AND e.canonical_name ILIKE $2
            """, cid, f"%{v['red_flag_symptom']}%")
            
            if any_flag:
                checks.append({"check": f"RED FLAG: '{v['red_flag_symptom']}'",
                               "status": "FAIL",
                               "detail": f"Exists but typed as '{any_flag['relationship_type']}' not RED_FLAG"})
            else:
                checks.append({"check": f"RED FLAG: '{v['red_flag_symptom']}'",
                               "status": "FAIL",
                               "detail": "Not found at all — CAUTION/DANGER section may not be extracted"})
            failed += 1
    
    # ── Check: Referral criterion ────────────────────────────────
    if v.get('expected_referral_trigger'):
        referral_raw = await conn.fetchval(
            "SELECT referral_criteria FROM conditions WHERE id = $1", cid
        )
        referral_list = json.loads(referral_raw) if referral_raw else []
        referral_text = ' '.join(referral_list).lower()
        
        if v['expected_referral_trigger'].lower() in referral_text:
            checks.append({"check": f"referral criterion: '{v['expected_referral_trigger']}'", "status": "PASS"})
            passed += 1
        else:
            checks.append({"check": f"referral criterion: '{v['expected_referral_trigger']}'",
                           "status": "FAIL",
                           "detail": f"Not in referral list. Got: {referral_list[:3]}"})
            failed += 1
    
    # ── Check: Risk factor captured ──────────────────────────────
    if v.get('check_risk_factor'):
        risk = await conn.fetchrow("""
            SELECT e.canonical_name, cr.feature_type
            FROM clinical_relationships cr
            JOIN clinical_entities e ON e.id = cr.source_entity_id
            WHERE cr.condition_id = $1
              AND cr.feature_type = 'associated_feature'
              AND e.canonical_name ILIKE $2
        """, cid, f"%{v['check_risk_factor']}%")
        
        if risk:
            checks.append({"check": f"risk factor: '{v['check_risk_factor']}'", "status": "PASS"})
            passed += 1
        else:
            checks.append({"check": f"risk factor: '{v['check_risk_factor']}'",
                           "status": "FAIL",
                           "detail": "Not captured as associated_feature — improve extraction prompt"})
            failed += 1
    
    total = passed + failed
    status = "PASS" if failed == 0 and total > 0 else ("PARTIAL" if passed > 0 else "FAIL")
    
    return {
        "vignette_id": v['id'],
        "name": v['name'],
        "stg_code": stg_code,
        "condition_found": condition['name'],
        "status": status,
        "passed": passed,
        "failed": failed,
        "checks": checks,
        "notes": v.get('notes', '')
    }


# ─────────────────────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────────────────────

def print_quality_report(issues: list[dict]):
    print(f"\n{'='*60}")
    print("DATA QUALITY REPORT")
    print(f"{'='*60}")
    
    if not issues:
        print("✅ All quality checks passed — no issues found")
        return
    
    by_severity = {"CRITICAL": [], "ERROR": [], "WARNING": [], "INFO": []}
    for issue in issues:
        sev = issue.get("severity", "INFO")
        by_severity.setdefault(sev, []).append(issue)
    
    icons = {"CRITICAL": "🚨", "ERROR": "❌", "WARNING": "⚠️ ", "INFO": "ℹ️ "}
    
    for severity in ["CRITICAL", "ERROR", "WARNING", "INFO"]:
        group = by_severity.get(severity, [])
        if not group:
            continue
        print(f"\n{icons[severity]} {severity} ({len(group)} issues)")
        for issue in group:
            print(f"   [{issue.get('stg_code', '?')}] {issue.get('condition', '?')}")
            print(f"   → {issue['message']}")
            if issue.get('fix'):
                print(f"   Fix: {issue['fix']}")
    
    criticals = len(by_severity.get("CRITICAL", []))
    errors = len(by_severity.get("ERROR", []))
    warnings = len(by_severity.get("WARNING", []))
    
    print(f"\nSummary: {criticals} critical, {errors} errors, {warnings} warnings")
    if criticals + errors == 0:
        print("✅ No blocking issues — safe to proceed")
    else:
        print("❌ Blocking issues found — fix before full ingestion")


def print_vignette_report(results: list[dict]):
    print(f"\n{'='*60}")
    print("CLINICAL VIGNETTE TEST RESULTS")
    print(f"{'='*60}")
    
    icons = {"PASS": "✅", "FAIL": "❌", "PARTIAL": "⚠️ ", "SKIPPED": "⏭️ "}
    
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    
    for r in results:
        icon = icons.get(r['status'], "?")
        print(f"\n{icon} {r['vignette_id']}: {r['name']} (STG {r['stg_code']})")
        
        if r['status'] == 'SKIPPED':
            print(f"   ⏭️  {r.get('reason', 'Skipped')}")
            total_skipped += 1
            continue
        
        print(f"   Condition: {r.get('condition_found', '?')}")
        
        for check in r.get('checks', []):
            ci = "✅" if check['status'] == 'PASS' else "❌"
            print(f"   {ci} {check['check']}")
            if check.get('detail'):
                print(f"      ↳ {check['detail']}")
        
        if r['status'] == 'PASS':
            total_passed += 1
        elif r['status'] == 'PARTIAL':
            total_passed += 0.5
            total_failed += 0.5
        else:
            total_failed += 1
        
        if r.get('notes'):
            print(f"   📌 {r['notes']}")
    
    total = len([r for r in results if r['status'] != 'SKIPPED'])
    print(f"\n{'─'*60}")
    print(f"Results: {int(total_passed)}/{total} vignettes passed, {total_skipped} skipped (not yet ingested)")
    
    if total > 0:
        pct = int((total_passed / total) * 100)
        if pct >= 80:
            print(f"✅ {pct}% pass rate — extraction quality is GOOD")
        elif pct >= 60:
            print(f"⚠️  {pct}% pass rate — extraction quality NEEDS IMPROVEMENT")
        else:
            print(f"❌ {pct}% pass rate — extraction quality is POOR — fix before full run")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

async def main(args):
    url = os.getenv("DATABASE_URL")
    if not url:
        print("❌ DATABASE_URL not set")
        return
    
    conn = await asyncpg.connect(url)
    
    print("CareMate STG Ingestion — Automated QA Validator")
    print("=" * 60)
    
    # Overview
    total = await conn.fetchval("SELECT COUNT(*) FROM conditions")
    print(f"Conditions in database: {total}")
    
    if not args.vignettes_only:
        print("\nRunning data quality checks...")
        issues = await run_quality_checks(conn)
        print_quality_report(issues)
    
    if not args.quality_only:
        print("\nRunning clinical vignette tests...")
        vignette_results = await run_vignette_tests(conn)
        print_vignette_report(vignette_results)
        
        # Save vignette results to JSON for tracking over time
        output_path = Path(__file__).parent.parent / "qa_vignette_results.json"
        with open(output_path, 'w') as f:
            json.dump(vignette_results, f, indent=2)
        print(f"\nFull results saved: {output_path}")
    
    await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CareMate STG Ingestion QA Validator")
    parser.add_argument("--vignettes-only", action="store_true", help="Only run vignette tests")
    parser.add_argument("--quality-only", action="store_true", help="Only run quality checks")
    args = parser.parse_args()
    
    asyncio.run(main(args))
