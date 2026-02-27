"""
Comprehensive Vignette Test Suite
----------------------------------
25 clinical vignettes × 2-3 patient-language variations = ~65 test cases.

Tests:
1. Symptom extraction: correct clinical terms identified regardless of phrasing
2. Condition matching: expected condition appears in top 3
3. Consistency: same condition ranks #1 across variations of same vignette
4. No vitals in extracted_symptoms
5. STG guidelines present for conditions

Run:
  python3 tests/test_vignettes_comprehensive.py
"""

import asyncio
import json
import time
import sys
import os
import httpx

API_URL = os.getenv(
    "API_URL", "https://caremate-api-production.up.railway.app"
)
API_KEY = os.getenv("API_KEY", "OZz-taMLuTaZt0fm1CERAP-FhW_00aev0CzpfGwYY1w")

# ── 25 Vignettes with variations ──────────────────────────────────────────

VIGNETTES = [
    # 1. Tonsillitis / Pharyngitis
    {
        "id": "V01",
        "expected_condition": "Tonsillitis And Pharyngitis",
        "expected_code": "19.6",
        "expected_symptoms": ["sore throat", "pharyngitis", "fever"],
        "variations": [
            {"complaint": "sore throat and fever", "patient": {"age": 25, "sex": "female"}, "vitals": {"temperature": 38.5}},
            {"complaint": "my throat is very painful and I feel hot", "patient": {"age": 25, "sex": "female"}, "vitals": {"temperature": 38.5}},
            {"complaint": "it hurts to swallow and I have a high temperature", "patient": {"age": 25, "sex": "female"}, "vitals": {"temperature": 38.5}},
        ],
    },
    # 2. Common Cold
    {
        "id": "V02",
        "expected_condition": "Common Cold (Viral Rhinitis)",
        "expected_code": "19.2",
        "expected_symptoms": ["runny nose", "rhinorrhoea", "sneez"],
        "variations": [
            {"complaint": "runny nose, sneezing and mild sore throat", "patient": {"age": 30, "sex": "male"}, "vitals": {}},
            {"complaint": "my nose won't stop running and I keep sneezing", "patient": {"age": 30, "sex": "male"}, "vitals": {}},
            {"complaint": "blocked nose with watery discharge and scratchy throat", "patient": {"age": 30, "sex": "male"}, "vitals": {}},
        ],
    },
    # 3. Hypertension
    {
        "id": "V03",
        "expected_condition": "Hypertension",
        "expected_code": "4.7",
        "expected_symptoms": ["headache", "dizziness"],
        "variations": [
            {"complaint": "headache and dizziness", "patient": {"age": 55, "sex": "male"}, "vitals": {"systolic": 170, "diastolic": 100}},
            {"complaint": "my head is pounding and I feel lightheaded", "patient": {"age": 55, "sex": "male"}, "vitals": {"systolic": 170, "diastolic": 100}},
            {"complaint": "bad headache with feeling faint and dizzy spells", "patient": {"age": 55, "sex": "male"}, "vitals": {"systolic": 170, "diastolic": 100}},
        ],
    },
    # 4. Urinary Tract Infection
    {
        "id": "V04",
        "expected_condition": "Urinary Tract Infection",
        "expected_code": "8.4",
        "expected_symptoms": ["dysuria", "burning", "urin"],
        "variations": [
            {"complaint": "burning when urinating and frequent urination", "patient": {"age": 35, "sex": "female"}, "vitals": {"temperature": 37.8}},
            {"complaint": "it burns when I pee and I have to go all the time", "patient": {"age": 35, "sex": "female"}, "vitals": {"temperature": 37.8}},
            {"complaint": "painful urination with urgency and lower belly pain", "patient": {"age": 35, "sex": "female"}, "vitals": {"temperature": 37.8}},
        ],
    },
    # 5. Gastroenteritis / Diarrhoea
    {
        "id": "V05",
        "expected_condition": "Diarrhoea",
        "expected_code": "3.2",
        "expected_symptoms": ["diarrhoea", "vomit", "nausea"],
        "variations": [
            {"complaint": "diarrhoea and vomiting since yesterday", "patient": {"age": 28, "sex": "male"}, "vitals": {"temperature": 37.5}},
            {"complaint": "runny tummy and throwing up, can't keep anything down", "patient": {"age": 28, "sex": "male"}, "vitals": {"temperature": 37.5}},
            {"complaint": "loose watery stools many times today with nausea", "patient": {"age": 28, "sex": "male"}, "vitals": {"temperature": 37.5}},
        ],
    },
    # 6. Pneumonia / Lower Respiratory Tract Infection
    {
        "id": "V06",
        "expected_condition": "Pneumonia",
        "expected_code": "16.4",
        "expected_symptoms": ["cough", "fever", "chest pain"],
        "variations": [
            {"complaint": "cough with yellow sputum, fever and chest pain", "patient": {"age": 45, "sex": "male"}, "vitals": {"temperature": 39.2, "respiratoryRate": 28, "oxygenSat": 93}},
            {"complaint": "bad cough bringing up green stuff, high fever and pain in my chest when I breathe", "patient": {"age": 45, "sex": "male"}, "vitals": {"temperature": 39.2, "respiratoryRate": 28, "oxygenSat": 93}},
        ],
    },
    # 7. Oral Thrush / Candidiasis
    {
        "id": "V07",
        "expected_condition": "Oral Candidiasis",
        "expected_code": "1.2",
        "expected_symptoms": ["white patches", "mouth", "thrush"],
        "variations": [
            {"complaint": "white patches in mouth and pain when eating", "patient": {"age": 40, "sex": "female"}, "vitals": {}},
            {"complaint": "my mouth has white spots that hurt, difficulty swallowing food", "patient": {"age": 40, "sex": "female"}, "vitals": {}},
        ],
    },
    # 8. Depression
    {
        "id": "V08",
        "expected_condition": "Depress",  # Matches both "Depression" and "Depressive Disorder"
        "expected_code": "15.2",
        "expected_symptoms": ["depressed", "sleep", "appetite", "sad"],
        "variations": [
            {"complaint": "feeling very sad and hopeless, can't sleep, no appetite for weeks", "patient": {"age": 38, "sex": "female"}, "vitals": {}},
            {"complaint": "I don't want to do anything anymore, I feel empty inside and tired all the time", "patient": {"age": 38, "sex": "female"}, "vitals": {}},
            {"complaint": "low mood, insomnia, loss of interest in everything for the past month", "patient": {"age": 38, "sex": "female"}, "vitals": {}},
        ],
    },
    # 9. Diabetes (Type 2)
    {
        "id": "V09",
        "expected_condition": "Diabetes",
        "expected_code": "5.1",
        "expected_symptoms": ["thirst", "polyuria", "polydipsia", "urinat"],
        "variations": [
            {"complaint": "excessive thirst, urinating frequently and losing weight", "patient": {"age": 50, "sex": "male"}, "vitals": {}},
            {"complaint": "I'm always thirsty and going to the toilet a lot, also losing weight without trying", "patient": {"age": 50, "sex": "male"}, "vitals": {}},
        ],
    },
    # 10. Skin Infection / Cellulitis
    {
        "id": "V10",
        "expected_condition": "Cellulitis",
        "expected_code": "2.3",
        "expected_symptoms": ["red", "swollen", "skin", "warm"],
        "variations": [
            {"complaint": "red swollen painful area on my leg that feels hot", "patient": {"age": 42, "sex": "male"}, "vitals": {"temperature": 38.0}},
            {"complaint": "my leg is very red and puffy, it hurts to touch and feels warm", "patient": {"age": 42, "sex": "male"}, "vitals": {"temperature": 38.0}},
        ],
    },
    # 11. Asthma
    {
        "id": "V11",
        "expected_condition": "Asthma",
        "expected_code": "16.1",
        "expected_symptoms": ["wheez", "breathless", "chest tight"],
        "variations": [
            {"complaint": "wheezing, chest tightness and difficulty breathing", "patient": {"age": 22, "sex": "female"}, "vitals": {"oxygenSat": 95}},
            {"complaint": "I can't breathe properly, my chest feels tight and there's a whistling sound", "patient": {"age": 22, "sex": "female"}, "vitals": {"oxygenSat": 95}},
        ],
    },
    # 12. Conjunctivitis
    {
        "id": "V12",
        "expected_condition": "Conjunctivitis",
        "expected_code": "18.1",
        "expected_symptoms": ["eye", "red", "discharge"],
        "variations": [
            {"complaint": "red itchy eyes with discharge since this morning", "patient": {"age": 15, "sex": "male"}, "vitals": {}},
            {"complaint": "my eyes are pink and sticky, there's yellow stuff coming out", "patient": {"age": 15, "sex": "male"}, "vitals": {}},
        ],
    },
    # 13. Otitis Media
    {
        "id": "V13",
        "expected_condition": "Otitis Media",
        "expected_code": "19.4",
        "expected_symptoms": ["ear", "pain", "otalgia"],
        "variations": [
            {"complaint": "ear pain and fever, child pulling at ear", "patient": {"age": 3, "sex": "male"}, "vitals": {"temperature": 38.8}},
            {"complaint": "my child is crying and holding his ear, he has a temperature", "patient": {"age": 3, "sex": "male"}, "vitals": {"temperature": 38.8}},
        ],
    },
    # 14. Peptic Ulcer / Dyspepsia
    {
        "id": "V14",
        "expected_condition": "Dyspepsia",
        "expected_code": "3.5",
        "expected_symptoms": ["epigastric", "stomach", "burn", "heartburn"],
        "variations": [
            {"complaint": "burning pain in upper stomach especially after eating", "patient": {"age": 48, "sex": "male"}, "vitals": {}},
            {"complaint": "my stomach burns in the top part, worse when I eat, sometimes comes up to my throat", "patient": {"age": 48, "sex": "male"}, "vitals": {}},
            {"complaint": "heartburn and epigastric pain for two weeks", "patient": {"age": 48, "sex": "male"}, "vitals": {}},
        ],
    },
    # 15. Scabies
    {
        "id": "V15",
        "expected_condition": "Scabies",
        "expected_code": "2.14",
        "expected_symptoms": ["itch", "rash", "pruritus"],
        "variations": [
            {"complaint": "intense itching worse at night with rash between fingers", "patient": {"age": 20, "sex": "female"}, "vitals": {}},
            {"complaint": "I can't stop scratching, especially at night, there are bumps between my fingers and on my wrists", "patient": {"age": 20, "sex": "female"}, "vitals": {}},
        ],
    },
    # 16. Malaria
    {
        "id": "V16",
        "expected_condition": "Malaria",
        "expected_code": "10.4",
        "expected_symptoms": ["fever", "chills", "rigors", "sweat"],
        "variations": [
            {"complaint": "high fever with chills and sweating, headache and body aches", "patient": {"age": 35, "sex": "male"}, "vitals": {"temperature": 39.5}},
            {"complaint": "shivering attacks then sweating a lot, very high temperature and feel terrible", "patient": {"age": 35, "sex": "male"}, "vitals": {"temperature": 39.5}},
        ],
    },
    # 17. Epilepsy / Seizures
    {
        "id": "V17",
        "expected_condition": "Epilepsy",
        "expected_code": "15.1",
        "expected_symptoms": ["seizure", "convuls", "fit"],
        "variations": [
            {"complaint": "had a seizure this morning, shaking all over and lost consciousness", "patient": {"age": 28, "sex": "male"}, "vitals": {}},
            {"complaint": "my husband had a fit, his whole body was shaking and he didn't know where he was after", "patient": {"age": 28, "sex": "male"}, "vitals": {}},
        ],
    },
    # 18. Tuberculosis
    {
        "id": "V18",
        "expected_condition": "Tuberculosis",
        "expected_code": "16.6",
        "expected_symptoms": ["cough", "night sweat", "weight loss"],
        "variations": [
            {"complaint": "persistent cough for 3 weeks, night sweats and weight loss", "patient": {"age": 40, "sex": "male"}, "vitals": {"temperature": 37.8}},
            {"complaint": "I've been coughing for weeks, waking up soaked in sweat and my clothes are getting loose", "patient": {"age": 40, "sex": "male"}, "vitals": {"temperature": 37.8}},
        ],
    },
    # 19. Impetigo
    {
        "id": "V19",
        "expected_condition": "Impetigo",
        "expected_code": "2.9",
        "expected_symptoms": ["sore", "crust", "blister", "skin"],
        "variations": [
            {"complaint": "crusty yellow sores around mouth and nose", "patient": {"age": 6, "sex": "female"}, "vitals": {}},
            {"complaint": "my child has blisters on her face that burst and become crusty and yellow", "patient": {"age": 6, "sex": "female"}, "vitals": {}},
        ],
    },
    # 20. Allergic Rhinitis
    {
        "id": "V20",
        "expected_condition": "Allergic Rhinitis",
        "expected_code": "19.1",
        "expected_symptoms": ["sneez", "itch", "nose", "watery"],
        "variations": [
            {"complaint": "constant sneezing, itchy nose and watery eyes", "patient": {"age": 18, "sex": "female"}, "vitals": {}},
            {"complaint": "my nose itches and I sneeze all the time, my eyes water a lot", "patient": {"age": 18, "sex": "female"}, "vitals": {}},
        ],
    },
    # 21. Lower Back Pain
    {
        "id": "V21",
        "expected_condition": "Pain",  # STG has "Acute Pain" (20.2) — no specific "Low Back Pain" condition
        "expected_code": "20.2",
        "expected_symptoms": ["back pain", "lumbar", "lower back"],
        "variations": [
            {"complaint": "lower back pain for the past week, worse when bending", "patient": {"age": 45, "sex": "male"}, "vitals": {}},
            {"complaint": "my back is killing me at the bottom, I can barely bend over", "patient": {"age": 45, "sex": "male"}, "vitals": {}},
        ],
    },
    # 22. Headache / Tension Headache
    {
        "id": "V22",
        "expected_condition": "Headache",
        "expected_code": "15.4",
        "expected_symptoms": ["headache", "head pain", "cephalgia"],
        "variations": [
            {"complaint": "headache like a band around my head for 3 days", "patient": {"age": 32, "sex": "female"}, "vitals": {}},
            {"complaint": "my head hurts all over like pressure squeezing it, been going on for days", "patient": {"age": 32, "sex": "female"}, "vitals": {}},
        ],
    },
    # 23. Vaginal Discharge / STI
    {
        "id": "V23",
        "expected_condition": "Vaginal Discharge",  # Matches VDS (12.1) or Vaginal Discharge/LAP (6.15)
        "expected_code": "12.1",
        "expected_symptoms": ["discharge", "vaginal", "itch"],
        "variations": [
            {"complaint": "abnormal vaginal discharge with bad smell and itching", "patient": {"age": 28, "sex": "female"}, "vitals": {}},
            {"complaint": "there's a smelly discharge down below and it's very itchy", "patient": {"age": 28, "sex": "female"}, "vitals": {}},
        ],
    },
    # 24. Wound / Laceration
    {
        "id": "V24",
        "expected_condition": "Soft Tissue",  # STG names it "Soft Tissue Injuries" (21.3.7)
        "expected_code": "21.3.7",
        "expected_symptoms": ["wound", "cut", "bleeding", "laceration"],
        "variations": [
            {"complaint": "deep cut on my hand from a knife, bleeding a lot", "patient": {"age": 30, "sex": "male"}, "vitals": {}},
            {"complaint": "I cut myself badly on glass, it won't stop bleeding", "patient": {"age": 30, "sex": "male"}, "vitals": {}},
        ],
    },
    # 25. Sinusitis
    {
        "id": "V25",
        "expected_condition": "Sinusitis",
        "expected_code": "19.5",
        "expected_symptoms": ["sinus", "facial pain", "nasal", "congestion"],
        "variations": [
            {"complaint": "pain and pressure in my face around my nose and forehead, thick nasal discharge", "patient": {"age": 35, "sex": "female"}, "vitals": {"temperature": 37.9}},
            {"complaint": "my face hurts above my eyes and beside my nose, blocked nose with green mucus", "patient": {"age": 35, "sex": "female"}, "vitals": {"temperature": 37.9}},
        ],
    },
]


# ── Test Runner ────────────────────────────────────────────────────────────

async def call_analyze(client: httpx.AsyncClient, complaint: str, patient: dict, vitals: dict) -> dict:
    """Call the analyze endpoint with retry on timeout."""
    payload = {"complaint": complaint, "patient": patient}
    if vitals:
        payload["vitals"] = vitals

    for attempt in range(2):
        t0 = time.monotonic()
        try:
            resp = await client.post(
                f"{API_URL}/api/triage/analyze",
                json=payload,
                headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
                timeout=60.0,
            )
            elapsed = time.monotonic() - t0

            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}", "elapsed": elapsed}

            result = resp.json()
            result["elapsed"] = elapsed
            return result
        except httpx.ReadTimeout:
            elapsed = time.monotonic() - t0
            if attempt == 0:
                print(f"    ⏱  Timeout after {elapsed:.0f}s, retrying...")
                continue
            return {"error": f"TIMEOUT after {elapsed:.0f}s (2 attempts)", "elapsed": elapsed}
        except Exception as e:
            elapsed = time.monotonic() - t0
            return {"error": f"{type(e).__name__}: {str(e)[:200]}", "elapsed": elapsed}


def check_symptoms_extracted(result: dict, expected_terms: list[str]) -> tuple[bool, list[str]]:
    """Check if any expected symptom terms appear in extracted_symptoms."""
    extracted = [s.lower() for s in result.get("extracted_symptoms", [])]
    extracted_joined = " ".join(extracted)
    matched = []
    for term in expected_terms:
        term_lower = term.lower()
        if any(term_lower in s for s in extracted) or term_lower in extracted_joined:
            matched.append(term)
    return len(matched) > 0, matched


def check_condition_match(result: dict, expected_name: str, expected_code: str) -> tuple[int, float]:
    """Check if expected condition is in results. Returns (rank, confidence) or (-1, 0)."""
    for i, c in enumerate(result.get("conditions", []), 1):
        code = c.get("condition_code", "")
        name = c.get("condition_name", "").lower()
        if code == expected_code or expected_name.lower() in name:
            return i, c.get("confidence", 0)
    return -1, 0.0


def check_no_vitals_in_symptoms(result: dict) -> bool:
    """Check that extracted_symptoms don't contain vital sign readings."""
    vitals_terms = ["blood pressure", "heart rate", "bpm", "mmhg", "spo2", "respiratory rate"]
    for s in result.get("extracted_symptoms", []):
        s_lower = s.lower()
        for vt in vitals_terms:
            if vt in s_lower:
                return False
    return True


def check_stg_guidelines(result: dict) -> tuple[bool, int]:
    """Check if stg_guidelines are present and have data."""
    stg = result.get("stg_guidelines", {})
    return len(stg) > 0, len(stg)


async def run_tests():
    """Run all vignette tests."""
    print("=" * 80)
    print("CAREMATE COMPREHENSIVE VIGNETTE TEST SUITE")
    print(f"API: {API_URL}")
    print(f"Vignettes: {len(VIGNETTES)}")
    total_variations = sum(len(v["variations"]) for v in VIGNETTES)
    print(f"Total test cases: {total_variations}")
    print("=" * 80)
    print()

    # Results tracking
    all_results = []
    pass_count = 0
    fail_count = 0
    total_time = 0
    condition_match_failures = []
    symptom_extraction_failures = []
    vitals_in_symptoms = []
    consistency_failures = []
    no_stg_guidelines = []

    async with httpx.AsyncClient() as client:
        for vig in VIGNETTES:
            vig_id = vig["id"]
            expected_name = vig["expected_condition"]
            expected_code = vig["expected_code"]
            expected_symptoms = vig["expected_symptoms"]

            print(f"\n{'─' * 70}")
            print(f"  {vig_id}: {expected_name} (STG {expected_code})")
            print(f"{'─' * 70}")

            variation_results = []
            top_conditions_per_variation = []

            for vi, var in enumerate(vig["variations"], 1):
                complaint = var["complaint"]
                patient = var.get("patient", {})
                vitals = var.get("vitals", {})

                print(f"\n  Variation {vi}: \"{complaint}\"")

                result = await call_analyze(client, complaint, patient, vitals)
                elapsed = result.get("elapsed", 0)
                total_time += elapsed

                if "error" in result:
                    print(f"    ❌ ERROR: {result['error']}")
                    fail_count += 1
                    variation_results.append({"pass": False, "error": result["error"]})
                    continue

                # Test 1: Symptom extraction
                syms_ok, syms_matched = check_symptoms_extracted(result, expected_symptoms)
                extracted = result.get("extracted_symptoms", [])
                if syms_ok:
                    print(f"    ✅ Symptoms: {extracted}")
                else:
                    print(f"    ⚠️  Symptoms: {extracted} (expected match for: {expected_symptoms})")
                    symptom_extraction_failures.append(f"{vig_id}v{vi}: got {extracted}, wanted match for {expected_symptoms}")

                # Test 2: Condition match
                rank, confidence = check_condition_match(result, expected_name, expected_code)
                conditions_summary = [(c["condition_name"], c["confidence"]) for c in result.get("conditions", [])[:4]]
                if rank >= 1 and rank <= 3:
                    print(f"    ✅ Condition: #{rank} {expected_name} ({confidence:.0%})")
                    if rank == 1:
                        pass_count += 1
                    else:
                        pass_count += 1
                        print(f"       (not #1 but in top 3)")
                elif rank > 3:
                    print(f"    ⚠️  Condition: #{rank} (below top 3) — {conditions_summary}")
                    condition_match_failures.append(f"{vig_id}v{vi}: {expected_name} ranked #{rank}, top was: {conditions_summary[0] if conditions_summary else 'none'}")
                    fail_count += 1
                else:
                    print(f"    ❌ Condition: NOT FOUND — got: {conditions_summary}")
                    condition_match_failures.append(f"{vig_id}v{vi}: {expected_name} NOT FOUND, got: {conditions_summary}")
                    fail_count += 1

                top_conditions_per_variation.append(
                    result.get("conditions", [{}])[0].get("condition_name", "?") if result.get("conditions") else "none"
                )

                # Test 3: No vitals in symptoms
                vitals_clean = check_no_vitals_in_symptoms(result)
                if not vitals_clean:
                    print(f"    ⚠️  Vitals found in extracted_symptoms!")
                    vitals_in_symptoms.append(f"{vig_id}v{vi}: {extracted}")

                # Test 4: STG guidelines present
                stg_ok, stg_count = check_stg_guidelines(result)
                if stg_ok:
                    print(f"    ✅ STG guidelines: {stg_count} conditions")
                else:
                    print(f"    ⚠️  No STG guidelines in response")
                    no_stg_guidelines.append(f"{vig_id}v{vi}")

                # Timing
                print(f"    ⏱  {elapsed:.1f}s | Acuity: {result.get('acuity', '?')}")

                variation_results.append({
                    "pass": rank >= 1 and rank <= 3,
                    "rank": rank,
                    "confidence": confidence,
                    "symptoms_ok": syms_ok,
                    "elapsed": elapsed,
                })

            # Test 5: Consistency across variations
            unique_tops = set(top_conditions_per_variation)
            if len(unique_tops) > 1:
                print(f"\n  ⚠️  INCONSISTENT: Top condition varied across variations: {top_conditions_per_variation}")
                consistency_failures.append(f"{vig_id}: {top_conditions_per_variation}")
            elif len(unique_tops) == 1:
                print(f"\n  ✅ CONSISTENT: All variations → {list(unique_tops)[0]}")

            all_results.append({
                "id": vig_id,
                "expected": expected_name,
                "variations": variation_results,
                "consistent": len(unique_tops) <= 1,
            })

    # ── Summary Report ─────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)

    print(f"\nTotal test cases:          {total_variations}")
    print(f"Condition in top 3:        {pass_count}/{total_variations} ({pass_count/total_variations*100:.0f}%)")
    print(f"Condition missed:          {fail_count}/{total_variations}")
    print(f"Avg response time:         {total_time/total_variations:.1f}s")
    print(f"Total test time:           {total_time:.0f}s")

    # Consistency
    consistent_count = sum(1 for r in all_results if r["consistent"])
    print(f"\nConsistency (same #1):     {consistent_count}/{len(VIGNETTES)} vignettes")

    # Symptom extraction
    sym_fail_count = len(symptom_extraction_failures)
    print(f"Symptom extraction ok:     {total_variations - sym_fail_count}/{total_variations}")

    # Vitals in symptoms
    print(f"No vitals in symptoms:     {total_variations - len(vitals_in_symptoms)}/{total_variations}")

    # STG guidelines
    print(f"STG guidelines present:    {total_variations - len(no_stg_guidelines)}/{total_variations}")

    # ── Failures detail ────────────────────────────────────────────────────
    if condition_match_failures:
        print(f"\n{'─' * 40}")
        print("CONDITION MATCH FAILURES:")
        for f in condition_match_failures:
            print(f"  ❌ {f}")

    if consistency_failures:
        print(f"\n{'─' * 40}")
        print("CONSISTENCY FAILURES (different #1 across variations):")
        for f in consistency_failures:
            print(f"  ⚠️  {f}")

    if symptom_extraction_failures:
        print(f"\n{'─' * 40}")
        print("SYMPTOM EXTRACTION ISSUES:")
        for f in symptom_extraction_failures:
            print(f"  ⚠️  {f}")

    if vitals_in_symptoms:
        print(f"\n{'─' * 40}")
        print("VITALS IN EXTRACTED_SYMPTOMS:")
        for f in vitals_in_symptoms:
            print(f"  ⚠️  {f}")

    if no_stg_guidelines:
        print(f"\n{'─' * 40}")
        print("MISSING STG GUIDELINES:")
        for f in no_stg_guidelines:
            print(f"  ⚠️  {f}")

    print("\n" + "=" * 80)
    return pass_count, fail_count


if __name__ == "__main__":
    passes, fails = asyncio.run(run_tests())
    sys.exit(0 if fails == 0 else 1)
