"""
Post-Enrichment Vignette Tests
-------------------------------
Tests the full triage pipeline with realistic clinical vignettes.
Each vignette has an expected top condition that should appear in the results.

Usage:
  # Start the server first:
  uvicorn api.main:app --reload --port 8000

  # Then run tests:
  python3 tests/test_vignettes.py
"""

import asyncio
import json
import sys
import httpx

BASE_URL = "http://localhost:8000"

VIGNETTES = [
    {
        "name": "Sore throat + fever",
        "complaint": "sore throat and difficulty swallowing with fever",
        "vitals": {"temperature": 38.5},
        "expected_condition": "Tonsillitis",
        "expected_stg": "19.6",
    },
    {
        "name": "Chest pain",
        "complaint": "sharp chest pain that gets worse when breathing",
        "vitals": {"heartRate": 95, "respiratoryRate": 22},
        "expected_condition": "Chest Pain",
        "expected_stg": None,  # multiple possible
    },
    {
        "name": "Burning urination",
        "complaint": "burning when urinating and frequent urge to pee",
        "vitals": {"temperature": 37.8},
        "expected_condition": "Urinary Tract Infection",
        "expected_stg": None,
    },
    {
        "name": "Child with diarrhoea",
        "complaint": "child has watery diarrhoea for 3 days with vomiting",
        "patient": {"age": 3, "sex": "male"},
        "vitals": {"temperature": 38.0},
        "expected_condition": "Diarrhoea",
        "expected_stg": None,
    },
    {
        "name": "Headache + stiff neck",
        "complaint": "severe headache with stiff neck and sensitivity to light",
        "vitals": {"temperature": 39.5},
        "expected_condition": "Meningitis",
        "expected_stg": None,
    },
    {
        "name": "White patches in mouth (HIV+)",
        "complaint": "white patches in mouth, painful, difficulty eating",
        "patient": {"age": 35, "sex": "female"},
        "vitals": {},
        "expected_condition": "Candidiasis",
        "expected_stg": "1.2",
    },
    {
        "name": "Cough + weight loss",
        "complaint": "persistent cough for 3 weeks with night sweats and weight loss",
        "vitals": {"temperature": 37.6},
        "expected_condition": "Tuberculosis",
        "expected_stg": None,
    },
    {
        "name": "Rash with itching",
        "complaint": "red itchy rash on arms and legs for a week",
        "vitals": {},
        "expected_condition": None,  # multiple dermatology conditions possible
        "expected_stg": None,
    },
    {
        "name": "High blood pressure",
        "complaint": "headache and dizziness, found to have high blood pressure",
        "vitals": {"systolic": 170, "diastolic": 100, "heartRate": 85},
        "expected_condition": "Hypertension",
        "expected_stg": None,
    },
    {
        "name": "Vaginal discharge",
        "complaint": "abnormal vaginal discharge with bad smell and itching",
        "patient": {"age": 28, "sex": "female"},
        "vitals": {},
        "expected_condition": None,  # multiple STI/vaginitis conditions
        "expected_stg": None,
    },
]


async def run_vignette(client: httpx.AsyncClient, vignette: dict, idx: int) -> dict:
    """Run a single vignette and check results."""
    payload = {
        "complaint": vignette["complaint"],
        "vitals": vignette.get("vitals", {}),
    }
    if vignette.get("patient"):
        payload["patient"] = vignette["patient"]

    print(f"\n{'='*70}")
    print(f"VIGNETTE {idx}: {vignette['name']}")
    print(f"Complaint: {vignette['complaint']}")
    print(f"{'='*70}")

    try:
        resp = await client.post(f"{BASE_URL}/api/triage/analyze", json=payload, timeout=60.0)
        if resp.status_code != 200:
            print(f"  ERROR: HTTP {resp.status_code} — {resp.text[:200]}")
            return {"name": vignette["name"], "pass": False, "error": resp.text[:200]}

        result = resp.json()

        # Display results
        print(f"  Acuity: {result.get('acuity', 'unknown')}")
        if result.get("acuity_reasons"):
            for r in result["acuity_reasons"][:3]:
                print(f"    - {r}")

        print(f"  Extracted symptoms: {result.get('extracted_symptoms', [])}")

        conditions = result.get("conditions", [])
        print(f"  Conditions ({len(conditions)}):")
        for i, c in enumerate(conditions[:5]):
            marker = ""
            if vignette.get("expected_condition") and vignette["expected_condition"].lower() in c.get("condition_name", "").lower():
                marker = " <<<< EXPECTED"
            if vignette.get("expected_stg") and c.get("condition_code") == vignette["expected_stg"]:
                marker = " <<<< EXPECTED"
            conf = c.get("confidence", 0)
            conf_str = f"{conf}%" if isinstance(conf, (int, float)) and conf > 1 else f"{conf}"
            print(f"    {i+1}. {c.get('condition_name', '?')} ({c.get('condition_code', '?')}) — {conf_str}{marker}")
            if c.get("matched_symptoms"):
                print(f"       Matched: {', '.join(c['matched_symptoms'][:5])}")

        # Check if expected condition is in top 5
        found = False
        if vignette.get("expected_condition"):
            for c in conditions[:5]:
                name = c.get("condition_name", "").lower()
                code = c.get("condition_code", "")
                if vignette["expected_condition"].lower() in name:
                    found = True
                    break
                if vignette.get("expected_stg") and code == vignette["expected_stg"]:
                    found = True
                    break
            if found:
                print(f"\n  PASS: Expected condition found in top 5")
            else:
                print(f"\n  FAIL: Expected '{vignette['expected_condition']}' NOT in top 5")
                # Check if it's anywhere in the results
                for i, c in enumerate(conditions):
                    name = c.get("condition_name", "").lower()
                    if vignette["expected_condition"].lower() in name:
                        print(f"  (Found at position {i+1})")
                        break
        else:
            found = True  # No specific expected condition
            print(f"\n  INFO: No specific expected condition — reviewed above")

        # Safety review
        if result.get("safety_review"):
            sr = result["safety_review"]
            print(f"  Safety: concerns={sr.get('concerns', [])}, missing={sr.get('missing_conditions', [])}")

        return {
            "name": vignette["name"],
            "pass": found,
            "acuity": result.get("acuity"),
            "top_conditions": [c.get("condition_name", "?") for c in conditions[:3]],
        }

    except httpx.ConnectError:
        print(f"  ERROR: Cannot connect to {BASE_URL} — is the server running?")
        return {"name": vignette["name"], "pass": False, "error": "Connection refused"}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"name": vignette["name"], "pass": False, "error": str(e)}


async def main():
    async with httpx.AsyncClient() as client:
        # Health check
        try:
            resp = await client.get(f"{BASE_URL}/api/health", timeout=5.0)
            health = resp.json()
            print(f"Server healthy: {health}")
        except Exception as e:
            print(f"Server not responding: {e}")
            print(f"Start with: uvicorn api.main:app --reload --port 8000")
            sys.exit(1)

        results = []
        for i, v in enumerate(VIGNETTES):
            r = await run_vignette(client, v, i + 1)
            results.append(r)

        # Summary
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        passed = sum(1 for r in results if r["pass"])
        total = len(results)
        for r in results:
            status = "PASS" if r["pass"] else "FAIL"
            top = ", ".join(r.get("top_conditions", [])[:2]) if r.get("top_conditions") else r.get("error", "?")
            print(f"  [{status}] {r['name']} — {top}")
        print(f"\n  {passed}/{total} passed")


if __name__ == "__main__":
    asyncio.run(main())
