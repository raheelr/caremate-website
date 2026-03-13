"""
Clinical Opportunities Engine
=============================

Deterministic, rules-based engine that surfaces proactive clinical
opportunities during a patient encounter:

- Age/sex-based screening reminders (SA PHC STG + national policy)
- Diagnosis-triggered workups and assessments
- Incidental findings from vitals (unrelated to primary diagnosis)
- SDOH & social assistance programs (SA-specific)
- Medication safety checks (drug interactions, contraindications)

Architecture: Same pattern as agents/sats.py — pure function, no DB
calls, no LLM calls. Rules are auditable, validated, <1ms evaluation.

Source: SA PHC STG 8th Edition, SA National Screening Policy,
SASSA grant guidelines, SADAG helpline information.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Cache injected by api/main.py at startup
_cache = None  # type: ignore  # ClinicalDataCache


def _get_drug_set(name: str) -> set[str]:
    """Get a named drug set from cache."""
    if _cache:
        return getattr(_cache, name, set()) or _cache.drug_classes.get(name, set())
    return set()



# ── Priority Ordering ───────────────────────────────────────────────────────

PRIORITY_RANK = {"urgent": 0, "warning": 1, "info": 2}


# ── Engine ───────────────────────────────────────────────────────────────────

class ClinicalOpportunitiesEngine:
    """Deterministic rules engine for clinical opportunities.

    No DB calls. No LLM calls. Pure evaluation of hard-coded rules
    against patient context. Same architecture as SATS/TEWS scoring.
    """

    def evaluate(
        self,
        patient_age: int | None = None,
        patient_sex: str | None = None,
        pregnancy_status: str | None = None,
        confirmed_diagnosis: str | None = None,
        diagnosis_stg_code: str | None = None,
        vitals: dict | None = None,
        prescriptions: list[dict] | None = None,
        extracted_symptoms: list[str] | None = None,
    ) -> list[dict]:
        """Evaluate all rules against current encounter context.

        Returns list of opportunity dicts sorted by priority (urgent > warning > info).
        """
        results: list[dict] = []
        dx_lower = (confirmed_diagnosis or "").lower()
        stg_code = diagnosis_stg_code or ""
        is_pregnant = (pregnancy_status or "").lower() in ("pregnant", "yes", "true")
        symptoms_lower = [s.lower() for s in (extracted_symptoms or [])]
        rx_names = self._extract_drug_names(prescriptions or [])

        rules = _cache.opportunity_rules if _cache else []
        for rule in rules:
            opp = self._evaluate_rule(
                rule, patient_age, patient_sex, is_pregnant,
                dx_lower, stg_code, vitals or {}, rx_names,
                symptoms_lower,
            )
            if opp:
                results.append(opp)

        results.sort(key=lambda o: PRIORITY_RANK.get(o["priority"], 99))
        return results

    def _evaluate_rule(
        self,
        rule: dict,
        age: int | None,
        sex: str | None,
        is_pregnant: bool,
        dx_lower: str,
        stg_code: str,
        vitals: dict,
        rx_names: set[str],
        symptoms_lower: list[str],
    ) -> dict | None:
        """Evaluate a single rule. Returns opportunity dict or None."""

        rule_type = rule.get("type")

        # ── Screening rules ──
        if rule_type == "screening":
            return self._check_screening(rule, age, sex, is_pregnant, dx_lower, symptoms_lower)

        # ── Diagnosis-triggered rules ──
        if rule_type == "diagnosis_triggered":
            return self._check_diagnosis_triggered(rule, dx_lower, stg_code)

        # ── Vitals nudges ──
        if rule_type == "vitals_nudge":
            return self._check_vitals(rule, vitals, dx_lower)

        # ── SDOH ──
        if rule_type == "sdoh":
            return self._check_sdoh(rule, age, is_pregnant, dx_lower, stg_code)

        # ── Medication safety ──
        if rule_type == "medication_safety":
            return self._check_medication(rule, rx_names, is_pregnant, vitals)

        return None

    # ── Screening ────────────────────────────────────────────────────────────

    def _check_screening(
        self, rule: dict, age: int | None, sex: str | None,
        is_pregnant: bool, dx_lower: str, symptoms_lower: list[str],
    ) -> dict | None:
        # Age range
        if not self._age_in_range(age, rule.get("min_age"), rule.get("max_age")):
            return None

        # Sex filter
        if rule.get("sex") and sex and sex.lower() != rule["sex"]:
            return None

        # Pregnancy exclusion
        if rule.get("exclude_pregnancy") and is_pregnant:
            return None

        # Pregnancy requirement
        if rule.get("require_pregnancy") and not is_pregnant:
            return None

        # Exclude if diagnosis matches
        for kw in rule.get("exclude_dx_contains", []):
            if kw in dx_lower:
                return None

        # Symptom/dx trigger (e.g., TB screening needs cough OR HIV)
        req = rule.get("require_symptom_or_dx")
        if req:
            has_symptom = any(s in sym for s in req.get("symptoms", []) for sym in symptoms_lower)
            has_dx = any(kw in dx_lower for kw in req.get("dx_contains", []))
            if not has_symptom and not has_dx:
                return None

        return self._format(rule, age=age)

    # ── Diagnosis-Triggered ──────────────────────────────────────────────────

    def _check_diagnosis_triggered(
        self, rule: dict, dx_lower: str, stg_code: str,
    ) -> dict | None:
        if not dx_lower:
            return None

        dx_match = any(kw in dx_lower for kw in rule.get("dx_contains", []))
        stg_match = any(stg_code.startswith(prefix) for prefix in rule.get("dx_stg_prefix", []))

        if not dx_match and not stg_match:
            return None

        return self._format(rule)

    # ── Vitals ───────────────────────────────────────────────────────────────

    def _check_vitals(
        self, rule: dict, vitals: dict, dx_lower: str,
    ) -> dict | None:
        check = rule.get("vitals_check")

        # Suppress if the finding IS the primary diagnosis
        for kw in rule.get("suppress_if_dx_contains", []):
            if kw in dx_lower:
                return None

        sbp = _to_float(vitals.get("systolic") or vitals.get("sbp"))
        dbp = _to_float(vitals.get("diastolic") or vitals.get("dbp"))
        hr = _to_float(vitals.get("heartRate") or vitals.get("heart_rate") or vitals.get("hr"))
        temp = _to_float(vitals.get("temperature") or vitals.get("temp"))
        spo2 = _to_float(vitals.get("oxygenSat") or vitals.get("oxygen_sat") or vitals.get("spo2"))

        if check == "elevated_bp":
            if sbp and sbp > 140 or dbp and dbp > 90:
                return self._format(rule, sbp=sbp, dbp=dbp)

        elif check == "tachycardia_non_febrile":
            if hr and hr > 100 and (temp is None or temp < 38):
                return self._format(rule, hr=hr, temp=temp or 0)

        elif check == "low_spo2":
            if spo2 and spo2 < 95:
                return self._format(rule, spo2=spo2)

        elif check == "hypothermia":
            if temp and temp < 35:
                return self._format(rule, temp=temp)

        return None

    # ── SDOH ─────────────────────────────────────────────────────────────────

    def _check_sdoh(
        self, rule: dict, age: int | None, is_pregnant: bool,
        dx_lower: str, stg_code: str,
    ) -> dict | None:
        # Age filter
        if not self._age_in_range(age, rule.get("min_age"), rule.get("max_age")):
            return None

        # Pregnancy requirement
        if rule.get("require_pregnancy") and not is_pregnant:
            return None

        # Diagnosis match (any keyword)
        dx_keywords = rule.get("dx_contains", []) or rule.get("dx_contains_any", [])
        stg_prefixes = rule.get("dx_stg_prefix", [])

        if dx_keywords or stg_prefixes:
            dx_match = any(kw in dx_lower for kw in dx_keywords)
            stg_match = any(stg_code.startswith(p) for p in stg_prefixes)
            if not dx_match and not stg_match:
                return None

        return self._format(rule)

    # ── Medication Safety ────────────────────────────────────────────────────

    def _check_medication(
        self, rule: dict, rx_names: set[str], is_pregnant: bool,
        vitals: dict,
    ) -> dict | None:
        check = rule.get("med_check")

        if check == "ace_in_pregnancy":
            if is_pregnant and rx_names & _get_drug_set("ace_inhibitors"):
                return self._format(rule)

        elif check == "warfarin_nsaid":
            has_warfarin = "warfarin" in rx_names
            has_nsaid = bool(rx_names & _get_drug_set("nsaids"))
            if has_warfarin and has_nsaid:
                return self._format(rule)

        elif check == "cns_stacking":
            cns_found = rx_names & _get_drug_set("cns_depressants")
            if len(cns_found) >= 2:
                return self._format(rule, cns_drugs=", ".join(sorted(cns_found)))

        elif check == "rifampicin_contraceptive":
            has_rifampicin = bool(rx_names & _get_drug_set("cyp450_inducers"))
            has_contraceptive = bool(rx_names & _get_drug_set("oral_contraceptives"))
            if has_rifampicin and has_contraceptive:
                return self._format(rule)

        elif check == "nsaid_in_pregnancy":
            if is_pregnant and rx_names & _get_drug_set("nsaids"):
                return self._format(rule)

        return None

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _age_in_range(
        self, age: int | None, min_age: int | None, max_age: int | None,
    ) -> bool:
        if age is None:
            return True  # Don't exclude if age unknown
        if min_age is not None and age < min_age:
            return False
        if max_age is not None and age > max_age:
            return False
        return True

    def _extract_drug_names(self, prescriptions: list[dict]) -> set[str]:
        """Extract normalised drug names from prescription list."""
        names = set()
        for rx in prescriptions:
            name = (
                rx.get("drug_generic")
                or rx.get("name")
                or rx.get("drug_name")
                or ""
            ).lower().replace("_", " ").strip()
            if name:
                # Add individual words too (e.g., "paracetamol oral" → "paracetamol")
                names.add(name)
                for word in name.split():
                    names.add(word)
        return names

    def _format(self, rule: dict, **kwargs) -> dict:
        """Format a rule into an opportunity result dict."""
        desc = rule["description"]
        try:
            desc = desc.format(**{k: v for k, v in kwargs.items() if v is not None})
        except (KeyError, IndexError):
            pass  # Template placeholders without matching kwargs are left as-is

        return {
            "id": rule["id"],
            "type": rule["type"],
            "title": rule["title"],
            "description": desc,
            "action_label": rule["action_label"],
            "priority": rule["priority"],
            "stg_reference": rule.get("stg_reference", ""),
        }


def _to_float(val) -> float | None:
    """Safely convert a value to float."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
