"""
Deterministic Question Engine
===============================

Selects STG-grounded assessment questions from clinical reasoning rules
extracted from the knowledge base. Zero LLM calls — pure deterministic
logic based on the current differential, known data, and discriminating power.

Architecture:
    Cache injection: api/main.py sets _cache at startup
    Called from: triage_agent.py analyze() and refine()
    Data source: clinical_reasoning_rules table (loaded into ClinicalDataCache)

Usage:
    questions = select_assessment_questions(
        differential=[{"stg_code": "4.2", "name": "Angina", "score": 0.85}, ...],
        known_symptoms={"chest pain", "shortness of breath"},
        known_vitals={"heart_rate": 92, "blood_pressure_systolic": 150},
        known_labs={},
        patient_age=55,
        patient_sex="male",
    )
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Injected at startup by api/main.py
_cache = None  # ClinicalDataCache


@dataclass
class AssessmentQuestion:
    """A single STG-grounded assessment question for triage."""
    id: str
    question: str
    type: str  # yes_no, numeric, select, free_text
    options: list[str] | None
    required: bool
    round: int
    source_citation: str
    grounding: str  # "verified" — from STG extraction
    condition_codes: list[str]  # Which conditions this question relates to
    discriminating_power: float
    is_red_flag: bool
    rule_type: str

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "question": self.question,
            "type": self.type,
            "required": self.required,
            "round": self.round,
            "source_citation": self.source_citation,
            "grounding": self.grounding,
            "is_red_flag": self.is_red_flag,
        }
        if self.options:
            d["options"] = self.options
        return d


def select_assessment_questions(
    differential: list[dict],
    known_symptoms: set[str] | None = None,
    known_vitals: dict | None = None,
    known_labs: dict | None = None,
    patient_age: int | None = None,
    patient_sex: str | None = None,
    current_round: int = 1,
    max_questions: int = 5,
) -> list[dict]:
    """Select the best assessment questions from reasoning rules.

    Args:
        differential: Top conditions from triage (list of dicts with stg_code, name, score)
        known_symptoms: Set of symptom strings already reported/confirmed
        known_vitals: Dict of vital name → value already provided
        known_labs: Dict of lab name → value already provided
        patient_age: Patient age in years
        patient_sex: "male" or "female"
        current_round: Current triage round (1 = initial, 2+ = refinement)
        max_questions: Maximum questions to return

    Returns:
        List of question dicts ready for the frontend (same format as existing questions)
    """
    if not _cache or not hasattr(_cache, "reasoning_rules") or not _cache.reasoning_rules:
        return []

    known_symptoms = known_symptoms or set()
    known_vitals = known_vitals or {}
    known_labs = known_labs or {}

    # Normalise known data for matching
    known_symptoms_lower = {s.lower().strip() for s in known_symptoms}
    known_vital_names = {k.lower().strip() for k in known_vitals.keys()}
    known_lab_names = {k.lower().strip() for k in known_labs.keys()}

    # Get top condition codes from the differential
    top_codes = []
    for cond in differential[:5]:
        code = cond.get("stg_code") or cond.get("condition_code", "")
        if code:
            top_codes.append(code)

    if not top_codes:
        return []

    # Collect all differential codes (for rules_out scoring)
    all_diff_codes = set(top_codes)

    # ── Step 1: Collect candidate rules ──────────────────────────────────
    candidate_rules: list[tuple[float, dict]] = []  # (score, rule)

    for rank, code in enumerate(top_codes):
        rules = _cache.reasoning_rules.get(code, [])
        for rule in rules:
            # Filter by demographics
            if not _passes_demographic_filter(rule, patient_age, patient_sex):
                continue

            # Filter out rules where data is already known
            if _is_already_known(rule, known_symptoms_lower, known_vital_names, known_lab_names):
                continue

            # Must have an assessment question
            if not rule.get("assessment_question"):
                continue

            # Score the rule
            score = _score_rule(rule, rank, all_diff_codes, top_codes)
            candidate_rules.append((score, rule))

    # ── Step 2: Deduplicate by normalised question text ──────────────────
    seen_questions: set[str] = set()
    unique_rules: list[tuple[float, dict]] = []

    for score, rule in sorted(candidate_rules, key=lambda x: -x[0]):
        normalised = _normalise_question(rule["assessment_question"])
        if normalised in seen_questions:
            continue
        seen_questions.add(normalised)
        unique_rules.append((score, rule))

    # ── Step 3: Build question dicts ─────────────────────────────────────
    questions = []
    for i, (score, rule) in enumerate(unique_rules[:max_questions]):
        # Find all conditions this rule relates to
        related_codes = [rule["condition_stg_code"]]

        # Build source citation from rule
        source_parts = []
        if rule.get("source_tag"):
            source_parts.append(rule["source_tag"])
        source_parts.append(f"STG {rule['condition_stg_code']}")
        source_citation = " — ".join(source_parts)

        q = {
            "id": f"rule_{current_round}_{i+1}",
            "question": rule["assessment_question"],
            "type": rule.get("question_type", "yes_no"),
            "required": False,
            "round": current_round,
            "source_citation": source_citation,
            "grounding": "verified",
        }

        # Add options for select type
        if rule.get("question_type") == "select" and rule.get("question_options"):
            q["options"] = rule["question_options"]

        # Flag red flags
        if rule.get("is_red_flag"):
            q["is_red_flag"] = True

        questions.append(q)

    if questions:
        logger.info(
            f"Question engine: {len(candidate_rules)} candidates → "
            f"{len(unique_rules)} unique → {len(questions)} selected "
            f"(round {current_round}, {len(top_codes)} conditions)"
        )

    return questions


# ── Private Helpers ──────────────────────────────────────────────────────────

def _passes_demographic_filter(
    rule: dict,
    patient_age: int | None,
    patient_sex: str | None,
) -> bool:
    """Check if a rule applies to this patient's demographics."""
    # Age filter
    if patient_age is not None:
        age_min = rule.get("applies_to_age_min")
        age_max = rule.get("applies_to_age_max")
        if age_min is not None and patient_age < age_min:
            return False
        if age_max is not None and patient_age > age_max:
            return False

    # Sex filter
    rule_sex = rule.get("applies_to_sex")
    if rule_sex and patient_sex and rule_sex != patient_sex:
        return False

    return True


def _is_already_known(
    rule: dict,
    known_symptoms_lower: set[str],
    known_vital_names: set[str],
    known_lab_names: set[str],
) -> bool:
    """Check if the information this rule asks about is already known."""
    rule_type = rule.get("rule_type", "")
    rule_data = rule.get("rule_data", {})

    if rule_type == "lab_threshold":
        test_name = (rule_data.get("test_name") or "").lower()
        if test_name and test_name in known_lab_names:
            return True

    elif rule_type == "vital_threshold":
        vital_name = (rule_data.get("vital_name") or "").lower()
        # Map common vital names
        vital_aliases = {
            "blood pressure": {"blood_pressure_systolic", "bp", "sbp"},
            "bp": {"blood_pressure_systolic", "bp", "sbp"},
            "heart rate": {"heart_rate", "hr", "pulse"},
            "pulse": {"heart_rate", "hr", "pulse"},
            "temperature": {"temperature", "temp"},
            "respiratory rate": {"respiratory_rate", "rr"},
            "oxygen saturation": {"spo2", "oxygen_saturation", "sats"},
            "spo2": {"spo2", "oxygen_saturation", "sats"},
        }
        aliases = vital_aliases.get(vital_name, {vital_name})
        if aliases & known_vital_names:
            return True

    elif rule_type in ("examination_finding", "clinical_sign"):
        finding = (rule_data.get("finding") or rule_data.get("sign_name") or "").lower()
        if finding and finding in known_symptoms_lower:
            return True

    elif rule_type == "history_discriminator":
        discriminator = (rule_data.get("discriminator") or "").lower()
        if discriminator and discriminator in known_symptoms_lower:
            return True

    return False


def _score_rule(
    rule: dict,
    rank: int,
    all_diff_codes: set[str],
    top_codes: list[str],
) -> float:
    """Score a rule for selection priority.

    Higher score = more useful question to ask right now.
    """
    base = float(rule.get("discriminating_power", 0.5))

    # Bonus for rules that separate conditions in the differential
    rules_out = set(rule.get("rules_out_codes") or [])
    overlap = rules_out & all_diff_codes
    base += 0.15 * len(overlap)

    # Bonus for red flags
    if rule.get("is_red_flag"):
        base += 0.10

    # Rank decay — prefer questions for higher-ranked conditions
    base -= 0.03 * rank

    # Prefer certain rule types that are most clinically actionable
    type_bonus = {
        "examination_finding": 0.05,
        "history_discriminator": 0.05,
        "vital_threshold": 0.03,
        "lab_threshold": 0.03,
        "severity_classifier": 0.02,
        "clinical_sign": 0.04,
    }
    base += type_bonus.get(rule.get("rule_type", ""), 0)

    # Small penalty for low-discrimination rules with no separation value
    if base < 0.4 and not overlap:
        base -= 0.10

    return base


def _normalise_question(question: str) -> str:
    """Normalise question text for deduplication."""
    q = question.lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


# ── Runtime Functions for Reasoning Rules ────────────────────────────────────
# These functions read from _cache.reasoning_rules and return structured data
# that gets wired into the triage response and prescription safety checks.


def get_referral_triggers(condition_codes: list[str]) -> dict[str, list[dict]]:
    """Get STG referral triggers for a list of condition codes.

    Filters reasoning rules for rule_type == "referral_trigger" matching
    the given conditions.

    Args:
        condition_codes: List of STG codes (e.g. ["4.7.1", "17.1"])

    Returns:
        Dict of condition_code → list of referral trigger dicts:
        [{criterion, refer_to, urgency, stg_code, is_red_flag}]
    """
    if not _cache or not _cache.reasoning_rules:
        return {}

    result: dict[str, list[dict]] = {}

    for code in condition_codes:
        rules = _cache.reasoning_rules.get(code, [])
        triggers = []
        for rule in rules:
            if rule.get("rule_type") != "referral_trigger":
                continue
            rd = rule.get("rule_data", {})
            triggers.append({
                "criterion": rd.get("criterion", ""),
                "refer_to": rd.get("refer_to", ""),
                "urgency": rd.get("urgency", "routine"),
                "stg_code": code,
                "source_citation": f"STG {code}",
                "is_red_flag": bool(rule.get("is_red_flag")),
            })
        if triggers:
            result[code] = triggers

    return result


def classify_severity(
    condition_code: str,
    vitals: dict | None = None,
    symptoms: list[str] | None = None,
) -> dict | None:
    """Match severity classification rules against patient data.

    Args:
        condition_code: STG code to check severity for
        vitals: Dict of vital name → value
        symptoms: List of reported symptom strings

    Returns:
        {severity, criteria_met, action, stg_code} or None if no match
    """
    if not _cache or not _cache.reasoning_rules:
        return None

    rules = _cache.reasoning_rules.get(condition_code, [])
    vitals = vitals or {}
    symptoms_lower = {s.lower() for s in (symptoms or [])}

    # Normalise vitals keys
    vitals_normalised = {k.lower().replace("_", " "): v for k, v in vitals.items()}
    # Add common aliases
    if "systolic" in vitals or "sbp" in vitals:
        vitals_normalised["blood pressure"] = vitals.get("systolic") or vitals.get("sbp")
    if "heart_rate" in vitals or "heartRate" in vitals:
        vitals_normalised["heart rate"] = vitals.get("heart_rate") or vitals.get("heartRate")
    if "oxygenSat" in vitals or "spo2" in vitals:
        vitals_normalised["spo2"] = vitals.get("oxygenSat") or vitals.get("spo2")
        vitals_normalised["oxygen saturation"] = vitals_normalised["spo2"]
    if "temperature" in vitals:
        vitals_normalised["temperature"] = vitals.get("temperature")
    if "respiratory_rate" in vitals or "respiratoryRate" in vitals:
        vitals_normalised["respiratory rate"] = vitals.get("respiratory_rate") or vitals.get("respiratoryRate")

    best_match = None
    best_severity_rank = -1
    severity_rank = {"mild": 0, "moderate": 1, "severe": 2, "life_threatening": 3, "critical": 3}

    for rule in rules:
        if rule.get("rule_type") != "severity_classifier":
            continue

        rd = rule.get("rule_data", {})
        criteria = rd.get("criteria", [])
        action = rd.get("action", "")
        severity_label = rd.get("severity", "")

        # Check if criteria are met
        criteria_met = []
        for crit in criteria:
            param = (crit.get("parameter") or "").lower()
            op = crit.get("operator", "")
            val = crit.get("value")
            unit = (crit.get("unit") or "").lower()

            # Try to match against vitals
            vital_val = vitals_normalised.get(param)
            if vital_val is not None and val is not None:
                try:
                    vital_float = float(vital_val)
                    val_float = float(val)
                    matched = False
                    if op == ">=" and vital_float >= val_float:
                        matched = True
                    elif op == "<=" and vital_float <= val_float:
                        matched = True
                    elif op == ">" and vital_float > val_float:
                        matched = True
                    elif op == "<" and vital_float < val_float:
                        matched = True
                    elif op == "==" and vital_float == val_float:
                        matched = True

                    if matched:
                        criteria_met.append(
                            f"{param} {op} {val} {unit}".strip()
                        )
                except (ValueError, TypeError):
                    pass

            # Try to match against symptoms/clinical signs
            if param in symptoms_lower or any(param in s for s in symptoms_lower):
                criteria_met.append(param)

        if criteria_met:
            rank = severity_rank.get(severity_label.lower(), 0)
            if rank > best_severity_rank:
                best_severity_rank = rank
                best_match = {
                    "severity": severity_label,
                    "criteria_met": criteria_met,
                    "action": action,
                    "stg_code": condition_code,
                    "source_citation": f"STG {condition_code}",
                }

    return best_match


def match_lab_rules(
    labs: dict,
    condition_codes: list[str],
) -> list[dict]:
    """Match lab threshold rules against structured lab results.

    Args:
        labs: Dict of test_name → value (numeric or string)
        condition_codes: List of STG codes in the differential

    Returns:
        List of matched lab rule dicts:
        [{test_name, value, threshold, operator, interpretation, confirms_code, source_citation}]
    """
    if not _cache or not _cache.reasoning_rules or not labs:
        return []

    # Normalise lab names for matching
    labs_normalised: dict[str, str] = {}
    for k, v in labs.items():
        labs_normalised[k.lower().strip()] = str(v)

    matches = []

    for code in condition_codes:
        rules = _cache.reasoning_rules.get(code, [])
        for rule in rules:
            if rule.get("rule_type") != "lab_threshold":
                continue

            rd = rule.get("rule_data", {})
            test_name = (rd.get("test_name") or "").lower().strip()
            if not test_name:
                continue

            # Find matching lab by name (exact or substring)
            lab_value_str = None
            matched_name = None
            for lname, lval in labs_normalised.items():
                if test_name in lname or lname in test_name:
                    lab_value_str = lval
                    matched_name = lname
                    break

            if lab_value_str is None:
                continue

            # Try numeric comparison
            op = rd.get("operator", "")
            threshold = rd.get("value")
            interpretation = rd.get("interpretation", "")

            # If threshold is None, just the presence of the lab is informative
            if threshold is None:
                matches.append({
                    "test_name": matched_name,
                    "value": lab_value_str,
                    "threshold": None,
                    "operator": op,
                    "interpretation": interpretation,
                    "confirms_code": code,
                    "source_citation": f"STG {code}",
                })
                continue

            try:
                lab_float = float(lab_value_str)
                threshold_float = float(threshold)
                matched = False

                if op == ">=" and lab_float >= threshold_float:
                    matched = True
                elif op == "<=" and lab_float <= threshold_float:
                    matched = True
                elif op == ">" and lab_float > threshold_float:
                    matched = True
                elif op == "<" and lab_float < threshold_float:
                    matched = True
                elif op == "==" and lab_float == threshold_float:
                    matched = True

                if matched:
                    matches.append({
                        "test_name": matched_name,
                        "value": lab_value_str,
                        "threshold": threshold,
                        "operator": op,
                        "interpretation": interpretation,
                        "confirms_code": code,
                        "source_citation": f"STG {code}",
                    })
            except (ValueError, TypeError):
                # Non-numeric comparison — just flag the finding
                if lab_value_str.lower() in ("positive", "detected", "reactive", "abnormal"):
                    matches.append({
                        "test_name": matched_name,
                        "value": lab_value_str,
                        "threshold": threshold,
                        "operator": op,
                        "interpretation": interpretation,
                        "confirms_code": code,
                        "source_citation": f"STG {code}",
                    })

    return matches


def check_vital_rules(
    vitals: dict,
    condition_codes: list[str],
) -> list[dict]:
    """Check condition-specific vital threshold rules.

    These are STG-defined thresholds that go beyond generic SATS/TEWS scoring.
    For example, SpO2 < 94% in asthma → "moderate severity".

    Args:
        vitals: Dict of vital name → value
        condition_codes: List of STG codes in the differential

    Returns:
        List of alert dicts:
        [{vital, value, threshold, operator, severity, interpretation, condition_code, source_citation}]
    """
    if not _cache or not _cache.reasoning_rules or not vitals:
        return []

    # Normalise vitals
    vitals_normalised: dict[str, float] = {}
    for k, v in vitals.items():
        if v is None:
            continue
        key = k.lower().replace("_", " ").strip()
        try:
            vitals_normalised[key] = float(v)
        except (ValueError, TypeError):
            continue

    # Add aliases
    for alias_src, alias_dst in [
        ("systolic", "blood pressure"), ("sbp", "blood pressure"),
        ("heartrate", "heart rate"), ("hr", "heart rate"),
        ("oxygensat", "spo2"), ("oxygen sat", "spo2"), ("oxygen saturation", "spo2"),
        ("temp", "temperature"),
        ("respiratoryrate", "respiratory rate"), ("rr", "respiratory rate"),
    ]:
        if alias_src in vitals_normalised and alias_dst not in vitals_normalised:
            vitals_normalised[alias_dst] = vitals_normalised[alias_src]

    alerts = []

    for code in condition_codes:
        rules = _cache.reasoning_rules.get(code, [])
        for rule in rules:
            if rule.get("rule_type") != "vital_threshold":
                continue

            rd = rule.get("rule_data", {})
            vital_name = (rd.get("vital_name") or "").lower().strip()
            op = rd.get("operator", "")
            threshold = rd.get("value")
            severity = rd.get("severity", "")
            interpretation = rd.get("interpretation", "")

            if not vital_name or threshold is None:
                continue

            # Find the vital
            vital_val = vitals_normalised.get(vital_name)
            if vital_val is None:
                # Try partial match
                for vk, vv in vitals_normalised.items():
                    if vital_name in vk or vk in vital_name:
                        vital_val = vv
                        break

            if vital_val is None:
                continue

            try:
                threshold_float = float(threshold)
                matched = False

                if op == ">=" and vital_val >= threshold_float:
                    matched = True
                elif op == "<=" and vital_val <= threshold_float:
                    matched = True
                elif op == ">" and vital_val > threshold_float:
                    matched = True
                elif op == "<" and vital_val < threshold_float:
                    matched = True
                elif op == "==" and vital_val == threshold_float:
                    matched = True

                if matched:
                    alerts.append({
                        "vital": vital_name,
                        "value": vital_val,
                        "threshold": threshold_float,
                        "operator": op,
                        "severity": severity,
                        "interpretation": interpretation,
                        "condition_code": code,
                        "source_citation": f"STG {code}",
                    })
            except (ValueError, TypeError):
                continue

    return alerts
