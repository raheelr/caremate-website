"""
Prescription Safety Checker
============================

Deterministic, batch safety checking for prescriptions against patient context.
No LLM calls. Pure DB lookups + in-memory rule evaluation.

Checks:
- Pregnancy contraindications (DB: medicines.pregnancy_safe + in-memory classes)
- Allergy: direct name match (any allergy) + cross-reactivity (6 drug classes)
- Drug-drug interactions (14 rules covering major SA primary care pairs)
- Cross-prescription interactions (Drug A vs Drug B both being prescribed)
- CNS depressant stacking (2+ sedating drugs)
- Paediatric dosing concerns (age < 12)
- Contraindications text from DB

Also pre-screens recommended (not-yet-prescribed) formulary drugs for
pregnancy + allergy issues so the UI can show "Contraindicated" badges
before the nurse clicks "Prescribe."

Scales across all 337 medicines in the STG knowledge base.
Target: <100ms response, no LLM.
"""

from __future__ import annotations

import logging
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)

# Cache injected by api/main.py at startup
_cache = None  # type: ignore  # ClinicalDataCache


def _get_allergy_drug_map() -> dict[str, set[str]]:
    """Get allergy keyword → drug class mapping from cache."""
    return _cache.allergy_drug_map if _cache else {}


def _get_interaction_rules() -> list[tuple[set[str], set[str], str, str]]:
    """Get drug-drug interaction rules from cache."""
    return _cache.interaction_rules if _cache else []


def _get_pregnancy_unsafe() -> dict[str, str]:
    """Get pregnancy-unsafe drug → reason mapping from cache."""
    return _cache.pregnancy_unsafe if _cache else {}


def _get_cns_depressants() -> set[str]:
    """Get CNS depressant drug set from cache."""
    return _cache.cns_depressants if _cache else set()


# ── Main Function ────────────────────────────────────────────────────────────

async def batch_check_prescription_safety(
    conn: asyncpg.Connection,
    prescriptions: list[dict],
    patient_context: dict,
    recommended_drugs: list[dict] | None = None,
    condition_codes: list[str] | None = None,
) -> dict:
    """Check all prescriptions and recommended drugs for safety issues.

    Args:
        conn: Database connection
        prescriptions: List of prescribed drugs [{name, dose, frequency, duration}, ...]
        patient_context: {age, sex, pregnancy_status, allergies, current_medications}
        recommended_drugs: Optional STG formulary drugs to pre-screen

    Returns:
        {
            prescription_alerts: [{drug_name, alerts: [{type, severity, message}]}],
            summary: {critical_count, warning_count, has_critical, summary_message},
            recommended_drug_alerts: [{drug_name, alerts: [{type, severity, message}]}],
        }
    """
    # Extract patient context
    age = patient_context.get("age")
    sex = patient_context.get("sex")
    preg_status = (patient_context.get("pregnancy_status") or "").lower()
    is_pregnant = preg_status in ("pregnant", "yes", "true")
    allergies_raw = patient_context.get("allergies") or ""
    current_meds_raw = patient_context.get("current_medications") or []

    # Normalise allergies to a set of lowercase strings
    if isinstance(allergies_raw, str):
        allergies = {a.strip().lower() for a in allergies_raw.split(",") if a.strip()}
    elif isinstance(allergies_raw, list):
        allergies = {a.strip().lower() for a in allergies_raw if isinstance(a, str) and a.strip()}
    else:
        allergies = set()

    # Normalise current medications
    if isinstance(current_meds_raw, str):
        current_meds = {m.strip().lower() for m in current_meds_raw.split(",") if m.strip()}
    elif isinstance(current_meds_raw, list):
        current_meds = set()
        for m in current_meds_raw:
            if isinstance(m, str) and m.strip():
                current_meds.add(m.strip().lower())
            elif isinstance(m, dict):
                name = (m.get("name") or m.get("drug_generic") or m.get("drug_name") or "").strip().lower()
                if name:
                    current_meds.add(name)
    else:
        current_meds = set()

    # Collect all drug names for batch DB lookup
    rx_names = []
    for rx in prescriptions:
        name = (rx.get("name") or rx.get("drug_generic") or rx.get("drug_name") or "").strip().lower()
        if name:
            rx_names.append(name)

    rec_names = []
    if recommended_drugs:
        for rd in recommended_drugs:
            name = (rd.get("name") or rd.get("drug_name") or "").strip().lower()
            if name:
                rec_names.append(name)

    all_names = list(set(rx_names + rec_names))

    # Batch DB lookup: one query for all drug names
    db_drugs = await _batch_lookup_drugs(conn, all_names)

    # Normalise condition codes
    cond_codes = condition_codes or []

    # Set of drugs in the current prescription batch (for cross-rx interaction checks)
    rx_name_set = set(rx_names)

    # Check each prescription
    prescription_alerts = []
    for rx in prescriptions:
        name = (rx.get("name") or rx.get("drug_generic") or rx.get("drug_name") or "").strip().lower()
        if not name:
            continue

        alerts = _check_single_drug(
            drug_name=name,
            db_info=db_drugs.get(name),
            is_pregnant=is_pregnant,
            allergies=allergies,
            current_meds=current_meds,
            other_prescribed=rx_name_set - {name},
            age=age,
        )

        # Check condition-drug modifiers (STG-extracted rules)
        if cond_codes:
            alerts.extend(_check_condition_drug_mods(name, cond_codes))

        if alerts:
            prescription_alerts.append({
                "drug_name": name,
                "alerts": alerts,
            })

    # Pre-screen recommended drugs (lightweight: pregnancy + allergy only)
    recommended_drug_alerts = []
    if recommended_drugs:
        for rd in recommended_drugs:
            name = (rd.get("name") or rd.get("drug_name") or "").strip().lower()
            if not name:
                continue

            alerts = _prescreen_recommended_drug(
                drug_name=name,
                db_info=db_drugs.get(name),
                is_pregnant=is_pregnant,
                allergies=allergies,
            )

            if alerts:
                recommended_drug_alerts.append({
                    "drug_name": name,
                    "treatment_line": rd.get("treatment_line", ""),
                    "alerts": alerts,
                })

    # Build summary
    all_alerts = []
    for pa in prescription_alerts:
        all_alerts.extend(pa["alerts"])

    critical_count = sum(1 for a in all_alerts if a["severity"] == "critical")
    warning_count = sum(1 for a in all_alerts if a["severity"] == "warning")

    if critical_count > 0:
        summary_message = f"{critical_count} critical safety issue{'s' if critical_count > 1 else ''} found. Review immediately."
    elif warning_count > 0:
        summary_message = f"{warning_count} warning{'s' if warning_count > 1 else ''} to review."
    else:
        summary_message = "No safety concerns identified."

    return {
        "prescription_alerts": prescription_alerts,
        "summary": {
            "critical_count": critical_count,
            "warning_count": warning_count,
            "has_critical": critical_count > 0,
            "has_warnings": warning_count > 0,
            "summary_message": summary_message,
        },
        "recommended_drug_alerts": recommended_drug_alerts,
    }


# ── DB Batch Lookup ──────────────────────────────────────────────────────────

async def _batch_lookup_drugs(
    conn: asyncpg.Connection,
    drug_names: list[str],
) -> dict[str, dict]:
    """Batch lookup drug info from medicines table. Returns {name_lower: row_dict}."""
    if not drug_names:
        return {}

    # Use ILIKE ANY for fuzzy matching
    patterns = [f"%{name}%" for name in drug_names]
    rows = await conn.fetch("""
        SELECT name, pregnancy_safe, pregnancy_notes, contraindications,
               adult_dose, paediatric_dose_mg_per_kg, paediatric_frequency,
               paediatric_note, schedule, routes
        FROM medicines
        WHERE name ILIKE ANY($1::text[])
    """, patterns)

    # Map back: for each requested name, find best match
    result: dict[str, dict] = {}
    for requested_name in drug_names:
        best_match = None
        for r in rows:
            db_name = r["name"].lower()
            if db_name == requested_name:
                best_match = dict(r)
                break
            elif requested_name in db_name or db_name in requested_name:
                if best_match is None:
                    best_match = dict(r)
        if best_match:
            result[requested_name] = best_match

    return result


# ── Single Drug Check ────────────────────────────────────────────────────────

def _check_single_drug(
    drug_name: str,
    db_info: dict | None,
    is_pregnant: bool,
    allergies: set[str],
    current_meds: set[str],
    other_prescribed: set[str],
    age: int | None,
) -> list[dict]:
    """Run all safety checks on a single prescribed drug. Returns list of alerts."""
    alerts = []

    # 1. Pregnancy check
    if is_pregnant:
        alerts.extend(_check_pregnancy(drug_name, db_info))

    # 2. Allergy cross-reactivity
    alerts.extend(_check_allergies(drug_name, allergies))

    # 3. Drug-drug interactions (against current meds)
    alerts.extend(_check_interactions(drug_name, current_meds))

    # 4. Cross-prescription interactions (against other drugs in this batch, excluding current_meds already checked)
    new_interactions = other_prescribed - current_meds
    if new_interactions:
        alerts.extend(_check_interactions(drug_name, new_interactions, prefix="Co-prescribed"))

    # 5. CNS depressant stacking
    cns_set = _get_cns_depressants()
    if drug_name in cns_set:
        other_cns = (current_meds | other_prescribed) & cns_set - {drug_name}
        if other_cns:
            alerts.append({
                "type": "cns_stacking",
                "severity": "warning",
                "message": f"Multiple CNS depressants: {drug_name} + {', '.join(sorted(other_cns))}. "
                           f"Risk of excessive sedation and respiratory depression.",
            })

    # 6. Paediatric dosing check
    if age is not None and age < 12:
        alerts.extend(_check_paediatric(drug_name, db_info))

    # 7. Contraindications text from DB
    if db_info and db_info.get("contraindications"):
        alerts.append({
            "type": "contraindications",
            "severity": "info",
            "message": f"STG contraindications: {db_info['contraindications']}",
        })

    return alerts


def _check_pregnancy(drug_name: str, db_info: dict | None) -> list[dict]:
    """Check drug pregnancy safety."""
    alerts = []

    # Check in-memory class rules first (these are definitive)
    pregnancy_unsafe = _get_pregnancy_unsafe()
    if drug_name in pregnancy_unsafe:
        alerts.append({
            "type": "pregnancy_contraindication",
            "severity": "critical",
            "message": f"{drug_name.title()} is contraindicated in pregnancy. "
                       f"{pregnancy_unsafe[drug_name]}",
        })
        return alerts

    # Check DB pregnancy_safe field
    if db_info:
        if db_info.get("pregnancy_safe") is False:
            notes = db_info.get("pregnancy_notes") or "Not safe in pregnancy per STG."
            alerts.append({
                "type": "pregnancy_contraindication",
                "severity": "critical",
                "message": f"{db_info.get('name', drug_name).title()} is contraindicated in pregnancy. {notes}",
            })
        elif db_info.get("pregnancy_safe") is None:
            alerts.append({
                "type": "pregnancy_unknown",
                "severity": "warning",
                "message": f"Pregnancy safety for {db_info.get('name', drug_name)} is not specified in STG. "
                           f"Verify before prescribing.",
            })

    return alerts


def _check_allergies(drug_name: str, allergies: set[str]) -> list[dict]:
    """Check for allergy matches. Two-phase approach:

    1. GENERIC direct-name matching — catches ANY allergy regardless of drug class.
       If patient says "allergic to X" and we're prescribing X, that's a match.
    2. CLASS-BASED cross-reactivity — if patient says "penicillin allergy", all
       penicillin-class drugs are flagged (amoxicillin, ampicillin, etc.).

    This scales to any allergy, not just penicillin/sulfa.
    """
    if not allergies:
        return []

    alerts = []

    # Phase 1: Direct name matching (catches ANY allergy)
    # Check if the drug name appears in any allergy entry, or vice versa
    for allergy in allergies:
        if not allergy:
            continue
        # Exact match
        if drug_name == allergy:
            alerts.append({
                "type": "allergy_direct",
                "severity": "critical",
                "message": f"Patient is allergic to {drug_name}. CONTRAINDICATED.",
            })
            return alerts
        # Substring match in either direction
        # e.g. allergy="amoxicillin" matches drug="amoxicillin/clavulanic acid"
        # e.g. allergy="benzathine benzylpenicillin" matches drug containing that name
        if len(allergy) >= 4 and (allergy in drug_name or drug_name in allergy):
            alerts.append({
                "type": "allergy_direct",
                "severity": "critical",
                "message": f"Patient is allergic to {allergy} — {drug_name} is CONTRAINDICATED.",
            })
            return alerts

    # Phase 2: Cross-reactivity class matching
    # e.g. allergy="penicillin" → block amoxicillin, ampicillin, etc.
    for allergy_keyword, drug_class in _get_allergy_drug_map().items():
        if allergy_keyword in allergies:
            # Exact class membership
            if drug_name in drug_class:
                alerts.append({
                    "type": "allergy_cross_reactivity",
                    "severity": "critical",
                    "message": f"Patient has {allergy_keyword} allergy — {drug_name} belongs to the same "
                               f"drug class and is CONTRAINDICATED.",
                })
                return alerts
            # Substring match for compound names
            # e.g. "benzathine benzylpenicillin" contains "penicillin"
            if any(class_drug in drug_name or drug_name in class_drug for class_drug in drug_class):
                alerts.append({
                    "type": "allergy_cross_reactivity",
                    "severity": "critical",
                    "message": f"Patient has {allergy_keyword} allergy — {drug_name} belongs to the same "
                               f"drug class and is CONTRAINDICATED.",
                })
                return alerts

    return alerts


def _check_interactions(
    drug_name: str,
    other_drugs: set[str],
    prefix: str = "",
) -> list[dict]:
    """Check drug-drug interactions against a set of other drugs."""
    alerts = []

    for group_a, group_b, severity, message in _get_interaction_rules():
        if drug_name in group_a:
            interacting = other_drugs & group_b
            if interacting:
                full_msg = f"{prefix + ': ' if prefix else ''}{message} (interacts with: {', '.join(sorted(interacting))})"
                alerts.append({
                    "type": "drug_interaction",
                    "severity": severity,
                    "message": full_msg,
                })
        elif drug_name in group_b:
            interacting = other_drugs & group_a
            if interacting:
                full_msg = f"{prefix + ': ' if prefix else ''}{message} (interacts with: {', '.join(sorted(interacting))})"
                alerts.append({
                    "type": "drug_interaction",
                    "severity": severity,
                    "message": full_msg,
                })

    return alerts


def _check_paediatric(drug_name: str, db_info: dict | None) -> list[dict]:
    """Check paediatric dosing concerns."""
    alerts = []

    if db_info:
        if db_info.get("paediatric_dose_mg_per_kg"):
            alerts.append({
                "type": "paediatric_dosing",
                "severity": "info",
                "message": f"Paediatric patient. Use weight-based dosing: "
                           f"{db_info['paediatric_dose_mg_per_kg']} mg/kg. "
                           f"{db_info.get('paediatric_note') or ''}".strip(),
            })
        else:
            alerts.append({
                "type": "paediatric_no_dose",
                "severity": "warning",
                "message": f"Paediatric patient but no paediatric dose in STG for "
                           f"{db_info.get('name', drug_name)}. Consult paediatric formulary.",
            })

    return alerts


# ── Recommended Drug Pre-Screen ──────────────────────────────────────────────

def _check_condition_drug_mods(
    drug_name: str,
    condition_codes: list[str],
) -> list[dict]:
    """Check if a drug is contraindicated/cautioned for any confirmed condition.

    Reads from _cache.reasoning_rules where rule_type == "drug_condition_mod".
    Returns alerts for contraindications and cautions based on STG rules.

    Args:
        drug_name: Normalised drug name (lowercase)
        condition_codes: List of STG codes for confirmed/suspected conditions

    Returns:
        List of alert dicts [{type, severity, message, stg_reference}]
    """
    if not _cache or not hasattr(_cache, "reasoning_rules") or not _cache.reasoning_rules:
        return []
    if not condition_codes:
        return []

    alerts = []

    for code in condition_codes:
        rules = _cache.reasoning_rules.get(code, [])
        for rule in rules:
            if rule.get("rule_type") != "drug_condition_mod":
                continue

            rd = rule.get("rule_data", {})
            rule_drug = (rd.get("drug") or "").lower().strip()
            modifier_type = (rd.get("modifier_type") or "").lower().strip()

            # Match drug name (exact or substring)
            if not rule_drug:
                continue
            if drug_name not in rule_drug and rule_drug not in drug_name:
                continue

            # Only alert on clinically significant modifier types
            if modifier_type == "contraindication":
                alternative = rd.get("alternative") or rd.get("specific_instruction") or ""
                msg = f"{drug_name.title()} is contraindicated for {rule.get('condition_name', code)}"
                if alternative:
                    msg += f". {alternative}"
                alerts.append({
                    "type": "condition_contraindication",
                    "severity": "critical",
                    "message": msg,
                    "stg_reference": f"STG {code}",
                })
            elif modifier_type == "caution":
                instruction = rd.get("specific_instruction") or rd.get("condition") or ""
                msg = f"Caution: {drug_name.title()} requires care with {rule.get('condition_name', code)}"
                if instruction:
                    msg += f". {instruction}"
                alerts.append({
                    "type": "condition_caution",
                    "severity": "warning",
                    "message": msg,
                    "stg_reference": f"STG {code}",
                })
            elif modifier_type == "avoid":
                alternative = rd.get("alternative") or ""
                msg = f"Avoid {drug_name.title()} in {rule.get('condition_name', code)}"
                if alternative:
                    msg += f". Use {alternative} instead"
                alerts.append({
                    "type": "condition_contraindication",
                    "severity": "critical",
                    "message": msg,
                    "stg_reference": f"STG {code}",
                })
            elif modifier_type == "dose_reduction":
                instruction = rd.get("specific_instruction") or rd.get("condition") or ""
                msg = f"Dose adjustment needed: {drug_name.title()} with {rule.get('condition_name', code)}"
                if instruction:
                    msg += f". {instruction}"
                alerts.append({
                    "type": "condition_dose_adjustment",
                    "severity": "warning",
                    "message": msg,
                    "stg_reference": f"STG {code}",
                })

    return alerts


def _prescreen_recommended_drug(
    drug_name: str,
    db_info: dict | None,
    is_pregnant: bool,
    allergies: set[str],
) -> list[dict]:
    """Lightweight pre-screen for recommended (not-yet-prescribed) drugs.

    Only checks pregnancy + allergy — enough to show "Contraindicated" badge
    on formulary list before the nurse prescribes.
    """
    alerts = []

    if is_pregnant:
        alerts.extend(_check_pregnancy(drug_name, db_info))

    alerts.extend(_check_allergies(drug_name, allergies))

    return alerts
