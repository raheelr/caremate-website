"""
Encounter Agent — Intelligent Clinical Documentation
-----------------------------------------------------
Three generation functions for encounter-stage clinical content:
- generate_soap_note()    — SOAP note from encounter data + STG context
- generate_care_plan()    — Patient-facing care plan grounded in STG
- generate_discharge_summary() — Clinician-facing discharge summary

Each function fetches STG context from the DB and uses Haiku to generate
structured, patient-specific output grounded in the SA Standard Treatment
Guidelines (STG/EML).
"""

import json
import logging
import re

import anthropic
import asyncpg

from db.database import get_condition_detail

logger = logging.getLogger(__name__)

HAIKU_MODEL = "claude-haiku-4-5-20251001"
_FALLBACK_MODELS = [HAIKU_MODEL, "claude-sonnet-4-5-20250929"]


async def _call_with_fallback(client: anthropic.AsyncAnthropic, **kwargs) -> anthropic.types.Message:
    """Try each model in order; fall back on 429/529 errors."""
    last_err = None
    for model in _FALLBACK_MODELS:
        try:
            return await client.messages.create(model=model, **kwargs)
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529):
                logger.warning(f"{model} unavailable ({e.status_code}), trying fallback")
                last_err = e
                continue
            raise
    raise last_err


# ── Shared STG context helper ───────────────────────────────────────────────


async def _fetch_stg_context(
    conn: asyncpg.Connection,
    condition_name: str,
    condition_code: str | None = None,
) -> dict:
    """Fetch STG guideline context for a condition.

    Returns: {name, stg_code, description, danger_signs, general_measures,
              referral_criteria, medicines: [{name, dose_context, treatment_line, ...}]}
    """
    # Resolve condition ID
    if condition_code:
        row = await conn.fetchrow(
            "SELECT id, name, stg_code FROM conditions WHERE stg_code = $1 LIMIT 1",
            condition_code,
        )
    else:
        row = None

    if not row:
        row = await conn.fetchrow(
            "SELECT id, name, stg_code FROM conditions WHERE name ILIKE $1 LIMIT 1",
            f"%{condition_name}%",
        )

    if not row:
        return {"name": condition_name, "stg_code": "", "found": False}

    detail = await get_condition_detail(conn, row["id"])
    if not detail:
        return {"name": condition_name, "stg_code": row["stg_code"] or "", "found": False}

    # Parse medicines JSON
    medicines = detail.get("medicines_json", [])
    if isinstance(medicines, str):
        medicines = json.loads(medicines)

    return {
        "name": detail.get("name", condition_name),
        "stg_code": detail.get("stg_code", ""),
        "description": (detail.get("description_text") or "")[:1500],
        "danger_signs": detail.get("danger_signs") or "",
        "general_measures": detail.get("general_measures") or "",
        "referral_criteria": detail.get("referral_criteria") or "",
        "medicines": [
            {
                "name": m.get("name", ""),
                "dose_context": m.get("dose_context", ""),
                "treatment_line": m.get("treatment_line", ""),
                "special_notes": m.get("special_notes", ""),
            }
            for m in (medicines or [])[:10]
        ],
        "found": True,
    }


def _format_collected_data(collected_data: dict) -> str:
    """Format collected assessment data into readable clinical text."""
    if not collected_data:
        return "No assessment data collected."

    parts = []
    vitals = {}
    symptoms = []
    labs = {}
    other = {}

    vital_keys = {
        "systolic_mm_hg", "diastolic_mm_hg", "heartRate", "heart_rate",
        "temperature", "spo2", "respiratory_rate", "weight_kg",
    }
    lab_keys = {
        "fpg_mmol_l", "rbg_mmol_l", "hba1c_pct", "creatinine",
        "egfr", "urine_dipstick",
    }

    for key, value in collected_data.items():
        if key in vital_keys:
            vitals[key] = value
        elif key in lab_keys:
            labs[key] = value
        elif isinstance(value, bool):
            if value:
                symptoms.append(key.replace("_", " "))
        else:
            other[key] = value

    if vitals:
        v_parts = []
        if "systolic_mm_hg" in vitals and "diastolic_mm_hg" in vitals:
            v_parts.append(f"BP {vitals['systolic_mm_hg']}/{vitals['diastolic_mm_hg']} mmHg")
        if vitals.get("heartRate") or vitals.get("heart_rate"):
            v_parts.append(f"HR {vitals.get('heartRate') or vitals.get('heart_rate')} bpm")
        if vitals.get("temperature"):
            v_parts.append(f"Temp {vitals['temperature']}°C")
        if vitals.get("spo2"):
            v_parts.append(f"SpO2 {vitals['spo2']}%")
        if vitals.get("respiratory_rate"):
            v_parts.append(f"RR {vitals['respiratory_rate']}/min")
        if vitals.get("weight_kg"):
            v_parts.append(f"Weight {vitals['weight_kg']} kg")
        parts.append(f"Vitals: {', '.join(v_parts)}")

    if symptoms:
        parts.append(f"Positive findings: {', '.join(symptoms)}")

    if labs:
        l_parts = [f"{k.replace('_', ' ')}: {v}" for k, v in labs.items()]
        parts.append(f"Lab results: {', '.join(l_parts)}")

    if other:
        for k, v in other.items():
            if v is not None and v != "" and v is not False:
                parts.append(f"{k.replace('_', ' ')}: {v}")

    return "\n".join(parts) if parts else "No significant assessment data."


def _format_prescriptions(prescriptions: list[dict]) -> str:
    """Format prescriptions into readable text."""
    if not prescriptions:
        return "No medications prescribed."
    lines = []
    for p in prescriptions:
        drug = p.get("drug_generic", "Unknown")
        dose = p.get("dose", "")
        freq = p.get("frequency", "")
        dur = p.get("duration", "")
        line = f"- {drug} {dose}mg {freq}"
        if dur:
            line += f" for {dur} days"
        lines.append(line)
    return "\n".join(lines)


# ── SOAP Note Generation ───────────────────────────────────────────────────


async def generate_soap_note(
    conn: asyncpg.Connection,
    condition_name: str,
    condition_code: str | None,
    patient: dict,
    chief_complaint: str | None,
    collected_data: dict,
    prescriptions: list[dict],
    triage_context: dict | None,
) -> dict:
    """Generate an STG-grounded SOAP note from encounter data.

    Returns: {soap_note: str, sections: {subjective, objective, assessment, plan}}
    """
    stg = await _fetch_stg_context(conn, condition_name, condition_code)

    # Build context for Haiku
    patient_info = f"{patient.get('age', 'unknown')}-year-old {patient.get('sex', 'unknown')}"
    if patient.get("name"):
        patient_info = f"{patient['name']}, {patient_info}"

    triage_summary = ""
    if triage_context:
        conditions = triage_context.get("conditions", [])
        if conditions:
            top = conditions[0]
            triage_summary = f"Triage: {top.get('condition_name', '')} (score {top.get('score', '')})"
        acuity = triage_context.get("acuity", "")
        sats = triage_context.get("sats_colour", "")
        if sats:
            triage_summary += f", SATS: {sats.upper()} ({acuity})"
        symptoms = triage_context.get("extracted_symptoms", [])
        if symptoms:
            triage_summary += f"\nExtracted symptoms: {', '.join(symptoms)}"

    stg_context = ""
    if stg.get("found"):
        stg_context = (
            f"STG Reference ({stg['stg_code']}):\n"
            f"Description: {stg['description'][:500]}\n"
            f"General measures: {stg['general_measures'][:300]}\n"
            f"Danger signs: {stg['danger_signs'][:300]}"
        )

    prompt = f"""Generate a concise clinical SOAP note for this encounter. Ground your note in the STG data provided.

PATIENT: {patient_info}
CHIEF COMPLAINT: {chief_complaint or condition_name}
CONFIRMED DIAGNOSIS: {condition_name}

ASSESSMENT DATA:
{_format_collected_data(collected_data)}

{triage_summary}

PRESCRIPTIONS:
{_format_prescriptions(prescriptions)}

{stg_context}

Write a professional SOAP note with exactly these 4 sections. Be concise and clinically precise.
Use the actual data provided — do not invent findings.
Reference the STG guideline where relevant (e.g., "per STG {stg.get('stg_code', '')}").

Format:
SUBJECTIVE:
[Chief complaint, history of present illness, relevant symptoms reported by patient]

OBJECTIVE:
[Vital signs, examination findings, lab results — only include what was actually measured]

ASSESSMENT:
[Working diagnosis, clinical reasoning, severity, relevant STG classification]

PLAN:
[Medications prescribed with doses, follow-up plan, patient education, referral criteria if applicable]"""

    client = anthropic.AsyncAnthropic(max_retries=3)
    response = await _call_with_fallback(
        client,
        max_tokens=1024,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    soap_text = response.content[0].text.strip()

    # Parse sections
    sections = _parse_soap_sections(soap_text)

    return {
        "soap_note": soap_text,
        "sections": sections,
    }


def _parse_soap_sections(text: str) -> dict:
    """Split a SOAP note into its 4 sections."""
    sections = {"subjective": "", "objective": "", "assessment": "", "plan": ""}

    # Try to split on section headers
    patterns = [
        (r"SUBJECTIVE:\s*\n?", "subjective"),
        (r"OBJECTIVE:\s*\n?", "objective"),
        (r"ASSESSMENT:\s*\n?", "assessment"),
        (r"PLAN:\s*\n?", "plan"),
    ]

    # Find all section positions
    positions = []
    for pattern, key in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            positions.append((match.start(), match.end(), key))

    positions.sort(key=lambda x: x[0])

    for i, (start, content_start, key) in enumerate(positions):
        if i + 1 < len(positions):
            content_end = positions[i + 1][0]
        else:
            content_end = len(text)
        sections[key] = text[content_start:content_end].strip()

    return sections


# ── Care Plan Generation ───────────────────────────────────────────────────


async def generate_care_plan(
    conn: asyncpg.Connection,
    condition_name: str,
    condition_code: str | None,
    patient: dict,
    prescriptions: list[dict],
    language: str = "en",
) -> dict:
    """Generate a patient-facing care plan grounded in STG.

    Returns: {care_plan: str, follow_up_days: int|None, danger_signs: [str], lifestyle_advice: [str]}
    """
    stg = await _fetch_stg_context(conn, condition_name, condition_code)

    patient_info = f"{patient.get('age', 'unknown')}-year-old {patient.get('sex', 'unknown')}"
    patient_name = patient.get("name", "Patient")

    stg_context = ""
    if stg.get("found"):
        stg_context = (
            f"STG Reference ({stg['stg_code']}):\n"
            f"General measures: {stg['general_measures'][:500]}\n"
            f"Danger signs: {stg['danger_signs'][:500]}\n"
            f"Referral criteria: {stg['referral_criteria'][:300]}"
        )
        if stg.get("medicines"):
            med_lines = []
            for m in stg["medicines"][:5]:
                line = m["name"]
                if m.get("dose_context"):
                    line += f" — {m['dose_context']}"
                med_lines.append(line)
            stg_context += f"\nSTG medicines: {'; '.join(med_lines)}"

    language_instruction = ""
    if language != "en":
        lang_names = {
            "zu": "isiZulu", "xh": "isiXhosa", "af": "Afrikaans",
            "nso": "Sepedi (Northern Sotho)", "tn": "Setswana",
            "st": "Sesotho (Southern Sotho)", "ts": "Xitsonga",
            "ss": "siSwati", "ve": "Tshivenda", "nr": "isiNdebele",
        }
        language_instruction = f"\nWrite the care plan in {lang_names.get(language, language)}."

    prompt = f"""Generate a patient-facing care plan for this encounter. Write for a patient with Grade 8 literacy level — use simple, clear language.

PATIENT: {patient_name}, {patient_info}
CONDITION: {condition_name}

PRESCRIBED MEDICATIONS:
{_format_prescriptions(prescriptions)}

{stg_context}
{language_instruction}

The care plan should include:
1. What is your condition (1-2 simple sentences)
2. Your medications — how to take them, with clear instructions
3. Things you should do at home (lifestyle advice from STG)
4. Danger signs — when to come back IMMEDIATELY (from STG danger signs)
5. Follow-up — when to return for your next visit

Also return these as structured data at the end in this exact format:
---STRUCTURED---
FOLLOW_UP_DAYS: [number or "none"]
DANGER_SIGNS: [comma-separated list]
LIFESTYLE_ADVICE: [comma-separated list]"""

    client = anthropic.AsyncAnthropic(max_retries=3)
    response = await _call_with_fallback(
        client,
        max_tokens=1024,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}],
    )

    full_text = response.content[0].text.strip()

    # Parse structured data from the end
    follow_up_days = None
    danger_signs = []
    lifestyle_advice = []
    care_plan = full_text

    # Find the structured section — Haiku may use different delimiters
    struct_match = re.search(r"(?:---STRUCTURED---|(?:\*\*)?STRUCTURED(?:\*\*)?)\s*\n", full_text, re.IGNORECASE)
    if struct_match:
        care_plan = full_text[:struct_match.start()].strip()
        structured = full_text[struct_match.end():]

        # Extract follow_up_days
        fu_match = re.search(r"FOLLOW_UP_DAYS:\s*(\d+)", structured, re.IGNORECASE)
        if fu_match:
            follow_up_days = int(fu_match.group(1))

        # Extract danger signs
        ds_match = re.search(r"DANGER_SIGNS:\s*(.+?)(?:\n|$)", structured, re.IGNORECASE)
        if ds_match:
            danger_signs = [s.strip() for s in ds_match.group(1).split(",") if s.strip()]

        # Extract lifestyle advice
        la_match = re.search(r"LIFESTYLE_ADVICE:\s*(.+?)(?:\n|$)", structured, re.IGNORECASE)
        if la_match:
            lifestyle_advice = [s.strip() for s in la_match.group(1).split(",") if s.strip()]

    return {
        "care_plan": care_plan,
        "follow_up_days": follow_up_days,
        "danger_signs": danger_signs,
        "lifestyle_advice": lifestyle_advice,
    }


# ── Discharge Summary Generation ──────────────────────────────────────────


async def generate_discharge_summary(
    conn: asyncpg.Connection,
    condition_name: str,
    condition_code: str | None,
    patient: dict,
    prescriptions: list[dict],
    collected_data: dict,
    triage_context: dict | None,
    soap_note: str | None,
) -> dict:
    """Generate a clinician-facing discharge summary.

    Returns: {summary: str, referral_needed: bool, follow_up_plan: str}
    """
    stg = await _fetch_stg_context(conn, condition_name, condition_code)

    patient_info = f"{patient.get('name', 'Patient')}, {patient.get('age', 'unknown')}-year-old {patient.get('sex', 'unknown')}"

    triage_summary = ""
    if triage_context:
        acuity = triage_context.get("acuity", "")
        sats = triage_context.get("sats_colour", "")
        symptoms = triage_context.get("extracted_symptoms", [])
        triage_summary = f"Triage: {acuity}"
        if sats:
            triage_summary += f" (SATS {sats.upper()})"
        if symptoms:
            triage_summary += f"\nPresenting symptoms: {', '.join(symptoms)}"

    stg_ref = ""
    if stg.get("found"):
        stg_ref = f"Referral criteria per STG {stg['stg_code']}: {stg['referral_criteria'][:300]}"

    soap_section = ""
    if soap_note:
        soap_section = f"\nSOAP NOTE (already generated):\n{soap_note[:1000]}"

    prompt = f"""Generate a concise clinician-facing discharge summary for handover/record.

PATIENT: {patient_info}
DIAGNOSIS: {condition_name}

ASSESSMENT DATA:
{_format_collected_data(collected_data)}

{triage_summary}

PRESCRIPTIONS:
{_format_prescriptions(prescriptions)}

{stg_ref}
{soap_section}

Write a discharge summary with:
1. Presenting complaint and key findings
2. Diagnosis confirmed
3. Treatment given and medications prescribed
4. Follow-up plan
5. Safety netting advice given

At the end, include exactly:
---META---
REFERRAL_NEEDED: [yes/no]
FOLLOW_UP_PLAN: [one line summary]"""

    client = anthropic.AsyncAnthropic(max_retries=3)
    response = await _call_with_fallback(
        client,
        max_tokens=512,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    full_text = response.content[0].text.strip()

    # Parse meta
    referral_needed = False
    follow_up_plan = "Follow up as clinically indicated."
    summary = full_text

    # Find meta section — Haiku may use different delimiters
    meta_match = re.search(r"(?:---META---|(?:\*\*)?META(?:\*\*)?)\s*\n", full_text, re.IGNORECASE)
    if meta_match:
        summary = full_text[:meta_match.start()].strip()
        meta = full_text[meta_match.end():]

        ref_match = re.search(r"REFERRAL_NEEDED:\s*(yes|no)", meta, re.IGNORECASE)
        if ref_match:
            referral_needed = ref_match.group(1).lower() == "yes"

        fu_match = re.search(r"FOLLOW_UP_PLAN:\s*(.+?)(?:\n|$)", meta, re.IGNORECASE)
        if fu_match:
            follow_up_plan = fu_match.group(1).strip()

    return {
        "summary": summary,
        "referral_needed": referral_needed,
        "follow_up_plan": follow_up_plan,
    }
