"""
Clinical Assistant Agent
------------------------
Conversational clinical Q&A agent for use during encounters.
Uses Anthropic SDK tool_use with 9 tools wrapping existing DB functions
and the markdown knowledge base.

Unlike the TriageAgent (deterministic pipeline), this agent uses a genuine
agentic loop — Haiku decides which tools to call based on the nurse's question.

Tools:
  1. search_guidelines       — search STG knowledge base (DB)
  2. lookup_condition         — full STG entry for a condition (DB)
  3. check_red_flags          — check symptoms against danger signs (DB)
  4. search_medications       — medication dosing + safety (DB)
  5. find_conditions          — differential diagnosis from symptoms (DB)
  6. check_drug_safety        — patient-specific drug safety check (in-memory)
  7. suggest_alternative      — find alternative drugs when one is contraindicated (DB)
  8. draft_referral_letter    — generate a referral letter with clinical context
  9. search_knowledge_base    — search extended markdown KB (file-based)
"""

import json
import logging
import os
import re
import anthropic
import asyncpg

from db.database import (
    search_knowledge_chunks,
    get_condition_detail,
    get_condition_red_flags,
    get_red_flag_matches,
    get_conditions_for_symptoms,
)
from agents.kb_search import search_markdown_kb

logger = logging.getLogger(__name__)

# Cache injected by api/main.py at startup (via triage_agent module)
_cache = None  # type: ignore  # ClinicalDataCache

# ── Tool definitions (Anthropic SDK format) ──────────────────────────────────

ASSISTANT_TOOLS = [
    {
        "name": "search_guidelines",
        "description": (
            "Search the SA Standard Treatment Guidelines (STG) knowledge base. "
            "Returns relevant text chunks with condition names and STG codes. "
            "Use this for any clinical question about treatment, management, or guidelines."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — clinical terms, condition names, or symptoms",
                },
                "condition_name": {
                    "type": "string",
                    "description": "Optional: filter results to a specific condition",
                },
                "max_chunks": {
                    "type": "integer",
                    "description": "Maximum chunks to return (default 6)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "lookup_condition",
        "description": (
            "Get the full STG guideline entry for a specific condition, including "
            "description, danger signs, general measures, medicines (with dosing), "
            "and referral criteria."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "condition_name": {
                    "type": "string",
                    "description": "The condition name to look up (e.g. 'Pneumonia', 'Hypertension')",
                },
            },
            "required": ["condition_name"],
        },
    },
    {
        "name": "check_red_flags",
        "description": (
            "Check symptoms or signs against RED_FLAG danger signs in the STG database. "
            "Returns which symptoms are danger signs and for which conditions. "
            "Use when asked about warning signs, when to refer, or urgent symptoms."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symptoms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of symptoms/signs to check against RED_FLAG database",
                },
                "condition_name": {
                    "type": "string",
                    "description": "Optional: check red flags for a specific condition only",
                },
            },
            "required": ["symptoms"],
        },
    },
    {
        "name": "search_medications",
        "description": (
            "Look up medication information including dosing, treatment line, "
            "pregnancy safety, paediatric dosing, and special notes. "
            "Searches the STG medicines database."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "drug_name": {
                    "type": "string",
                    "description": "Name of the medication to look up",
                },
                "condition_name": {
                    "type": "string",
                    "description": "Optional: condition context for condition-specific dosing",
                },
            },
            "required": ["drug_name"],
        },
    },
    {
        "name": "find_conditions",
        "description": (
            "Search the clinical knowledge graph for conditions matching given symptoms. "
            "Returns ranked conditions with match counts and matched features. "
            "Use when asked 'what could cause X?' or 'differential for Y'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symptoms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of symptoms to match against the knowledge graph",
                },
                "patient_age": {
                    "type": "integer",
                    "description": "Patient age in years (for demographic filtering)",
                },
                "patient_sex": {
                    "type": "string",
                    "enum": ["male", "female"],
                    "description": "Patient sex (for demographic filtering)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum conditions to return (default 5)",
                },
            },
            "required": ["symptoms"],
        },
    },
    {
        "name": "check_drug_safety",
        "description": (
            "Check whether a specific drug is safe for this patient given their age, "
            "sex, pregnancy status, and current medications. Returns contraindications, "
            "pregnancy safety, interactions with currently prescribed drugs, and age-specific "
            "dosing concerns. Use when a nurse asks 'Can I give X to this patient?'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "drug_name": {
                    "type": "string",
                    "description": "Name of the drug to check safety for",
                },
                "patient_age": {
                    "type": "integer",
                    "description": "Patient's age in years",
                },
                "patient_sex": {
                    "type": "string",
                    "enum": ["male", "female"],
                    "description": "Patient's sex",
                },
                "pregnancy_status": {
                    "type": "string",
                    "enum": ["pregnant", "not_pregnant", "unknown"],
                    "description": "Whether the patient is pregnant",
                },
                "current_medications": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of medications the patient is currently taking or being prescribed",
                },
            },
            "required": ["drug_name"],
        },
    },
    {
        "name": "suggest_alternative",
        "description": (
            "When a drug is contraindicated or unsuitable for a patient, find alternative "
            "medications from the STG for the same condition. Returns other treatment lines "
            "(second-line, alternative) with dosing. Use when a drug can't be used and the "
            "nurse needs to know what else to prescribe."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "condition_name": {
                    "type": "string",
                    "description": "The condition being treated",
                },
                "excluded_drug": {
                    "type": "string",
                    "description": "The drug that is contraindicated or unsuitable (to exclude from results)",
                },
                "reason": {
                    "type": "string",
                    "description": "Why the drug is excluded (e.g. 'pregnancy', 'allergy', 'interaction')",
                },
            },
            "required": ["condition_name", "excluded_drug"],
        },
    },
    {
        "name": "draft_referral_letter",
        "description": (
            "Draft a referral letter to a higher level of care (district hospital, specialist) "
            "for the current patient. Pre-fills clinical context from the encounter: diagnosis, "
            "vitals, treatment given, reason for referral. The nurse can review and edit before "
            "finalizing. Use when asked to write a referral or when escalation is needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "referral_reason": {
                    "type": "string",
                    "description": "Primary reason for referral (e.g. 'Failed first-line treatment', 'Suspected peritonsillar abscess')",
                },
                "referral_destination": {
                    "type": "string",
                    "description": "Where the patient is being referred to (e.g. 'District Hospital', 'ENT specialist')",
                },
                "urgency": {
                    "type": "string",
                    "enum": ["routine", "urgent", "emergency"],
                    "description": "How urgently the patient needs to be seen",
                },
                "treatment_given": {
                    "type": "string",
                    "description": "Treatment already administered at this facility",
                },
            },
            "required": ["referral_reason"],
        },
    },
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the extended clinical knowledge base (markdown files) for hospital-level, "
            "paediatric, maternal, obstetric, and triage guidelines BEYOND the primary care STG database. "
            "Use this when: (1) the STG database doesn't cover a topic, (2) you need hospital-level "
            "or specialist management, (3) you need paediatric-specific dosing/management, "
            "(4) you need maternal/perinatal care guidance, or (5) you need SATS triage reference. "
            "Sources include: Hospital Level Adult EML (2019), Paediatric Hospital STG (2017), "
            "Maternal & Perinatal Care (2024), O&G guidelines (SASOG), SATS triage manual."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — condition name, symptom, drug name, or clinical question",
                },
                "source": {
                    "type": "string",
                    "enum": [
                        "all",
                        "hospital-eml",
                        "paediatric-eml",
                        "maternal-perinatal",
                        "obstetrics-gynae",
                        "sats-triage",
                        "stg-primary",
                        "road-to-health",
                    ],
                    "description": "Filter to a specific knowledge source (default: all)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum sections to return (default 5)",
                },
            },
            "required": ["query"],
        },
    },
]

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are CareMate Clinical Assistant — a knowledgeable, concise clinical colleague supporting South African primary healthcare nurses during patient encounters.

{encounter_section}

YOUR ROLE:
- Answer clinical questions grounded in the SA Standard Treatment Guidelines (STG/EML)
- Help with medication dosing, drug safety, patient education, and clinical decision support
- Help with care coordination: referral letters, alternative medications, patient explanations
- You have tools to search the STG database — ALWAYS use them before answering clinical questions
- You also have search_knowledge_base to search BEYOND the STG database — Hospital Level EML, Paediatric EML, Maternal & Perinatal Care, O&G guidelines, and SATS triage reference. Use it when the STG doesn't cover a topic or you need hospital/specialist-level guidance.
- CRITICAL CARE LEVEL RULE: The nurse works at PRIMARY CARE level. When you find information from Hospital Level EML or Paediatric Hospital EML, frame it as REFERRAL CONTEXT — tell the nurse WHY to refer and WHAT the hospital will do, but do NOT instruct the nurse to perform hospital-level treatments. Say: "This requires referral. At hospital level, they will [treatment]. Your role is to stabilise, refer urgently, and explain to the patient what to expect."
- Never guess or answer from general medical knowledge — if neither the STG nor the extended knowledge base covers it, say so clearly

RULES:
1. ALWAYS search before answering. Call search_guidelines or lookup_condition first.
2. Cite your sources: "According to STG section 4.7.1, the first-line treatment is..."
3. For medication questions, use search_medications for exact STG dosing.
4. For "Can I give this drug?", use check_drug_safety with the patient's details. CRITICAL: ALWAYS include pregnancy_status, patient_age, patient_sex, and current_medications in the tool parameters — do NOT omit any of these. If the patient is pregnant (from encounter context or conversation), ALWAYS pass pregnancy_status="pregnant". If the result says "pregnancy safety not specified" or returns pregnancy_unknown, ALWAYS follow up by calling search_knowledge_base with the drug name + "pregnancy safety" to check the Maternal & Perinatal Care guidelines.
5. When a drug is contraindicated, use suggest_alternative to find STG alternatives.
6. For danger signs or referral criteria, use check_red_flags.
7. When asked to write a referral, use draft_referral_letter with clinical context.
8. If the STG doesn't cover a topic, try search_knowledge_base for hospital-level or specialist guidelines before saying "not covered". If found, cite the source clearly: "According to Hospital Level EML..." or "According to Maternal & Perinatal Care guidelines..."
9. If NEITHER the STG database NOR the extended knowledge base covers a topic, clearly separate your response:
   - First: "**Guidelines:** [Not covered in current STG or extended knowledge base]"
   - Then: "**Clinical consideration (not from guidelines):** [practical advice]"
   - End with: "⚕ Please verify with a senior clinician — this advice is not from the validated guidelines."
10. Keep responses concise and actionable — the nurse is busy at the point of care.
11. If you detect a safety concern, flag it prominently with **⚠ SAFETY** heading.
12. Format responses with markdown for readability (bold, bullets, headings).

PROACTIVE PATIENT CONTEXT — SHOW YOUR REASONING:
You have the patient's full encounter context above. USE IT PROACTIVELY and VISIBLY in every response.

CRITICAL RESPONSE FORMAT for medication recommendations:
Start EVERY medication recommendation with a "Patient Safety Check" section that explicitly lists EACH relevant patient factor and how it affects your recommendation. Example format:

**Patient Safety Check:**
- **Pregnancy (pregnant):** Checked pregnancy safety for all recommended drugs
- **Allergy (penicillin):** First-line penicillin-based antibiotics are contraindicated → using azithromycin instead
- **Current medications (Asthavent):** No interactions with recommended drugs

Then give your recommendation.

This format is MANDATORY — do NOT skip it. The nurse must see that you checked EVERY factor. If the encounter context lists pregnancy, allergies, or medications, you MUST mention each one explicitly in your safety check, even if there are no concerns ("no interactions found" is still valuable to state).

Additional context rules:
1. **Pregnancy**: ALWAYS state pregnancy status and its impact on drug choice. Name specific drugs that are contraindicated (NSAIDs, ACE inhibitors, statins, valproate, doxycycline, etc.).
2. **Allergies**: ALWAYS explain WHY you chose an alternative. Say "Since this patient is allergic to penicillin, the first-line treatment (phenoxymethylpenicillin / amoxicillin / benzathine benzylpenicillin) cannot be used. The STG second-line is azithromycin." Never just jump to the alternative without explaining.
3. **Age**: If child (<12), use paediatric dosing automatically. If elderly (>65), consider renal function and sedation risk.
4. **Current medications**: Check for interactions BEFORE recommending. State the result.
5. **Vitals**: If vitals are abnormal, factor this into your recommendations.
6. **Chronic conditions**: Consider how they affect drug choice.
7. **Sex**: Consider sex-specific conditions and drug considerations.

The nurse must see that CareMate retains and applies ALL context from triage — this contextual awareness is what makes CareMate more than a guideline lookup tool.

CLINICAL JUDGEMENT — CONNECTING THE DOTS:
- When a patient's labs, vitals, or clinical findings indicate a MORE SEVERE condition than the current diagnosis, proactively flag it.
- Examples: eGFR <45 with diabetes → flag CKD stage 3b, recommend nephrologist referral. Persistent BP >160/100 despite treatment → flag resistant hypertension. HbA1c >10% → flag uncontrolled diabetes, consider insulin initiation or referral.
- Always consider whether findings cross a REFERRAL THRESHOLD — if so, prominently recommend: "**⚠ REFERRAL RECOMMENDED:** [reason] — this patient may need specialist assessment beyond primary care scope."
- Think holistically: a question about one thing (e.g. drug dosing) may reveal a bigger clinical picture (e.g. organ damage, disease progression) that the nurse needs to know about.

CARE LEVEL BOUNDARIES:
- Primary Care (STG): Nurse manages. Give full treatment guidance.
- Hospital Level (EML): Doctor manages. Give REFERRAL CONTEXT only — why to refer, what to expect, how urgent, what to stabilise before transfer. Do NOT give hospital treatment protocols as instructions to the nurse.
- Specialist: Flag clearly — "This requires specialist assessment" — and help write a strong referral letter.
- When referral is needed, help the nurse: (1) stabilise the patient per STG, (2) explain to the patient why they're being referred and what will happen, (3) write a referral letter with full clinical context.

CARE COORDINATION:
- When a nurse asks you to explain something to a patient, use simple language (Grade 8 literacy level), avoid medical jargon, and include practical instructions (what to do, when to return, danger signs to watch for).
- When writing referral letters, include: patient demographics, diagnosis with STG code, relevant vitals, treatment given at PHC, reason for referral, and urgency. Use Hospital EML context to explain what care the patient needs at the next level — this makes referral letters stronger and more specific.
- If a drug is not safe, ALWAYS suggest what to use instead by calling suggest_alternative.
- Consider the full encounter context: diagnosis, vitals, prescriptions, and patient demographics when answering any question."""


def _extract_pregnancy_status(encounter_context: dict) -> str | None:
    """Extract pregnancy status from any location in the encounter context."""
    # Check patient.pregnancy_status
    patient = encounter_context.get("patient", {})
    if patient.get("pregnancy_status") and patient["pregnancy_status"] != "unknown":
        return patient["pregnancy_status"]
    # Check top-level pregnancy_status
    if encounter_context.get("pregnancy_status") and encounter_context["pregnancy_status"] != "unknown":
        return encounter_context["pregnancy_status"]
    # Check triage_results for pregnancy
    triage = encounter_context.get("triage_results", encounter_context.get("triage", {}))
    if isinstance(triage, dict):
        if triage.get("pregnancy_status") and triage["pregnancy_status"] != "unknown":
            return triage["pregnancy_status"]
        # Check patient context within triage
        tp = triage.get("patient", {})
        if isinstance(tp, dict) and tp.get("pregnancy_status") and tp["pregnancy_status"] != "unknown":
            return tp["pregnancy_status"]
    # Check history/core_history
    history = encounter_context.get("history", encounter_context.get("core_history", {}))
    if isinstance(history, dict) and history.get("pregnancy_status"):
        return history["pregnancy_status"]
    return None


def _build_system_prompt(encounter_context: dict | None) -> str:
    """Build system prompt with optional encounter context."""
    if encounter_context:
        parts = ["CURRENT ENCOUNTER:"]
        patient = encounter_context.get("patient", {})
        if patient:
            age = patient.get("age", "unknown")
            sex = patient.get("sex", "unknown")
            parts.append(f"- Patient: {age}-year-old {sex}")

        condition = encounter_context.get("condition")
        condition_code = encounter_context.get("conditionCode") or encounter_context.get("stg_code", "")
        if condition:
            if isinstance(condition, dict):
                name = condition.get("name", "")
                code = condition.get("stg_code", "") or condition_code
            else:
                name = str(condition)
                code = condition_code
            if name:
                parts.append(f"- Working diagnosis: {name} (STG {code})" if code else f"- Working diagnosis: {name}")
        complaint = encounter_context.get("chief_complaint")
        if complaint:
            parts.append(f"- Chief complaint: {complaint}")
        triage = encounter_context.get("triage", {})
        if triage:
            acuity = triage.get("acuity", "")
            colour = triage.get("sats_colour", "")
            if colour:
                parts.append(f"- SATS triage: {colour.upper()} ({acuity})")
        vitals = encounter_context.get("vitals", {})
        if vitals:
            vital_strs = []
            vital_flags = []
            for k, v in vitals.items():
                if v is not None:
                    vital_strs.append(f"{k}={v}")
                    try:
                        val = float(v) if not isinstance(v, (int, float)) else v
                        if k in ("systolic_bp", "sbp") and val >= 160:
                            vital_flags.append(f"⚠ ELEVATED BP: {k}={v} — check for hypertensive emergency")
                        elif k in ("heart_rate", "hr", "pulse") and val >= 120:
                            vital_flags.append(f"⚠ TACHYCARDIA: {k}={v}")
                        elif k in ("heart_rate", "hr", "pulse") and val < 50:
                            vital_flags.append(f"⚠ BRADYCARDIA: {k}={v}")
                        elif k in ("spo2", "oxygen_saturation") and val < 94:
                            vital_flags.append(f"⚠ LOW SpO2: {k}={v}% — assess for respiratory distress")
                        elif k in ("temperature", "temp") and val >= 39:
                            vital_flags.append(f"⚠ HIGH FEVER: {k}={v}°C")
                        elif k in ("respiratory_rate", "rr") and val >= 25:
                            vital_flags.append(f"⚠ TACHYPNOEA: {k}={v}")
                    except (ValueError, TypeError):
                        pass
            if vital_strs:
                parts.append(f"- Vitals: {', '.join(vital_strs)}")
            for flag in vital_flags:
                parts.append(f"- {flag}")

        # Current medications (from prescriptions, history, or triage)
        prescriptions = encounter_context.get("prescriptions", [])
        current_meds = encounter_context.get("current_medications", [])
        if prescriptions:
            rx_strs = []
            for rx in prescriptions:
                if isinstance(rx, dict):
                    drug = rx.get("drug_generic", rx.get("name", "unknown"))
                    dose = rx.get("dose", "")
                    freq = rx.get("frequency", "")
                    dur = rx.get("duration", "")
                    parts_rx = [drug]
                    if dose:
                        parts_rx.append(dose)
                    if freq:
                        parts_rx.append(freq)
                    if dur:
                        parts_rx.append(f"for {dur}")
                    rx_strs.append(" ".join(parts_rx))
                else:
                    rx_strs.append(str(rx))
            parts.append(f"- Current prescriptions: {'; '.join(rx_strs)}")
        if current_meds:
            if isinstance(current_meds, list):
                med_list = ", ".join(str(m) for m in current_meds)
            else:
                med_list = str(current_meds)
            parts.append(f"- Pre-existing medications: {med_list}")

        # History / comorbidities
        history_data = encounter_context.get("history", encounter_context.get("core_history", {}))
        if not isinstance(history_data, dict):
            history_data = {}
        conditions_hx = history_data.get("chronic_conditions") or history_data.get("comorbidities")
        if conditions_hx:
            if isinstance(conditions_hx, list):
                parts.append(f"- Chronic conditions: {', '.join(str(c) for c in conditions_hx)}")
            else:
                parts.append(f"- Chronic conditions: {conditions_hx}")

        meds_hx = history_data.get("current_medications") or history_data.get("medications")
        if meds_hx and not current_meds:
            if isinstance(meds_hx, list):
                parts.append(f"- Pre-existing medications: {', '.join(str(m) for m in meds_hx)}")
            else:
                parts.append(f"- Pre-existing medications: {meds_hx}")

        symptoms = encounter_context.get("extracted_symptoms", [])
        if symptoms:
            parts.append(f"- Extracted symptoms: {', '.join(symptoms)}")

        # ── Build CRITICAL PATIENT FACTORS block ──
        # This is a separate, prominent section that the LLM MUST reference in responses
        critical_factors = []

        # Pregnancy
        pregnancy = _extract_pregnancy_status(encounter_context)
        is_pregnant = pregnancy and pregnancy.lower() in ("pregnant", "yes", "true")
        if is_pregnant:
            critical_factors.append("PREGNANT — check ALL drugs for pregnancy safety. Contraindicated: NSAIDs, ACE inhibitors, statins, valproate, doxycycline, methotrexate.")
        elif pregnancy and pregnancy != "unknown":
            parts.append(f"- Pregnancy status: {pregnancy}")

        # Allergies
        allergies = (
            history_data.get("allergies")
            or encounter_context.get("allergies")
        )
        allergy_str = None
        if allergies:
            allergy_str = ', '.join(str(a) for a in allergies) if isinstance(allergies, list) else str(allergies)
            if allergy_str.lower().strip() in ("nkda", "none", "no", "no known drug allergies", "nil", ""):
                allergy_str = None
        if allergy_str:
            critical_factors.append(f"ALLERGIC TO {allergy_str.upper()} — do NOT prescribe this drug or cross-reactive drugs. Use second-line alternative.")

        # Current medications requiring interaction checks
        all_meds = []
        if prescriptions:
            for rx in prescriptions:
                if isinstance(rx, dict):
                    all_meds.append(rx.get("drug_generic", rx.get("name", "")))
                else:
                    all_meds.append(str(rx))
        if current_meds:
            if isinstance(current_meds, list):
                all_meds.extend(str(m) for m in current_meds)
            elif current_meds:
                all_meds.append(str(current_meds))
        if meds_hx and not current_meds:
            if isinstance(meds_hx, list):
                all_meds.extend(str(m) for m in meds_hx)
            elif meds_hx:
                all_meds.append(str(meds_hx))
        if all_meds:
            critical_factors.append(f"ON MEDICATIONS: {', '.join(all_meds)} — check interactions before prescribing anything new.")

        # Abnormal vitals
        if vital_flags:
            for flag in vital_flags:
                critical_factors.append(flag.replace("⚠ ", ""))

        # Build the critical factors block
        if critical_factors:
            parts.append("")
            parts.append("⚠⚠⚠ CRITICAL PATIENT FACTORS — YOU MUST ADDRESS EACH ONE IN YOUR RESPONSE: ⚠⚠⚠")
            for i, factor in enumerate(critical_factors, 1):
                parts.append(f"  {i}. {factor}")
            parts.append(">>> When recommending medications, START your response by stating how each critical factor above affects your recommendation. <<<")

        encounter_section = "\n".join(parts)
    else:
        encounter_section = "No active encounter — answering general STG questions."

    return SYSTEM_PROMPT_TEMPLATE.format(encounter_section=encounter_section)


# ── Tool execution ───────────────────────────────────────────────────────────


async def _exec_search_guidelines(conn: asyncpg.Connection, params: dict) -> dict:
    """Execute search_guidelines tool."""
    query = params["query"]
    condition_name = params.get("condition_name")
    max_chunks = params.get("max_chunks", 6)

    # If condition_name given, resolve to condition_id
    condition_id = None
    if condition_name:
        row = await conn.fetchrow(
            "SELECT id FROM conditions WHERE name ILIKE $1 LIMIT 1",
            f"%{condition_name}%",
        )
        if row:
            condition_id = row["id"]

    chunks = await search_knowledge_chunks(conn, query, condition_id=condition_id, limit=max_chunks)

    return {
        "chunks": [
            {
                "condition_name": c.get("condition_name", ""),
                "stg_code": c.get("stg_code", ""),
                "section_role": c.get("section_role", ""),
                "text": (c.get("chunk_text", "") or "")[:1500],  # cap length
            }
            for c in chunks
        ],
        "total_found": len(chunks),
    }


async def _exec_lookup_condition(conn: asyncpg.Connection, params: dict) -> dict:
    """Execute lookup_condition tool."""
    name = params["condition_name"]
    row = await conn.fetchrow(
        "SELECT id FROM conditions WHERE name ILIKE $1 LIMIT 1",
        f"%{name}%",
    )
    if not row:
        return {"error": f"Condition '{name}' not found in the STG database."}

    detail = await get_condition_detail(conn, row["id"])
    if not detail:
        return {"error": f"No STG detail available for '{name}'."}

    # Build clean response
    medicines = detail.get("medicines_json", [])
    if isinstance(medicines, str):
        medicines = json.loads(medicines)

    patient_pregnant = params.get("_patient_pregnant", False)
    patient_allergies = (params.get("_patient_allergies") or "").lower()

    annotated_meds = []
    for m in (medicines or []):
        med = {
            "name": m.get("name", ""),
            "dose_context": m.get("dose_context", ""),
            "treatment_line": m.get("treatment_line", ""),
            "special_notes": m.get("special_notes", ""),
            "pregnancy_safe": m.get("pregnancy_safe"),
            "paediatric_dose_mg_per_kg": m.get("paediatric_dose_mg_per_kg"),
        }
        # Annotate pregnancy safety when patient is pregnant
        if patient_pregnant:
            if m.get("pregnancy_safe") is False:
                med["pregnancy_warning"] = f"⚠ CONTRAINDICATED in pregnancy"
            elif m.get("pregnancy_safe") is None:
                med["pregnancy_warning"] = "Pregnancy safety unknown — verify"
            else:
                med["pregnancy_note"] = "✓ Safe in pregnancy"
        # Flag allergy matches
        if patient_allergies and patient_allergies in m.get("name", "").lower():
            med["allergy_warning"] = f"⚠ Patient allergic to {patient_allergies}"
        annotated_meds.append(med)

    result = {
        "name": detail.get("name", ""),
        "stg_code": detail.get("stg_code", ""),
        "description": (detail.get("description", "") or "")[:2000],
        "danger_signs": detail.get("danger_signs", ""),
        "general_measures": detail.get("general_measures", ""),
        "referral_criteria": detail.get("referral_criteria", ""),
        "medicines": annotated_meds,
    }
    if patient_pregnant:
        result["patient_context"] = "⚠ Patient is PREGNANT — check all medicines for pregnancy safety before prescribing."

    # ── Enrich with STG reasoning rules (deterministic, from cache) ──
    stg_code = detail.get("stg_code", "")
    if _cache and stg_code and stg_code in _cache.reasoning_rules:
        rules = _cache.reasoning_rules[stg_code]

        # Referral triggers — when to refer urgently
        referral_triggers = [
            {
                "criterion": r["rule_data"].get("criterion", ""),
                "refer_to": r["rule_data"].get("refer_to", ""),
                "urgency": r["rule_data"].get("urgency", ""),
                "is_red_flag": r["is_red_flag"],
            }
            for r in rules if r["rule_type"] == "referral_trigger"
        ]
        if referral_triggers:
            result["referral_triggers"] = referral_triggers

        # Severity classifiers — how to grade severity
        severity_criteria = [
            {
                "severity": r["rule_data"].get("severity", ""),
                "criteria": r["rule_data"].get("criteria", ""),
                "action": r["rule_data"].get("action", ""),
            }
            for r in rules if r["rule_type"] == "severity_classifier"
        ]
        if severity_criteria:
            result["severity_criteria"] = severity_criteria

        # Investigation recommendations — what to order
        investigations = [
            {
                "test_name": r["rule_data"].get("test_name", ""),
                "reason": r["rule_data"].get("reason", ""),
                "timing": r["rule_data"].get("timing", ""),
            }
            for r in rules if r["rule_type"] == "investigation_rec"
        ]
        if investigations:
            result["investigations"] = investigations

        # Lab/vital thresholds — condition-specific thresholds
        thresholds = [
            {
                "test": r["rule_data"].get("test_name", r["rule_data"].get("vital_name", "")),
                "threshold": r["rule_data"].get("threshold", ""),
                "interpretation": r["rule_data"].get("interpretation", ""),
                "direction": r["rule_data"].get("direction", ""),
            }
            for r in rules if r["rule_type"] in ("lab_threshold", "vital_threshold")
        ]
        if thresholds:
            result["clinical_thresholds"] = thresholds

        # Key discriminating features (top 5 by discriminating power)
        key_features = sorted(
            [r for r in rules if r["rule_type"] in ("history_sign", "exam_sign", "investigation_sign")
             and r["discriminating_power"] >= 0.7],
            key=lambda r: r["discriminating_power"],
            reverse=True,
        )[:5]
        if key_features:
            result["key_discriminating_features"] = [
                {
                    "feature": r["rule_data"].get("feature", r.get("assessment_question", "")),
                    "type": r["rule_type"],
                    "discriminating_power": r["discriminating_power"],
                    "is_red_flag": r["is_red_flag"],
                }
                for r in key_features
            ]

    return result


async def _exec_check_red_flags(conn: asyncpg.Connection, params: dict) -> dict:
    """Execute check_red_flags tool."""
    symptoms = [s.lower().strip() for s in params["symptoms"]]
    condition_name = params.get("condition_name")

    # Global red flag match
    matches = await get_red_flag_matches(conn, symptoms)

    # If condition specified, also get all red flags for that condition
    condition_flags = []
    if condition_name:
        row = await conn.fetchrow(
            "SELECT id FROM conditions WHERE name ILIKE $1 LIMIT 1",
            f"%{condition_name}%",
        )
        if row:
            condition_flags = await get_condition_red_flags(conn, row["id"])

    return {
        "triggered": [
            {
                "symptom": m["canonical_name"],
                "condition": m["condition_name"],
            }
            for m in matches
        ],
        "condition_danger_signs": [f["canonical_name"] for f in condition_flags],
        "requires_escalation": len(matches) > 0,
    }


async def _exec_search_medications(conn: asyncpg.Connection, params: dict) -> dict:
    """Execute search_medications tool."""
    drug_name = params["drug_name"]
    condition_name = params.get("condition_name")

    # Try condition-specific dosing first
    if condition_name:
        row = await conn.fetchrow("""
            SELECT m.name, m.adult_dose, m.adult_frequency, m.adult_duration,
                   m.paediatric_dose_mg_per_kg, m.paediatric_frequency, m.paediatric_note,
                   m.contraindications, m.pregnancy_safe, m.pregnancy_notes,
                   m.routes,
                   cm.dose_context, cm.treatment_line, cm.age_group, cm.special_notes
            FROM medicines m
            LEFT JOIN condition_medicines cm ON cm.medicine_id = m.id
            LEFT JOIN conditions c ON c.id = cm.condition_id
            WHERE m.name ILIKE $1
            AND (c.name ILIKE $2 OR c.stg_code = $2)
            LIMIT 1
        """, f"%{drug_name}%", f"%{condition_name}%")
    else:
        row = None

    # Fallback: just look up the medicine
    if not row:
        row = await conn.fetchrow("""
            SELECT m.name, m.adult_dose, m.adult_frequency, m.adult_duration,
                   m.paediatric_dose_mg_per_kg, m.paediatric_frequency, m.paediatric_note,
                   m.contraindications, m.pregnancy_safe, m.pregnancy_notes,
                   m.routes,
                   NULL as dose_context, NULL as treatment_line,
                   NULL as age_group, NULL as special_notes
            FROM medicines m
            WHERE m.name ILIKE $1
            LIMIT 1
        """, f"%{drug_name}%")

    if not row:
        return {"error": f"Medication '{drug_name}' not found in the STG medicines database."}

    d = dict(row)
    patient_pregnant = params.get("_patient_pregnant", False)

    result = {
        "name": d.get("name", ""),
        "adult_dose": d.get("adult_dose", ""),
        "adult_frequency": d.get("adult_frequency", ""),
        "adult_duration": d.get("adult_duration", ""),
        "dose_context": d.get("dose_context", ""),
        "treatment_line": d.get("treatment_line", ""),
        "routes": d.get("routes", []),
        "paediatric_dose_mg_per_kg": d.get("paediatric_dose_mg_per_kg"),
        "paediatric_frequency": d.get("paediatric_frequency", ""),
        "paediatric_note": d.get("paediatric_note", ""),
        "pregnancy_safe": d.get("pregnancy_safe"),
        "pregnancy_notes": d.get("pregnancy_notes", ""),
        "contraindications": d.get("contraindications", ""),
        "special_notes": d.get("special_notes", ""),
    }

    # Annotate pregnancy safety when patient IS pregnant
    if patient_pregnant:
        if d.get("pregnancy_safe") is False:
            result["pregnancy_warning"] = f"⚠ PATIENT IS PREGNANT — {d.get('name', drug_name)} is CONTRAINDICATED in pregnancy. {d.get('pregnancy_notes') or 'Do NOT prescribe.'}"
        elif d.get("pregnancy_safe") is None:
            result["pregnancy_warning"] = f"⚠ PATIENT IS PREGNANT — pregnancy safety for {d.get('name', drug_name)} is NOT specified in STG. Verify before prescribing."
        else:
            result["pregnancy_safe_confirmed"] = f"✓ {d.get('name', drug_name)} is safe in pregnancy."

    return result


async def _exec_find_conditions(conn: asyncpg.Connection, params: dict) -> dict:
    """Execute find_conditions tool."""
    symptoms = [s.lower().strip() for s in params["symptoms"]]
    patient_age = params.get("patient_age")
    patient_sex = params.get("patient_sex")
    limit = params.get("limit", 5)

    conditions = await get_conditions_for_symptoms(
        conn,
        symptom_names=symptoms,
        patient_age=patient_age,
        patient_sex=patient_sex,
        limit=limit,
    )

    return {
        "conditions": [
            {
                "name": c["name"],
                "stg_code": c["stg_code"],
                "match_count": c["match_count"],
                "matched_features": c.get("matched_features", []),
            }
            for c in conditions
        ],
        "total_found": len(conditions),
    }


async def _exec_check_drug_safety(conn: asyncpg.Connection, params: dict) -> dict:
    """Check drug safety for this patient — pregnancy, interactions, age concerns."""
    drug_name = params["drug_name"].lower().strip()
    patient_age = params.get("patient_age")
    patient_sex = params.get("patient_sex")
    pregnancy_status = params.get("pregnancy_status", "unknown")
    current_meds = [m.lower().strip() for m in params.get("current_medications", [])]

    # Look up the drug in STG
    row = await conn.fetchrow("""
        SELECT name, pregnancy_safe, pregnancy_notes, contraindications,
               adult_dose, paediatric_dose_mg_per_kg, schedule, routes
        FROM medicines WHERE name ILIKE $1 LIMIT 1
    """, f"%{drug_name}%")

    if not row:
        return {"error": f"Drug '{drug_name}' not found in STG medicines database."}

    d = dict(row)
    warnings = []
    safe = True

    # Pregnancy check — in-memory class rules first, then DB
    is_pregnant = pregnancy_status in ("pregnant", "yes", "true")
    drug_lower = d["name"].lower()
    pregnancy_unsafe = _cache.pregnancy_unsafe if _cache else {}
    if is_pregnant:
        if drug_lower in pregnancy_unsafe:
            safe = False
            warnings.append({
                "type": "pregnancy_contraindication",
                "severity": "critical",
                "message": f"**{d['name']}** is contraindicated in pregnancy. {pregnancy_unsafe[drug_lower]}",
            })
        elif d.get("pregnancy_safe") is False:
            safe = False
            notes = d.get("pregnancy_notes") or "Not safe in pregnancy per STG."
            warnings.append({
                "type": "pregnancy_contraindication",
                "severity": "critical",
                "message": f"**{d['name']}** is contraindicated in pregnancy. {notes}",
            })
        elif d.get("pregnancy_safe") is None:
            warnings.append({
                "type": "pregnancy_unknown",
                "severity": "caution",
                "message": f"Pregnancy safety for {d['name']} is not specified in STG. Consult specialist.",
            })

    # Drug-drug interaction checks — from cache
    interaction_rules = _cache.interaction_rules if _cache else []
    current_meds_set = set(current_meds)
    for group_a, group_b, severity, message in interaction_rules:
        if drug_lower in group_a and current_meds_set & group_b:
            interacting = current_meds_set & group_b
            warnings.append({
                "type": "drug_interaction",
                "severity": severity,
                "message": f"{message} (interacts with: {', '.join(interacting)})",
            })
        elif drug_lower in group_b and current_meds_set & group_a:
            interacting = current_meds_set & group_a
            warnings.append({
                "type": "drug_interaction",
                "severity": severity,
                "message": f"{message} (interacts with: {', '.join(interacting)})",
            })

    # CNS depressant stacking
    cns_set = _cache.cns_depressants if _cache else set()
    if drug_lower in cns_set:
        other_cns = current_meds_set & cns_set
        if other_cns:
            warnings.append({
                "type": "cns_stacking",
                "severity": "warning",
                "message": f"Multiple CNS depressants: {drug_lower} + {', '.join(other_cns)}. Risk of excessive sedation and respiratory depression.",
            })

    # Paediatric dosing note
    if patient_age is not None and patient_age < 12 and d.get("paediatric_dose_mg_per_kg"):
        warnings.append({
            "type": "paediatric_dosing",
            "severity": "info",
            "message": f"Paediatric patient. Use weight-based dosing: {d['paediatric_dose_mg_per_kg']} mg/kg.",
        })
    elif patient_age is not None and patient_age < 12 and not d.get("paediatric_dose_mg_per_kg"):
        warnings.append({
            "type": "paediatric_no_dose",
            "severity": "caution",
            "message": f"Paediatric patient but no paediatric dose in STG for {d['name']}. Consult paediatric formulary.",
        })

    # Contraindications text
    if d.get("contraindications"):
        warnings.append({
            "type": "contraindications",
            "severity": "info",
            "message": f"STG contraindications: {d['contraindications']}",
        })

    # Drug-condition modifiers from STG reasoning rules (491 rules)
    # Check if this drug has condition-specific warnings/contraindications
    if _cache and _cache.reasoning_rules:
        for stg_code, rules in _cache.reasoning_rules.items():
            for r in rules:
                if r["rule_type"] != "drug_condition_mod":
                    continue
                rd = r["rule_data"]
                rule_drug = (rd.get("drug_name") or "").lower()
                if not rule_drug or rule_drug not in drug_lower:
                    continue
                mod_type = rd.get("modification", "")
                alternative = rd.get("alternative", "")
                reason = rd.get("reason", "")
                sev = "critical" if mod_type == "contraindicated" else "warning"
                if mod_type == "contraindicated":
                    safe = False
                msg = (
                    f"STG {r['condition_stg_code']}: {d['name']} is {mod_type} "
                    f"in {r['condition_name']}. {reason}"
                )
                if alternative:
                    msg += f" Alternative: {alternative}."
                warnings.append({
                    "type": "condition_drug_modifier",
                    "severity": sev,
                    "message": msg,
                    "condition": r["condition_name"],
                    "stg_code": r["condition_stg_code"],
                })

    return {
        "drug_name": d["name"],
        "safe": safe and len([w for w in warnings if w["severity"] in ("critical", "warning")]) == 0,
        "warnings": warnings,
        "pregnancy_safe": d.get("pregnancy_safe"),
        "adult_dose": d.get("adult_dose", ""),
        "routes": d.get("routes", []),
    }


async def _exec_suggest_alternative(conn: asyncpg.Connection, params: dict) -> dict:
    """Find alternative medicines for a condition when one drug is excluded."""
    condition_name = params["condition_name"]
    excluded_drug = params["excluded_drug"].lower().strip()
    reason = params.get("reason", "contraindicated")

    # Resolve condition
    row = await conn.fetchrow(
        "SELECT id, name, stg_code FROM conditions WHERE name ILIKE $1 LIMIT 1",
        f"%{condition_name}%",
    )
    if not row:
        return {"error": f"Condition '{condition_name}' not found in STG database."}

    # Get all medicines for this condition
    medicines = await conn.fetch("""
        SELECT m.name, m.adult_dose, m.adult_frequency, m.adult_duration,
               m.pregnancy_safe, m.paediatric_dose_mg_per_kg, m.routes,
               cm.treatment_line, cm.dose_context, cm.special_notes
        FROM condition_medicines cm
        JOIN medicines m ON m.id = cm.medicine_id
        WHERE cm.condition_id = $1
        ORDER BY
            CASE cm.treatment_line
                WHEN 'first_line' THEN 1
                WHEN 'second_line' THEN 2
                WHEN 'alternative' THEN 3
                WHEN 'adjunct' THEN 4
                ELSE 5
            END
    """, row["id"])

    patient_pregnant = params.get("_patient_pregnant", False)
    patient_allergies = params.get("_patient_allergies", "")

    alternatives = []
    for m in medicines:
        d = dict(m)
        if excluded_drug in d["name"].lower():
            continue  # Skip the excluded drug
        # Skip allergy-related drugs
        if patient_allergies and patient_allergies.lower() in d["name"].lower():
            continue
        alt = {
            "name": d["name"],
            "treatment_line": d.get("treatment_line", ""),
            "adult_dose": d.get("adult_dose", ""),
            "adult_frequency": d.get("adult_frequency", ""),
            "adult_duration": d.get("adult_duration", ""),
            "pregnancy_safe": d.get("pregnancy_safe"),
            "routes": d.get("routes", []),
            "dose_context": d.get("dose_context", ""),
            "special_notes": d.get("special_notes", ""),
        }
        # Flag pregnancy-unsafe alternatives prominently
        if patient_pregnant and d.get("pregnancy_safe") is False:
            alt["pregnancy_warning"] = f"⚠ {d['name']} is NOT safe in pregnancy — do not use."
        elif patient_pregnant and d.get("pregnancy_safe") is None:
            alt["pregnancy_warning"] = f"Pregnancy safety unknown for {d['name']} — verify before prescribing."
        elif patient_pregnant and d.get("pregnancy_safe") is True:
            alt["pregnancy_safe_confirmed"] = True
        alternatives.append(alt)

    # Sort: pregnancy-safe first when patient is pregnant
    if patient_pregnant:
        alternatives.sort(key=lambda a: (
            0 if a.get("pregnancy_safe_confirmed") else (2 if a.get("pregnancy_warning", "").startswith("⚠") else 1)
        ))

    return {
        "condition": row["name"],
        "stg_code": row["stg_code"],
        "excluded_drug": excluded_drug,
        "exclusion_reason": reason,
        "alternatives": alternatives,
        "total_alternatives": len(alternatives),
        "patient_context": {"pregnant": patient_pregnant} if patient_pregnant else {},
    }


async def _exec_draft_referral_letter(conn: asyncpg.Connection, params: dict) -> dict:
    """Build referral letter context from encounter data + STG.

    Returns a structured dict that Haiku will format into a professional letter.
    The heavy lifting (formatting, tone) is done by Haiku, not this function.
    """
    referral_reason = params["referral_reason"]
    destination = params.get("referral_destination", "District Hospital")
    urgency = params.get("urgency", "routine")
    treatment_given = params.get("treatment_given", "")

    # Include patient demographics deterministically — don't rely on Haiku
    # to extract them from the system prompt
    patient_info = {}
    if params.get("_patient_age"):
        patient_info["age"] = params["_patient_age"]
    if params.get("_patient_sex"):
        patient_info["sex"] = params["_patient_sex"]
    if params.get("_patient_pregnant"):
        patient_info["pregnant"] = True
    if params.get("_patient_allergies"):
        patient_info["allergies"] = params["_patient_allergies"]
    if params.get("_patient_medications"):
        patient_info["current_medications"] = params["_patient_medications"]
    if params.get("_patient_vitals"):
        patient_info["vitals"] = params["_patient_vitals"]
    if params.get("_diagnosis"):
        patient_info["diagnosis"] = params["_diagnosis"]
    if params.get("_stg_code"):
        patient_info["stg_code"] = params["_stg_code"]

    return {
        "template": "referral_letter",
        "referral_reason": referral_reason,
        "referral_destination": destination,
        "urgency": urgency,
        "treatment_given": treatment_given,
        "patient": patient_info,
        "instructions": (
            "Compose a professional referral letter using the patient data provided in this "
            "tool result AND the encounter context from the system prompt. "
            "Include: Date, From (this PHC facility), To (destination), Patient demographics "
            "(age, sex, pregnancy status), Diagnosis with STG code, Relevant vitals, "
            "Allergies, Current medications, Treatment given at PHC, "
            "Reason for referral, Urgency, and Requested action."
        ),
    }


async def _exec_search_knowledge_base(conn: asyncpg.Connection, params: dict) -> dict:
    """Search markdown knowledge base files (no DB needed, conn ignored)."""
    query = params.get("query", "")
    source = params.get("source", "all")
    max_results = params.get("max_results", 5)

    results = search_markdown_kb(query, source=source, max_results=max_results)

    if not results:
        return {
            "found": False,
            "message": f"No knowledge base results for '{query}'" + (f" in {source}" if source != "all" else ""),
            "sections": [],
        }

    return {
        "found": True,
        "query": query,
        "source_filter": source,
        "total_results": len(results),
        "sections": results,
    }


# Tool dispatch
TOOL_DISPATCH = {
    "search_guidelines": _exec_search_guidelines,
    "lookup_condition": _exec_lookup_condition,
    "check_red_flags": _exec_check_red_flags,
    "search_medications": _exec_search_medications,
    "find_conditions": _exec_find_conditions,
    "check_drug_safety": _exec_check_drug_safety,
    "suggest_alternative": _exec_suggest_alternative,
    "draft_referral_letter": _exec_draft_referral_letter,
    "search_knowledge_base": _exec_search_knowledge_base,
}

# Tool name → friendly label for transparency
TOOL_LABELS = {
    "search_guidelines": "Searched STG guidelines",
    "lookup_condition": "Looked up condition details",
    "check_red_flags": "Checked danger signs",
    "search_medications": "Looked up medication",
    "find_conditions": "Searched conditions",
    "check_drug_safety": "Checked drug safety",
    "suggest_alternative": "Found alternative medications",
    "draft_referral_letter": "Drafted referral letter",
    "search_knowledge_base": "Searched extended knowledge base",
}


# ── Deterministic patient context injection ─────────────────────────────────
# Never trust Haiku to relay patient data correctly. Inject from encounter context.


def _inject_patient_context(block, pt_ctx: dict, encounter_context: dict):
    """Inject real patient context into tool parameters before execution.

    This ensures tools get ground-truth patient data (pregnancy, age, sex,
    medications, allergies) regardless of what Haiku passes in its tool call.
    """
    name = block.name

    # ── check_drug_safety: override ALL patient fields ──
    if name == "check_drug_safety":
        if pt_ctx.get("pregnancy_status"):
            old = block.input.get("pregnancy_status", "NOT SET")
            block.input["pregnancy_status"] = pt_ctx["pregnancy_status"]
            if old != pt_ctx["pregnancy_status"]:
                logger.info(f"[INJECT] check_drug_safety: pregnancy_status {old} → {pt_ctx['pregnancy_status']}")
        if pt_ctx.get("age") is not None:
            block.input["patient_age"] = pt_ctx["age"]
        if pt_ctx.get("sex"):
            block.input["patient_sex"] = pt_ctx["sex"]
        if pt_ctx.get("current_medications"):
            # Always override — Haiku often passes incomplete list
            block.input["current_medications"] = pt_ctx["current_medications"]

    # ── find_conditions: inject age/sex for demographic filtering ──
    elif name == "find_conditions":
        if pt_ctx.get("age") is not None and "patient_age" not in block.input:
            block.input["patient_age"] = pt_ctx["age"]
        if pt_ctx.get("sex") and "patient_sex" not in block.input:
            block.input["patient_sex"] = pt_ctx["sex"]

    # ── suggest_alternative: inject pregnancy context so results are filtered ──
    elif name == "suggest_alternative":
        if pt_ctx.get("is_pregnant"):
            block.input["_patient_pregnant"] = True
        if pt_ctx.get("allergies"):
            block.input["_patient_allergies"] = pt_ctx["allergies"]

    # ── draft_referral_letter: inject full patient demographics ──
    elif name == "draft_referral_letter":
        patient = encounter_context.get("patient", {})
        if patient:
            block.input["_patient_age"] = patient.get("age")
            block.input["_patient_sex"] = patient.get("sex")
        if pt_ctx.get("is_pregnant"):
            block.input["_patient_pregnant"] = True
        vitals = encounter_context.get("vitals", {})
        if vitals:
            block.input["_patient_vitals"] = vitals
        condition = encounter_context.get("condition")
        if condition:
            if isinstance(condition, dict):
                block.input["_diagnosis"] = condition.get("name", "")
                block.input["_stg_code"] = condition.get("stg_code", "")
            else:
                block.input["_diagnosis"] = str(condition)
        if pt_ctx.get("allergies"):
            block.input["_patient_allergies"] = pt_ctx["allergies"]
        if pt_ctx.get("current_medications"):
            block.input["_patient_medications"] = pt_ctx["current_medications"]

    # ── search_medications: inject pregnancy flag for result annotation ──
    elif name == "search_medications":
        if pt_ctx.get("is_pregnant"):
            block.input["_patient_pregnant"] = True

    # ── lookup_condition: inject pregnancy flag to annotate medicines ──
    elif name == "lookup_condition":
        if pt_ctx.get("is_pregnant"):
            block.input["_patient_pregnant"] = True
        if pt_ctx.get("allergies"):
            block.input["_patient_allergies"] = pt_ctx["allergies"]


# ── ClinicalAssistant class ─────────────────────────────────────────────────


class ClinicalAssistant:
    """Conversational clinical Q&A agent with STG-grounded tool use."""

    def __init__(self, pool: asyncpg.Pool):
        self.client = anthropic.AsyncAnthropic(max_retries=3)
        self.pool = pool
        self.models = [
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-5-20250929",
        ]

    async def _call_with_fallback(self, **kwargs) -> anthropic.types.Message:
        """Try each model in order; fall back on 429/529 errors."""
        last_err = None
        for model in self.models:
            try:
                return await self.client.messages.create(
                    model=model,
                    max_tokens=2048,
                    temperature=0.1,
                    tools=ASSISTANT_TOOLS,
                    **kwargs,
                )
            except anthropic.APIStatusError as e:
                if e.status_code in (429, 529):
                    logger.warning(f"{model} unavailable ({e.status_code}), trying next model")
                    last_err = e
                    continue
                raise
        raise last_err

    async def chat(
        self,
        message: str,
        conversation_history: list[dict],
        encounter_context: dict | None = None,
    ) -> dict:
        """
        Process a user message with tool use.

        Args:
            message: The nurse's question
            conversation_history: Previous messages from DB (list of {role, content, tool_calls})
            encounter_context: Current patient/encounter state

        Returns:
            {
                "response": str,          # Assistant's text response
                "sources": [...],         # STG sources referenced
                "tools_used": [...],      # Tool names called
                "tool_calls_detail": [...] # Full tool call/result pairs for DB storage
            }
        """
        system = _build_system_prompt(encounter_context)

        # Build messages array from history + new message
        messages = self._build_messages(conversation_history, message)

        tools_used = []
        tool_calls_detail = []
        sources = []
        max_tool_rounds = 4

        async with self.pool.acquire() as conn:
            for _round in range(max_tool_rounds):
                response = await self._call_with_fallback(
                    system=system, messages=messages,
                )

                # Check if done (text response, no more tool calls)
                if response.stop_reason == "end_turn":
                    break

                # Process tool calls
                tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
                if not tool_use_blocks:
                    break

                # Add assistant message with tool_use blocks
                messages.append({
                    "role": "assistant",
                    "content": [
                        {"type": b.type, **({"text": b.text} if b.type == "text" else {"id": b.id, "name": b.name, "input": b.input})}
                        for b in response.content
                    ],
                })

                # ── Extract patient context ONCE for deterministic injection ──
                # All tools get real patient data, not what Haiku decides to pass
                _pt_ctx = {}
                if encounter_context:
                    _pt = encounter_context.get("patient", {})
                    _preg = _extract_pregnancy_status(encounter_context)
                    _pt_ctx = {
                        "pregnancy_status": _preg,
                        "is_pregnant": _preg and _preg.lower() in ("pregnant", "yes", "true"),
                        "age": _pt.get("age"),
                        "sex": _pt.get("sex"),
                    }
                    # Resolve current medications from all sources
                    _meds = encounter_context.get("current_medications")
                    if isinstance(_meds, str):
                        _meds = [m.strip() for m in _meds.split(",") if m.strip()]
                    _pt_ctx["current_medications"] = _meds or []
                    # Allergies
                    _history = encounter_context.get("history", encounter_context.get("core_history", {}))
                    if not isinstance(_history, dict):
                        _history = {}
                    _allergies = _history.get("allergies") or encounter_context.get("allergies")
                    if isinstance(_allergies, list):
                        _allergies = ", ".join(str(a) for a in _allergies)
                    if _allergies and str(_allergies).lower().strip() in ("nkda", "none", "no", "no known drug allergies", "nil", ""):
                        _allergies = None
                    _pt_ctx["allergies"] = _allergies

                # Execute each tool and collect results
                tool_results = []
                for block in tool_use_blocks:
                    handler = TOOL_DISPATCH.get(block.name)
                    if handler:
                        # ── Deterministic context injection ──
                        # Override/fill patient data so tools use encounter truth,
                        # not whatever Haiku decides to pass
                        if encounter_context:
                            _inject_patient_context(block, _pt_ctx, encounter_context)
                        try:
                            result = await handler(conn, block.input)
                        except Exception as e:
                            logger.error(f"Tool {block.name} failed: {e}", exc_info=True)
                            result = {"error": f"Tool execution failed: {str(e)}"}
                    else:
                        result = {"error": f"Unknown tool: {block.name}"}

                    tools_used.append(block.name)
                    tool_calls_detail.append({
                        "tool": block.name,
                        "input": block.input,
                        "output": result,
                    })

                    # Extract sources from search results
                    if block.name == "search_guidelines":
                        for chunk in result.get("chunks", []):
                            sources.append({
                                "stg_code": chunk.get("stg_code", ""),
                                "condition_name": chunk.get("condition_name", ""),
                                "section_role": chunk.get("section_role", ""),
                                "excerpt": (chunk.get("text", "") or "")[:300],
                            })
                    elif block.name == "lookup_condition":
                        if "stg_code" in result:
                            sources.append({
                                "stg_code": result.get("stg_code", ""),
                                "condition_name": result.get("name", ""),
                                "section_role": "FULL_ENTRY",
                                "excerpt": (result.get("description", "") or "")[:300],
                            })
                    elif block.name == "search_knowledge_base":
                        for section in result.get("sections", []):
                            sources.append({
                                "stg_code": "",
                                "condition_name": section.get("parent_heading", ""),
                                "section_role": section.get("heading", ""),
                                "excerpt": (section.get("content", "") or "")[:300],
                                "source_label": section.get("source_label", ""),
                                "source_file": section.get("source_file", ""),
                            })

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                    })

                # Add tool results as user message
                messages.append({"role": "user", "content": tool_results})

        # Extract final text response
        response_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                response_text += block.text

        # Deduplicate tools_used
        unique_tools = list(dict.fromkeys(tools_used))

        return {
            "response": response_text,
            "sources": sources,
            "tools_used": unique_tools,
            "tool_calls_detail": tool_calls_detail,
        }

    def _build_messages(
        self,
        history: list[dict],
        new_message: str,
    ) -> list[dict]:
        """Build Anthropic messages array from conversation history + new message.

        History entries have: role, content, tool_calls (list of call/result pairs).
        We reconstruct the full message sequence for multi-turn context.
        """
        messages = []

        for entry in history:
            role = entry["role"]
            content = entry["content"]

            if role == "user":
                messages.append({"role": "user", "content": content})
            elif role == "assistant":
                # If this turn had tool calls, we need to replay them
                tc = entry.get("tool_calls") or []
                if tc:
                    # For simplicity in history replay, just include the final text
                    # The full tool call sequence is too complex to replay accurately
                    # and Haiku doesn't need it for context
                    messages.append({"role": "assistant", "content": content})
                else:
                    messages.append({"role": "assistant", "content": content})

        # Add the new user message
        messages.append({"role": "user", "content": new_message})

        return messages
