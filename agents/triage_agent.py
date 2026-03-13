"""
Triage Agent
------------
Core clinical triage logic using Anthropic SDK tool_use.

Two models:
  - claude-haiku-4-5-20251001 for the tool-calling loop (cheap, fast, 4-6 turns)
  - claude-sonnet-4-6 for final structured synthesis (one call, quality matters)

Agent loop:
  complaint → extract_symptoms → expand_synonyms → search_conditions
           → score_differential → check_safety_flags → get_condition_detail (top 3)
           → Sonnet synthesises final JSON matching frontend contract
"""

import asyncio
import json
import logging
import re
import anthropic
import asyncpg
from typing import Optional

from agents.tools import TOOL_HANDLERS
from agents.sats import compute_sats_acuity
from agents.question_engine import (
    select_assessment_questions as _select_rule_questions,
    get_referral_triggers,
    classify_severity,
    match_lab_rules,
    check_vital_rules,
)
from db.database import get_condition_rich_content, get_condition_features_batch, get_condition_details_batch

logger = logging.getLogger(__name__)

# Cache injected by api/main.py at startup
_cache = None  # type: ignore  # ClinicalDataCache


# ── STG text cleaning ────────────────────────────────────────────────────────

def _clean_stg_references(text: str) -> str:
    """Remove broken cross-references and academic markers from STG text.

    The raw STG PDF contains navigation cues (see Section X, table below, etc.)
    and evidence-level tags (LoE:IIIb16) that don't make sense in our app.
    This strips them while preserving the clinical content.
    """
    if not text:
        return text

    # Strip PDF extraction noise: [Page 401], # CHAPTER 12: ..., page headers
    text = re.sub(r"\[Page\s+\d+\]\s*", "", text)
    text = re.sub(r"^#+\s*CHAPTER\s+\d+:.*$", "", text, flags=re.MULTILINE)
    # Strip STG internal page numbers at start of text: "12.5\n" or "4.14"
    text = re.sub(r"^\d+\.\d+\s*\n", "", text)

    # Strip evidence level markers: LoE:IIb41, LoE:IIIb16, LoE: IVb15, etc.
    text = re.sub(r"\s*LoE:\s*[A-Za-z0-9]+", "", text)

    # ── Parenthetical cross-references (handle FIRST to avoid orphaned text) ──
    # "(See Section 4.1: Prevention of ...)" → removed entirely
    text = re.sub(
        r"\s*\(\s*[Ss]ee\s+[Ss]ection\s+[\d.]+(?::\s*[^)]+)?\s*\)",
        "",
        text,
    )
    # "(See Chapter 11: HIV and AIDS)" → removed entirely
    text = re.sub(
        r"\s*\(\s*[Ss]ee\s+[Cc]hapter\s+\d+(?::\s*[^)]+)?\s*\)",
        "",
        text,
    )
    # "(Refer to Section 11.8.2: Candidiasis)" → removed entirely
    text = re.sub(
        r"\s*\(\s*[Rr]efer\s+to\s+[Ss]ection\s+[\d.]+(?::\s*[^)]+)?\s*\)",
        "",
        text,
    )
    # "(See Table 4.5: Treatment of...)" → removed entirely
    text = re.sub(
        r"\s*\(\s*[Ss]ee\s+[Tt]able\s+[\d.]+(?::\s*[^)]+)?\s*\)",
        "",
        text,
    )

    # ── Non-parenthetical cross-references ──
    # "See Section 4.1: Prevention of ..." → "(Ref: Section 4.1)"
    text = re.sub(
        r"[Ss]ee [Ss]ection\s+([\d.]+):\s*[^.)\n]+",
        r"(Ref: Section \1)",
        text,
    )
    text = re.sub(
        r"[Ss]ee [Ss]ection\s+([\d.]+)",
        r"(Ref: Section \1)",
        text,
    )

    # "See Chapter 11: HIV and AIDS" → "(Ref: Chapter 11)"
    text = re.sub(
        r"[Ss]ee [Cc]hapter\s+(\d+):\s*[^.)\n]+",
        r"(Ref: Chapter \1)",
        text,
    )
    text = re.sub(
        r"[Ss]ee [Cc]hapter\s+(\d+)",
        r"(Ref: Chapter \1)",
        text,
    )

    # "Refer to Section/Figure" variants
    text = re.sub(
        r"[Rr]efer to [Ss]ection\s+([\d.]+):\s*[^.)\n]+",
        r"(Ref: Section \1)",
        text,
    )
    text = re.sub(
        r"[Rr]efer to [Ss]ection\s+([\d.]+)",
        r"(Ref: Section \1)",
        text,
    )
    text = re.sub(
        r"\s*\(?[Rr]efer to [Ff]igure\s+[\d.]+\s*(below|above)?\s*\.?\)?\s*",
        " ",
        text,
    )

    # "see/refer to table below/above [for/on X]" → remove
    text = re.sub(
        r"\s*\(?(?:[Ss]ee|[Rr]efer to)\s+[Tt]able\s*(below|above)\s*(?:(?:for|on)\s+[^.)\n]*)?\s*\.?\)?\s*",
        " ",
        text,
    )

    # "See Table 4.5: Treatment of..." → "(Ref: Table 4.5)"
    text = re.sub(
        r"[Ss]ee [Tt]able\s+([\d.]+):\s*[^.)\n]+",
        r"(Ref: Table \1)",
        text,
    )
    text = re.sub(
        r"[Ss]ee [Tt]able\s+([\d.]+)",
        r"(Ref: Table \1)",
        text,
    )

    # ── Convert ALL-CAPS section headers to markdown headers ──
    # "CLINICAL PRESENTATION" → "## Clinical Presentation"
    # "MONITORING AND FOLLOW-UP" → "## Monitoring And Follow-Up"
    text = re.sub(
        r"^([A-Z][A-Z\s&/,\-]{4,})$",
        lambda m: "## " + m.group(1).strip().title(),
        text,
        flags=re.MULTILINE,
    )

    # ── Cleanup ──
    # Fix double/nested parens from removals
    text = re.sub(r"\(\(Ref:", "(Ref:", text)
    text = re.sub(r"\)\)", ")", text)

    # Clean up double spaces left by removals
    text = re.sub(r"  +", " ", text)
    # Clean up orphaned parens: "( )" or "()"
    text = re.sub(r"\(\s*\)", "", text)
    # Clean up orphaned ", ," or " , " left from removals
    text = re.sub(r",\s*,", ",", text)
    # Clean up trailing orphaned prepositions before periods: " for." → "."
    text = re.sub(r"\s+(for|of|to|from|in|on|with|and|or)\s*\.", ".", text)
    # Clean up space before punctuation: " ." → "." and " ," → ","
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    # Remove blank lines at start of text
    text = re.sub(r"^\s*\n", "", text)

    return text.strip()


# ── STG text formatting ──────────────────────────────────────────────────────

def _format_stg_text(raw: str, max_summary: int = 300) -> dict:
    """Format raw STG text into structured markdown with a concise summary.

    Returns {"summary": str, "full": str}.
    - summary: first meaningful sentences, capped at max_summary chars
    - full: cleaned text with proper markdown bullet points and line breaks
    """
    if not raw or not raw.strip():
        return {"summary": "", "full": ""}

    text = _clean_stg_references(raw.strip())

    # Normalise bullet markers to markdown "- "
    text = re.sub(r"^[»►•●]\s*", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"^[–—]\s+", "- ", text, flags=re.MULTILINE)

    # Ensure blank line before bullet lists so markdown renders correctly
    text = re.sub(r"([^\n])\n(- )", r"\1\n\n\2", text)

    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Build summary: first prose sentences only (skip bullets, headers, tables)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    summary_parts = []
    summary_len = 0
    for para in paragraphs:
        # Skip non-prose content for the summary
        if para.startswith("- ") or para.startswith("## ") or "|" in para:
            continue
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", para)
        for sent in sentences:
            if summary_len + len(sent) > max_summary and summary_parts:
                break
            summary_parts.append(sent)
            summary_len += len(sent) + 1
        if summary_len >= max_summary:
            break

    summary = " ".join(summary_parts).strip()
    if not summary and paragraphs:
        # Fallback: first paragraph truncated
        summary = paragraphs[0][:max_summary]
        if len(paragraphs[0]) > max_summary:
            summary = summary.rsplit(" ", 1)[0] + "..."

    return {"summary": summary, "full": text}


def _split_to_bullet_list(raw: str) -> list[str]:
    """Split STG text into a list of individual items (for danger_signs, referral_criteria).

    Handles bullet markers (», -, •), numbered items, and newline-separated entries.
    Returns a list of clean strings.
    """
    if not raw or not raw.strip():
        return []

    text = _clean_stg_references(raw.strip())

    # Normalise bullet markers
    text = re.sub(r"^[»►•●]\s*", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"^[–—]\s+", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+[.)]\s+", "- ", text, flags=re.MULTILINE)

    items = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip leading "- " marker
        if line.startswith("- "):
            line = line[2:].strip()
        if line and len(line) > 3:
            items.append(line)

    return items


SYSTEM_PROMPT = """You are a clinical triage assistant for South African primary healthcare nurses.
You have access to the digitised SA Standard Treatment Guidelines (STG/EML) through database tools.

YOUR PROCESS — call tools in this order:
1. extract_symptoms — parse the nurse's complaint into standardised clinical terms
2. expand_synonyms — broaden with patient language variants from the database
3. search_conditions — find matching conditions from the knowledge graph
   IMPORTANT: pass BOTH the expanded terms as "symptoms" AND the original
   extracted terms as "original_symptoms" so scoring can prioritise conditions
   that match multiple distinct complaints
4. score_differential — rank conditions by symptom match + context
5. check_safety_flags — check for RED_FLAG danger signs and vitals thresholds
6. get_condition_detail — fetch full STG entry for top 3 conditions

RULES:
- NEVER search by condition name — always match through symptoms
- Always check for RED_FLAGS — patient safety is paramount
- Be specific in clinical terms: "epigastric pain" not "pain"
- Include patient age/sex/pregnancy context when relevant
- Source everything to STG sections

After calling all tools, provide a brief clinical summary of your findings.
"""


SYNTHESIS_PROMPT = """Produce triage JSON. Use ONLY verified conditions below.

Complaint: {complaint} | Patient: {patient} | Vitals: {vitals}

Conditions (from STG knowledge graph search, ranked by feature match score):
{verified_conditions}

Safety: {safety_data}

ACUITY SELECTION (use vitals_acuity from safety data as baseline):
- "routine": normal vitals, no red flags — THIS IS THE DEFAULT
- "priority": mild vital abnormality (temp >= 39C, HR > 120 or < 50) OR a single confirmed red flag
- "urgent": severe vitals (BP >= 180, SpO2 < 92, RR >= 30) OR multiple red flags
- Do NOT escalate beyond the vitals_acuity unless red_flags are listed in the safety data.
  Speculation about unconfirmed conditions is NOT grounds for escalation.

REQUIREMENTS:
- Return 4-5 conditions from the verified list above
- The conditions above are ranked by automated STG feature matching. Use your clinical
  judgment to re-rank based on the specific complaint, patient context, and vitals.
  Conditions marked "(lab-confirmed)" MUST stay at #1 — lab results are confirmed diagnoses.
  Conditions marked "(vitals-based)" MUST stay at the top — vitals are direct measurements.
- condition_symptoms: 3-4 verification questions per condition. Use DANGER SIGNS and
  KEY FEATURES provided for each condition to ask clinically specific questions.
  Prioritize: (1) danger sign symptoms that determine urgency,
  (2) features that distinguish between the top 2-3 conditions,
  (3) features NOT already in the complaint that would confirm/rule out the condition.
  Ask "Does the pain worsen when swallowing liquids?" NOT "When did you first notice?"
- extracted_symptoms: include all identified symptoms as descriptive phrases
- assessment_questions: 4-5 discriminating questions that help differentiate between
  the top 2-3 conditions. Focus on features present in one condition but absent in others.
- condition_code/condition_name MUST match verified list exactly
- source_references format "STG X.Y"
- confidence: top condition 0.80-0.95, decrease for lower-ranked conditions

JSON schema:
{{"extracted_symptoms":["symptom phrase"],"acuity":"routine|priority|urgent","acuity_reasons":["short reason"],"acuity_sources":["STG X.Y"],"conditions":[{{"condition_code":"X.Y","condition_name":"Name","confidence":0.85,"matched_symptoms":["syms"],"reasoning":"brief why","source_references":["STG X.Y"]}}],"condition_symptoms":{{"Condition Name":[{{"id":"cs_X.Y_1","question":"Do you have...?"}}]}},"needs_assessment":true,"assessment_questions":[{{"id":"hyp_X.Y_1","question":"Q?","type":"yes_no","required":false,"round":1,"source_citation":"STG X.Y","grounding":"verified"}}]}}

Return ONLY valid JSON."""


REFINE_SYNTHESIS_PROMPT = """Based on the triage refinement data below, produce a JSON response.

## Context
Complaint: {complaint}
Current conditions: {conditions}
All answers so far: {all_time_answers}
Current round: {current_round}

## Confirmed symptoms (answered "yes"): {confirmed}
## Denied symptoms (answered "no"): {denied}
## Condition details: {condition_details}

## VERIFIED CONDITIONS (use ONLY these stg_codes and names)
{verified_conditions}

## Required JSON Schema
Return a JSON object with these exact fields:
{{
  "refinement_source": "rules",
  "conditions": [
    {{
      "condition_code": "MUST be a stg_code from the verified list",
      "condition_name": "MUST be the exact name from the verified list",
      "confidence": 0.85,
      "reasoning": "updated reasoning based on answers"
    }}
  ],
  "next_round_questions": [
    {{
      "id": "unique_id",
      "question": "focused follow-up question",
      "type": "yes_no",
      "required": false,
      "round": {next_round},
      "source_citation": "STG <stg_code> from verified list, or empty string",
      "grounding": "verified if traceable to a stg_code, unverified otherwise"
    }}
  ],
  "red_flag_alert": null
}}

CRITICAL — CITATION RULES:
- condition_code and condition_name MUST come from the verified list, no exceptions
- source_citation MUST use format "STG <stg_code>" from verified list, or be empty string
- Do NOT invent STG section numbers, names, or page numbers
- grounding is "verified" ONLY if source_citation references a verified stg_code

OTHER RULES:
- Adjust confidence UP for conditions whose features were confirmed, DOWN for denied
- Generate 2-4 NEW questions that discriminate between the remaining likely conditions
- Focus on features that are present in one top condition but not others
- If any confirmed symptom is a RED_FLAG, set red_flag_alert to a description string
- If round >= 5 or conditions are clearly differentiated, set next_round_questions to null
- Return ONLY valid JSON
"""


SLIM_ASSESSMENT_PROMPT = """Re-rank these conditions and generate assessment questions.

Complaint: {complaint} | Patient: {patient}

Conditions (from STG, current order by feature-match score):
{conditions_summary}

TASK 1 — RANK: Re-order ALL condition codes by clinical likelihood. "(lab-confirmed)" stays #1. "(vitals-based)" stays at top.
TASK 2 — QUESTIONS: 4-5 yes/no questions to differentiate top conditions. Target features present in one but absent in others. Do NOT ask about symptoms already in the complaint.

Return ONLY valid JSON:
{{"ranked_codes":["code1","code2"],"questions":["Does the patient have...?","Is there any...?"]}}"""


class TriageAgent:

    def __init__(self, pool: asyncpg.Pool):
        self.client = anthropic.AsyncAnthropic(max_retries=3)
        self.pool = pool
        self.haiku = "claude-haiku-4-5-20251001"
        self.sonnet = "claude-sonnet-4-6"
        self._fallback_models = [self.haiku, "claude-sonnet-4-5-20250929"]

    async def _call_with_fallback(self, **kwargs) -> anthropic.types.Message:
        """Try each model in order; fall back on 429/529 errors."""
        last_err = None
        for model in self._fallback_models:
            try:
                return await self.client.messages.create(model=model, **kwargs)
            except anthropic.APIStatusError as e:
                if e.status_code in (429, 529):
                    logger.warning(f"{model} unavailable ({e.status_code}), trying fallback")
                    last_err = e
                    continue
                raise
        raise last_err

    # ── Main analyse endpoint ────────────────────────────────────────────────

    async def analyze(
        self,
        complaint: str,
        patient: Optional[dict] = None,
        vitals: Optional[dict] = None,
        core_history: Optional[dict] = None,
        lab_results: Optional[list] = None,
    ) -> dict:
        """
        Run full triage analysis. Returns AnalyzeResponse-shaped dict.

        Pipeline (3 LLM calls, rest are direct DB queries):
        1. Haiku: extract symptoms from complaint (temperature=0)
        2. DB: expand_synonyms → search_conditions → score → safety → detail
        3. Sonnet: synthesise final JSON (temperature=0)
        """
        import time
        t0 = time.monotonic()

        # ── Step 1: Extract symptoms with Haiku (single LLM call) ─────────
        patient_context = ""
        is_child = False
        is_pregnant = False
        patient_sex = None
        pregnancy_status = "unknown"
        if patient:
            age = patient.get("age")
            sex = patient.get("sex", "unknown")
            preg = patient.get("pregnancy_status", "unknown")
            pregnancy_status = preg
            patient_context = f"Patient: age {age}, sex {sex}, pregnancy: {preg}"
            if age and age < 12:
                is_child = True
            if preg == "pregnant":
                is_pregnant = True
            if sex in ("male", "female"):
                patient_sex = sex

        # ── Deterministic lab result scan (pre-LLM) ─────────────────
        structured_labs = None
        if lab_results:
            structured_labs = [
                {"test_name": lr.get("test_name", ""), "result": lr.get("result", "")}
                if isinstance(lr, dict) else
                {"test_name": getattr(lr, "test_name", ""), "result": getattr(lr, "result", "")}
                for lr in lab_results
            ]
        patient_age_for_lab = patient.get("age") if patient else None
        lab_matches = self._extract_lab_results(complaint, pregnancy_status, structured_labs, patient_age=patient_age_for_lab)
        if lab_matches:
            logger.info(f"Lab results detected: {[lr['id'] for lr in lab_matches]}")

        vitals_context = ""
        if vitals:
            vitals_context = ", ".join(f"{k}={v}" for k, v in vitals.items() if v is not None)

        extract_prompt = (
            f"Extract SYMPTOMS from this complaint. For EACH symptom, include BOTH the "
            f"common/patient-friendly term AND the formal clinical term.\n\n"
            f"Examples:\n"
            f"- 'sore throat' AND 'pharyngitis'\n"
            f"- 'headache' AND 'cephalgia'\n"
            f"- 'stiff neck' AND 'neck stiffness' AND 'nuchal rigidity'\n"
            f"- 'burning when urinating' AND 'dysuria'\n"
            f"- 'throwing up' AND 'vomiting'\n\n"
            f"IMPORTANT: Only extract SYMPTOMS (what the patient feels/reports). "
            f"Do NOT include vital sign readings (blood pressure, heart rate, temperature values, "
            f"SpO2, respiratory rate). Vitals are recorded separately.\n\n"
            f"Complaint: {complaint}\n"
            f"{patient_context}\n\n"
            f"Return ONLY a flat JSON array with ALL symptom terms (both common and clinical), "
            f"e.g. [\"sore throat\", \"pharyngitis\", \"fever\", \"high temperature\"]"
        )

        try:
            extract_response = await self._call_with_fallback(
                max_tokens=512,
                temperature=0,
                messages=[{"role": "user", "content": extract_prompt}],
            )
            raw = extract_response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            symptoms = json.loads(raw)
            if not isinstance(symptoms, list):
                symptoms = [str(symptoms)]
            symptoms = [s.lower().strip() for s in symptoms if s.strip()]
        except Exception as e:
            logger.error(f"Symptom extraction failed: {e}")
            # Fallback: use complaint words
            symptoms = [w.lower().strip() for w in complaint.split() if len(w) > 3]

        # Inject lab-derived symptom terms into the symptom list
        if lab_matches:
            for lr in lab_matches:
                for sym in lr.get("add_symptoms", []):
                    if sym.lower() not in symptoms:
                        symptoms.append(sym.lower())
            logger.info(f"Symptoms after lab injection: {symptoms}")

        logger.info(f"[{time.monotonic()-t0:.1f}s] Extracted symptoms: {symptoms}")
        tool_results = {
            "extract_symptoms": {"symptoms": symptoms, "count": len(symptoms)},
        }

        # ── Step 2: Direct DB calls (no LLM — all STG-grounded) ─────────
        # 2a. Expand synonyms
        expand_result = await TOOL_HANDLERS["expand_synonyms"](
            {"clinical_terms": symptoms}, self.pool
        )
        tool_results["expand_synonyms"] = expand_result
        expanded_terms = expand_result.get("expanded_terms", symptoms)
        logger.info(f"[{time.monotonic()-t0:.1f}s] Expanded to {len(expanded_terms)} terms")

        # Extract medications from core history
        medications = []
        if core_history and core_history.get("medications"):
            meds_raw = core_history["medications"]
            if isinstance(meds_raw, str):
                medications = [m.strip() for m in meds_raw.replace(",", ";").split(";") if m.strip()]
            elif isinstance(meds_raw, list):
                medications = [str(m).strip() for m in meds_raw if m]

        # 2b. Search conditions (pure DB search — graph + STG text fallbacks)
        patient_age = patient.get("age") if patient else None
        search_result = await TOOL_HANDLERS["search_conditions"](
            {"symptoms": expanded_terms, "original_symptoms": symptoms,
             "patient_is_child": is_child, "patient_is_pregnant": is_pregnant,
             "patient_sex": patient_sex, "patient_age": patient_age,
             "pregnancy_status": pregnancy_status,
             "medications": medications,
             "_skip_vector_search": True},
            self.pool,
        )
        tool_results["search_conditions"] = search_result
        conditions = search_result.get("conditions", [])
        top_names = [f"{c.get('name','')} ({c.get('adjusted_score',0):.3f})" for c in conditions[:5]]
        logger.info(f"[{time.monotonic()-t0:.1f}s] Found {len(conditions)} conditions: {top_names}")

        # 2b-extra. Vitals-based condition injection (100% STG-grounded)
        # STG defines Hypertension as systolic >= 140 mmHg or diastolic >= 90 mmHg.
        # If vitals show elevated BP, inject/boost Hypertension — the graph has no
        # symptom edges for headache/dizziness → hypertension.
        conditions = await self._inject_vitals_conditions(conditions, vitals or {}, patient_age=patient_age)

        # 2b-extra-2. Lab-confirmed condition injection (100% deterministic)
        if lab_matches:
            conditions = await self._inject_lab_conditions(conditions, lab_matches, patient_age=patient_age)

        # 2b-extra-3. Duration-aware scoring (100% deterministic)
        # Penalise self-limiting conditions for long durations (cold > 1 week),
        # boost chronic/infectious conditions when duration matches (TB > 2 weeks).
        conditions = self._apply_duration_modifiers(conditions, core_history or {})

        # Filter out pregnancy-required conditions from search results when
        # patient is explicitly not pregnant. Applied early so downstream
        # stages (details, features, LLM) never see them.
        pregnancy_required_codes = _cache.pregnancy_required_codes if _cache else set()
        if pregnancy_status.lower().strip() in (
            "not pregnant", "no", "none", "negative", "n/a",
        ):
            conditions = [
                c for c in conditions
                if c.get("stg_code", "") not in pregnancy_required_codes
            ]

        # Update tool_results so _extract_verified_conditions sees the filtered conditions
        tool_results["search_conditions"]["conditions"] = conditions

        # 2c. Parallel: safety flags + condition details + features + acuity
        condition_ids = [c["id"] for c in conditions[:15]]
        top_ids = [c["id"] for c in conditions[:10]]

        # Build condition_id map now (needed for features query)
        condition_id_map: dict[str, int] = {}
        for c in conditions:
            if c.get("stg_code") and c.get("id"):
                condition_id_map[c["stg_code"]] = c["id"]

        async def _get_safety():
            if not condition_ids:
                return {}
            return await TOOL_HANDLERS["check_safety_flags"](
                {"symptoms": expanded_terms, "condition_ids": condition_ids,
                 "vitals": vitals or {}}, self.pool)

        async def _get_details_batch():
            async with self.pool.acquire() as conn:
                raw = await get_condition_details_batch(conn, top_ids)
            # Convert to the format expected by the rest of the code
            import json as _json
            details = {}
            for cid, row in raw.items():
                meds = row.get("medicines_json", [])
                if isinstance(meds, str):
                    meds = _json.loads(meds)
                referral = row.get("referral_criteria", "[]")
                if isinstance(referral, str):
                    try:
                        referral = _json.loads(referral)
                    except (ValueError, TypeError):
                        referral = [referral] if referral else []
                details[cid] = {
                    "condition_id": row["id"],
                    "stg_code": row["stg_code"],
                    "name": row["name"],
                    "chapter": row.get("chapter_name", ""),
                    "description": row.get("description_text", ""),
                    "general_measures": row.get("general_measures", ""),
                    "medicine_treatment": row.get("medicine_treatment", ""),
                    "danger_signs": row.get("danger_signs", ""),
                    "referral_criteria": referral,
                    "medicines": meds,
                    "source_pages": row.get("source_pages", []),
                }
            return details

        async def _get_features():
            # Use top 5 from search results (don't need verified_sorted)
            top5_ids = [c["id"] for c in conditions[:5]]
            async with self.pool.acquire() as conn:
                return await get_condition_features_batch(conn, top5_ids)

        # Run ALL three DB queries in parallel
        safety_result, details, features_by_condition = await asyncio.gather(
            _get_safety(), _get_details_batch(), _get_features()
        )
        tool_results["check_safety_flags"] = safety_result
        tool_results["condition_details"] = details
        patient_age = patient.get("age") if patient else None
        acuity_info = self._compute_vitals_acuity(
            vitals or {}, complaint=complaint, symptoms=symptoms,
            patient_age=patient_age,
        )
        tool_results["vitals_acuity"] = acuity_info
        sats_colour = acuity_info.get("sats_colour", "green")
        tews_score = acuity_info.get("tews_score", 0)
        logger.info(f"[{time.monotonic()-t0:.1f}s] Safety + details + features done | SATS: {sats_colour} (TEWS {tews_score})")

        # ── Step 3: Build response (mostly deterministic) ────────────────
        # Strategy: launch LLM calls ASAP, then do deterministic work while they run.
        verified = self._extract_verified_conditions(tool_results)
        verified_sorted = sorted(
            verified.values(), key=lambda v: v.get("score", 0), reverse=True
        )

        # Promote vitals-injected conditions to the top
        for i, v in enumerate(verified_sorted):
            features = v.get("matched_features", [])
            if any("vitals-based" in f for f in features):
                if i > 0:
                    verified_sorted.insert(0, verified_sorted.pop(i))
                break

        # Promote lab-confirmed conditions to the top (above vitals)
        for i, v in enumerate(verified_sorted):
            features = v.get("matched_features", [])
            if any("lab-confirmed" in f for f in features):
                if i > 0:
                    verified_sorted.insert(0, verified_sorted.pop(i))
                break

        # Filter out pregnancy-required conditions when patient is explicitly
        # not pregnant. This must happen BEFORE the LLM re-ranking call,
        # otherwise the LLM promotes them back and position-based confidence
        # assignment (0.90, 0.78, ...) erases the deterministic penalty.
        preg_codes = _cache.pregnancy_required_codes if _cache else set()
        if pregnancy_status.lower().strip() in (
            "not pregnant", "no", "none", "negative", "n/a",
        ):
            before = len(verified_sorted)
            verified_sorted = [
                v for v in verified_sorted
                if v["stg_code"] not in preg_codes
            ]
            filtered = before - len(verified_sorted)
            if filtered:
                logger.info(
                    f"Pregnancy filter: removed {filtered} pregnancy-required "
                    f"conditions from verified_sorted (patient not pregnant)"
                )

        # Fast-path check (expanded threshold: 1.5x instead of 2.5x)
        top_score = verified_sorted[0].get("score", 0) if verified_sorted else 0
        second_score = verified_sorted[1].get("score", 0) if len(verified_sorted) > 1 else 0
        safety_flags = tool_results.get("check_safety_flags", {})
        has_flags = (
            len(safety_flags.get("red_flags_triggered", [])) > 0
            or len(safety_flags.get("vitals_flags", [])) > 0
        )
        use_fast_path = (
            top_score > 0
            and (second_score == 0 or top_score / max(second_score, 0.001) > 1.5)
            and not has_flags
            and acuity_info.get("acuity") == "routine"
        )

        # Launch BOTH LLM calls NOW (before deterministic work)
        t_llm_start = time.monotonic()
        safety_review_task = asyncio.create_task(
            self._run_safety_review(
                complaint, patient, symptoms, conditions, acuity_info, vitals, details
            )
        )

        ranked_codes: list[str] = []
        assessment_task = None
        if not use_fast_path:
            assessment_task = asyncio.create_task(
                self._generate_assessment_questions(
                    complaint, patient, verified_sorted,
                    features_by_condition, condition_id_map,
                )
            )

        # Yield to event loop so tasks can start their HTTP requests
        await asyncio.sleep(0)

        # Do deterministic work WHILE LLM calls run in background
        reported_symptoms = set(symptoms)
        condition_symptoms = self._build_condition_symptoms(
            verified_sorted, features_by_condition, condition_id_map, reported_symptoms
        )

        # Now await LLM results
        if use_fast_path:
            logger.info("Fast-path: dominant condition, no flags, skipping LLM")
            assessment_questions: list[dict] = []
            safety_review = await safety_review_task
        else:
            (ranked_codes, assessment_questions), safety_review = await asyncio.gather(
                assessment_task, safety_review_task
            )
        logger.info(f"[{time.monotonic()-t0:.1f}s] LLM calls took {time.monotonic()-t_llm_start:.1f}s")

        # Inject deterministic clinical questions based on demographics
        # Same pattern as pregnancy status / vitals — always fires for the
        # right demographic, never relies on the LLM to remember
        deterministic_questions = self._build_deterministic_questions(
            patient, complaint, symptoms
        )

        # Inject STG-grounded rule-based questions (zero LLM, from reasoning rules)
        rule_questions = _select_rule_questions(
            differential=verified_sorted[:5],
            known_symptoms=set(symptoms),
            known_vitals=vitals or {},
            known_labs={lab.get("test_name", ""): lab.get("value") for lab in (lab_results or [])},
            patient_age=patient.get("age") if patient else None,
            patient_sex=patient_sex,
            current_round=1,
            max_questions=5,
        )

        # Assemble: deterministic first, rule-based second, LLM last
        # Deduplicate by normalised question text
        seen_q = set()
        final_questions = []
        for q in deterministic_questions + rule_questions + assessment_questions:
            q_text = (q.get("question") or "").lower().strip()
            if q_text and q_text not in seen_q:
                seen_q.add(q_text)
                final_questions.append(q)
        assessment_questions = final_questions[:8]  # Cap at 8 (5 + some overflow)

        # Build full response deterministically
        # Build structured labs dict for rule matching
        labs_dict = {}
        if lab_results:
            for lr in lab_results:
                if isinstance(lr, dict):
                    tn = lr.get("test_name", "")
                    rv = lr.get("result", "")
                else:
                    tn = getattr(lr, "test_name", "")
                    rv = getattr(lr, "result", "")
                if tn:
                    labs_dict[tn] = rv

        result = self._build_full_response(
            tool_results, condition_symptoms, assessment_questions,
            vitals=vitals, symptoms=symptoms, labs=labs_dict or None,
        )

        # Apply LLM re-ranking if available
        if ranked_codes:
            result_conds = result.get("conditions", [])
            cond_by_code = {c["condition_code"]: c for c in result_conds}
            reordered = []
            for code in ranked_codes:
                if code in cond_by_code:
                    reordered.append(cond_by_code.pop(code))
            # Append any conditions not in ranked_codes (preserves original order)
            for c in result_conds:
                if c["condition_code"] in cond_by_code:
                    reordered.append(c)
            result["conditions"] = reordered
            logger.info(f"LLM re-ranked conditions: {ranked_codes[:5]}")

        # Enforce lab-confirmed conditions at top (overrides LLM ranking)
        lab_codes = set()
        for v in verified_sorted:
            if any("lab-confirmed" in f for f in v.get("matched_features", [])):
                lab_codes.add(v["stg_code"])

        # Enforce vitals-injected conditions at top (overrides LLM ranking)
        vitals_codes = set()
        for v in verified_sorted:
            if any("vitals-based" in f for f in v.get("matched_features", [])):
                vitals_codes.add(v["stg_code"])

        if lab_codes or vitals_codes:
            result_conds = result.get("conditions", [])
            lab_conds = [c for c in result_conds if c.get("condition_code") in lab_codes]
            vitals_conds = [c for c in result_conds
                           if c.get("condition_code") in vitals_codes
                           and c.get("condition_code") not in lab_codes]
            other_conds = [c for c in result_conds
                          if c.get("condition_code") not in lab_codes
                          and c.get("condition_code") not in vitals_codes]
            result["conditions"] = lab_conds + vitals_conds + other_conds

        # Recalculate confidence based on final position (top=0.90, decreasing)
        if ranked_codes or vitals_codes or lab_codes:
            final_conds = result.get("conditions", [])
            conf_values = [0.90, 0.78, 0.65, 0.52, 0.40]
            for i, c in enumerate(final_conds):
                c["confidence"] = conf_values[i] if i < len(conf_values) else 0.30

        logger.info(f"[{time.monotonic()-t0:.1f}s] Synthesis + safety done")

        # Merge safety review into result
        if safety_review and not safety_review.get("safe", True):
            result["safety_review"] = {
                "concerns": safety_review.get("concerns", []),
                "missing_conditions": safety_review.get("missing_conditions", []),
                "reviewed": True,
            }
            corrected = safety_review.get("corrected_acuity")
            if corrected:
                acuity_rank = {"routine": 0, "priority": 1, "urgent": 2}
                if acuity_rank.get(corrected, 0) > acuity_rank.get(result.get("acuity", "routine"), 0):
                    safety_flags = tool_results.get("check_safety_flags", {})
                    has_deterministic_flags = (
                        len(safety_flags.get("red_flags_triggered", [])) > 0
                        or len(safety_flags.get("vitals_flags", [])) > 0
                    )
                    if corrected == "urgent" and not has_deterministic_flags:
                        corrected = "priority"
                        logger.info("Safety reviewer wanted 'urgent' but no deterministic "
                                    "red flags found — capped at 'priority'")

                    if acuity_rank.get(corrected, 0) > acuity_rank.get(result.get("acuity", "routine"), 0):
                        result["acuity"] = corrected
                        for concern in safety_review.get("concerns", [])[:3]:
                            result.setdefault("acuity_reasons", []).append(concern)
                        result.setdefault("acuity_sources", []).append("Safety reviewer")

        # ── Add structured STG guidelines only for conditions in result ────
        final_conditions = result.get("conditions", [])
        final_codes = {c.get("condition_code") for c in final_conditions}
        final_names = {c.get("condition_name", c.get("name", "")).lower() for c in final_conditions}

        # Backfill: fetch details for any final conditions not already in details
        detail_codes = {d.get("stg_code") for d in details.values() if not d.get("error")}
        detail_names = {d.get("name", "").lower() for d in details.values() if not d.get("error")}
        for fc in final_conditions:
            fc_code = fc.get("condition_code", "")
            fc_name = (fc.get("condition_name") or fc.get("name", "")).lower()
            if fc_code in detail_codes or fc_name in detail_names:
                continue
            # Missing — look up and fetch
            try:
                async with self.pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT id FROM conditions WHERE stg_code = $1 LIMIT 1", fc_code
                    )
                if row:
                    d = await TOOL_HANDLERS["get_condition_detail"](
                        {"condition_id": row["id"]}, self.pool
                    )
                    details[row["id"]] = d
                    detail_codes.add(d.get("stg_code"))
                    detail_names.add(d.get("name", "").lower())
            except Exception as e:
                logger.warning(f"Failed to backfill detail for {fc_code}: {e}")

        stg_guidelines = {}
        for cid, detail in details.items():
            if detail.get("error"):
                continue
            name = detail.get("name", "")
            stg_code = detail.get("stg_code", "")

            # Only include STG data for conditions that made it to the final result
            # Match on exact code OR condition name (handles code variants like 4.7 vs 4.7.1)
            if stg_code not in final_codes and name.lower() not in final_names:
                continue

            # Parse medicines into clean list + check for cautions
            meds = detail.get("medicines", [])
            medicine_list = []
            medication_warnings = []
            patient_meds_lower = [
                m.strip().lower()
                for m in (core_history or {}).get("medications", "").split(",")
                if m.strip() and m.strip().lower() not in ("none", "nil", "n/a", "")
            ]
            patient_age = (patient or {}).get("age")
            is_paediatric = patient_age is not None and patient_age < 18
            is_pregnant = (patient or {}).get("pregnancy_status") == "yes"
            for m in (meds if isinstance(meds, list) else []):
                med_entry = {
                    "name": m.get("name", ""),
                    "treatment_line": m.get("treatment_line", ""),
                    "dose": m.get("dose_context", ""),
                    "special_notes": m.get("special_notes", ""),
                }
                # Add paediatric dosing for child patients
                if is_paediatric and m.get("paediatric_dose_mg_per_kg"):
                    med_entry["paediatric_dose_mg_per_kg"] = m["paediatric_dose_mg_per_kg"]
                    med_entry["paediatric_frequency"] = m.get("paediatric_frequency", "")
                    if m.get("paediatric_note"):
                        med_entry["paediatric_note"] = m["paediatric_note"]
                # Add pregnancy safety for pregnant patients
                if is_pregnant and m.get("pregnancy_safe") is not None:
                    med_entry["pregnancy_safe"] = m["pregnancy_safe"]
                    if m.get("pregnancy_notes"):
                        med_entry["pregnancy_notes"] = m["pregnancy_notes"]
                if med_entry["name"]:
                    medicine_list.append(med_entry)
                    # Flag if patient is already on this medicine
                    med_lower = med_entry["name"].lower()
                    for pm in patient_meds_lower:
                        if pm in med_lower or med_lower in pm:
                            medication_warnings.append(
                                f"Patient reports taking {pm} — already prescribed for this condition"
                            )
                    # Surface special notes with caution keywords
                    notes = med_entry.get("special_notes", "") or ""
                    if any(kw in notes.lower() for kw in ("avoid", "contraindic", "caution", "do not", "monitor")):
                        medication_warnings.append(
                            f"{med_entry['name']}: {notes[:150]}"
                        )

            # Parse referral criteria
            referral = detail.get("referral_criteria", [])
            if isinstance(referral, str):
                try:
                    referral = json.loads(referral)
                except (json.JSONDecodeError, TypeError):
                    referral = [referral] if referral else []

            # NOTE: detail keys come from handle_get_condition_detail (tools.py)
            # which remaps: description_text→description, medicines_json→medicines
            desc = _format_stg_text(detail.get("description", ""))
            gm = _format_stg_text(detail.get("general_measures", ""))
            danger_signs_list = _split_to_bullet_list(
                detail.get("danger_signs", "")
            )

            guideline_entry = {
                "stg_code": stg_code,
                "description": desc["summary"],
                "description_full": desc["full"],
                "general_measures": gm["summary"],
                "general_measures_full": gm["full"],
                "danger_signs": danger_signs_list,
                "medicines": medicine_list,
                "referral_criteria": referral,
                "source_pages": detail.get("source_pages", []),
            }
            if medication_warnings:
                guideline_entry["medication_warnings"] = medication_warnings

            # Fetch rich content (tables, algorithms) from knowledge_chunks
            cond_id = detail.get("condition_id")
            if cond_id:
                async with self.pool.acquire() as conn:
                    rich = await get_condition_rich_content(conn, cond_id)
                if rich:
                    # Clean cross-references from rich content too
                    for item in rich:
                        item["content"] = _clean_stg_references(item["content"])
                    guideline_entry["clinical_tables"] = rich

            stg_guidelines[name] = guideline_entry
        result["stg_guidelines"] = stg_guidelines

        return result

    # ── Refine endpoint ──────────────────────────────────────────────────────

    async def refine(
        self,
        complaint: str,
        conditions: list[dict],
        answers: dict,
        all_time_answers: dict,
        current_round: int,
        patient: Optional[dict] = None,
        request_next_round: bool = False,
        stg_feature_data: Optional[dict] = None,
    ) -> dict:
        """
        Re-score conditions based on assessment answers.
        More deterministic than analyze — uses Haiku once for synthesis.
        """
        # Partition answers into confirmed/denied symptoms
        confirmed = []
        denied = []
        for q_id, answer in all_time_answers.items():
            ans = str(answer).lower().strip()
            if ans in ("yes", "true", "1"):
                confirmed.append(q_id)
            elif ans in ("no", "false", "0"):
                denied.append(q_id)

        # Fetch condition details for top conditions to inform re-scoring
        condition_details = []
        for c in conditions[:5]:
            code = c.get("condition_code", "")
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT id, stg_code, name, danger_signs FROM conditions WHERE stg_code = $1",
                    code,
                )
            if row:
                condition_details.append(dict(row))

        # Check safety on confirmed symptoms
        red_flag_alert = None
        if confirmed:
            condition_ids = [cd["id"] for cd in condition_details]
            if condition_ids:
                safety = await TOOL_HANDLERS["check_safety_flags"](
                    {"symptoms": confirmed, "condition_ids": condition_ids, "vitals": {}},
                    self.pool,
                )
                if safety.get("requires_escalation"):
                    flags = safety.get("red_flags_triggered", [])
                    flag_names = [f["flag"] for f in flags]
                    red_flag_alert = f"Red flags detected: {', '.join(flag_names)}" if flag_names else None

        # Build verified conditions lookup for citation scrubbing
        verified = {}
        for cd in condition_details:
            code = cd.get("stg_code", "")
            if code:
                verified[code] = {
                    "stg_code": code,
                    "name": cd.get("name", ""),
                }
        verified_text = "\n".join(
            f"- stg_code: {v['stg_code']}, name: {v['name']}"
            for v in verified.values()
        ) or "No conditions verified."

        # Use Haiku to synthesise refined conditions + next questions
        prompt = REFINE_SYNTHESIS_PROMPT.format(
            complaint=complaint,
            conditions=json.dumps(conditions, default=str),
            all_time_answers=json.dumps(all_time_answers),
            current_round=current_round,
            confirmed=json.dumps(confirmed),
            denied=json.dumps(denied),
            condition_details=json.dumps(condition_details, default=str),
            next_round=current_round + 1,
            verified_conditions=verified_text,
        )

        try:
            response = await self._call_with_fallback(
                max_tokens=4096,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(text)
        except (json.JSONDecodeError, anthropic.APIError, IndexError) as e:
            logger.error(f"Refine synthesis failed: {e}")
            # Fallback: return conditions unchanged
            result = {
                "refinement_source": "rules",
                "conditions": conditions,
                "next_round_questions": None,
            }

        # Post-process: scrub hallucinated references (lenient — preserve frontend condition names)
        result = self._scrub_references(result, verified, strict=False)

        # Override red flag from our deterministic check
        if red_flag_alert:
            result["red_flag_alert"] = red_flag_alert

        # Augment next-round questions with rule-based questions
        if request_next_round and current_round < 5:
            # Build known symptoms from confirmed answers
            all_known = set(confirmed)
            rule_next = _select_rule_questions(
                differential=[
                    {"stg_code": c.get("condition_code", ""), "name": c.get("condition_name", "")}
                    for c in conditions[:5]
                ],
                known_symptoms=all_known,
                known_vitals={},
                known_labs={},
                patient_age=patient.get("age") if patient else None,
                patient_sex=(patient.get("sex") or "").lower() if patient else None,
                current_round=current_round + 1,
                max_questions=3,
            )
            if rule_next:
                llm_next = result.get("next_round_questions") or []
                # Deduplicate
                seen_q = {(q.get("question") or "").lower().strip() for q in llm_next}
                for rq in rule_next:
                    q_text = (rq.get("question") or "").lower().strip()
                    if q_text not in seen_q:
                        seen_q.add(q_text)
                        llm_next.append(rq)
                result["next_round_questions"] = llm_next[:5]

        # Don't generate more questions past round 5
        if current_round >= 5:
            result["next_round_questions"] = None

        if not request_next_round:
            result["next_round_questions"] = None

        # Propagate match quality from original analyze if available
        if stg_feature_data and "match_quality" in stg_feature_data:
            result["match_quality"] = stg_feature_data["match_quality"]
            if "low_confidence_warning" in stg_feature_data:
                result["low_confidence_warning"] = stg_feature_data["low_confidence_warning"]

        return result

    # ── Private helpers ──────────────────────────────────────────────────────

    async def _run_safety_review(self, complaint, patient, symptoms, conditions, acuity_info, vitals=None, condition_details=None):
        """Run safety review concurrently with synthesis using search results."""
        computed_acuity = acuity_info.get("acuity", "routine")

        # Collect danger signs from top 3 conditions
        danger_signs_text = ""
        if condition_details:
            parts = []
            for _cid, d in list(condition_details.items())[:3]:
                ds = d.get("danger_signs", "")
                if ds:
                    parts.append(f"{d.get('name','')}: {ds[:200]}")
            if parts:
                danger_signs_text = "\n".join(parts)

        prompt = (
            f"Complaint: {complaint}\n"
            f"Symptoms: {', '.join(symptoms[:8])}\n"
            f"Vitals: {json.dumps(vitals or {})}\n"
            f"Computed acuity: {computed_acuity}\n"
            f"Top conditions: {', '.join(c.get('name','') for c in conditions[:3])}\n"
        )
        if danger_signs_text:
            prompt += f"Danger signs:\n{danger_signs_text}\n"
        prompt += (
            "\nCheck for missed red flags matching the STATED symptoms. "
            "Do NOT infer unreported symptoms. Do NOT escalate on speculation. "
            "Return {\"safe\":true} if acuity is correct, or "
            "{\"safe\":false,\"concerns\":[\"...\"],\"corrected_acuity\":\"priority|urgent\"} if not."
        )

        try:
            response = await self._call_with_fallback(
                max_tokens=256,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except Exception as e:
            logger.warning(f"Safety review error: {e}")
            return {"safe": True}

    @staticmethod
    def _compute_vitals_acuity(
        vitals: dict,
        complaint: str = "",
        symptoms: list[str] | None = None,
        patient_age: int | None = None,
    ) -> dict:
        """Compute acuity using the South African Triage Scale (SATS).

        Uses TEWS (Triage Early Warning Score) vital sign scoring plus
        clinical discriminators from the SATS training manual.

        Returns dict with: acuity, sats_colour, sats_priority, tews_score,
        reasons, target_minutes — backward compatible with old 3-tier system.
        """
        return compute_sats_acuity(
            vitals=vitals,
            complaint=complaint,
            symptoms=symptoms,
            patient_age=patient_age,
        )

    async def _inject_vitals_conditions(self, conditions: list, vitals: dict, patient_age: int | None = None) -> list:
        """
        Inject/boost conditions based on vital signs using the vitals_condition_mapping table.
        Fully data-driven — no hardcoded condition logic.

        Two marker modes from the table:
        - force_rank_one=TRUE: "vitals-based" marker → forced to #1 in synthesis
        - force_rank_one=FALSE: "noted" marker → included but LLM ranks it naturally
        """
        if not vitals:
            return conditions

        from db.database import get_vitals_mappings

        async with self.pool.acquire() as conn:
            mappings = await get_vitals_mappings(conn, vitals, patient_age=patient_age)

        if not mappings:
            return conditions

        existing_ids = {c["id"] for c in conditions}
        injected = False

        for mapping in mappings:
            cid = mapping["condition_id"]
            score = mapping["score_boost"]
            stg_code = mapping["stg_code"]
            condition_name = mapping["condition_name"]
            severity = mapping["severity_label"] or ""
            force = mapping["force_rank_one"]
            stg_ref = mapping["stg_reference"] or f"STG {stg_code}"
            vital_name = mapping["vital_name"]
            vital_value = vitals.get(vital_name)

            # Build feature description
            marker = "vitals-based" if force else "noted"
            feat = f"{vital_name}={vital_value} → {severity} ({marker}, {stg_ref})"

            if cid in existing_ids:
                for c in conditions:
                    if c["id"] == cid:
                        if score > c.get("adjusted_score", 0):
                            c["adjusted_score"] = score
                            c["raw_score"] = max(c.get("raw_score", 0), score)
                        if feat not in c.get("matched_features", []):
                            c.setdefault("matched_features", []).append(feat)
                        break
            else:
                conditions.append({
                    "id": cid,
                    "stg_code": stg_code,
                    "name": condition_name,
                    "chapter_name": mapping.get("chapter_name", ""),
                    "extraction_confidence": float(mapping.get("extraction_confidence", 1.0) or 1.0),
                    "duration_profile": mapping.get("duration_profile"),
                    "match_count": 1,
                    "raw_score": score,
                    "matched_features": [feat],
                    "symptom_groups_matched": 1,
                    "adjusted_score": score,
                })
                existing_ids.add(cid)
            injected = True
            logger.info(f"Vitals injection: {condition_name} ({vital_name}={vital_value}, score={score}, force={force})")

        if injected:
            conditions.sort(key=lambda c: (c.get("adjusted_score", 0), c.get("raw_score", 0)), reverse=True)

        return conditions

    # ── Lab-Result-to-Condition Injection ──────────────────────────
    # Deterministic mapping of confirmed lab results to STG conditions.
    # Two input paths: (1) regex on complaint text, (2) structured API input.
    # Follows the same pattern as _inject_vitals_conditions.

    @staticmethod
    def _extract_lab_results(
        complaint: str,
        pregnancy_status: str = "unknown",
        structured_labs: list[dict] | None = None,
        patient_age: int | None = None,
    ) -> list[dict]:
        """
        Deterministic lab result extraction from complaint text and/or structured input.

        Path 1 — Text parsing: regex scan on complaint for known lab result patterns.
        Path 2 — Structured input: match lab_results[].test_name against config.

        Returns list of matched lab results, each with:
        - id, marker_label, target_code, force_rank_one, score_boost, add_symptoms
        """
        import re

        lab_result_patterns = _cache.lab_result_patterns if _cache else []
        complaint_lower = complaint.lower()
        is_pregnant = pregnancy_status.lower().strip() in ("pregnant", "yes")
        is_child = patient_age is not None and patient_age < 18
        matched_ids: set[str] = set()
        matches: list[dict] = []

        def _resolve_code(lab: dict) -> str | None:
            codes = lab["condition_codes"]
            if is_pregnant and "pregnant" in codes:
                return codes["pregnant"]
            if is_child and "child" in codes:
                return codes["child"]
            return codes.get("default")

        def _build_match(lab: dict) -> dict:
            return {
                "id": lab["id"],
                "marker_label": lab["marker_label"],
                "target_code": _resolve_code(lab),
                "force_rank_one": lab["force_rank_one"],
                "score_boost": lab["score_boost"],
                "add_symptoms": lab.get("add_symptoms", []),
            }

        # Path 1: Regex scan on complaint text
        for lab in lab_result_patterns:
            if lab["id"] in matched_ids:
                continue
            threshold = lab.get("numeric_threshold")
            direction = lab.get("threshold_direction", "below")
            for pattern in lab["patterns"]:
                m = re.search(pattern, complaint_lower)
                if m:
                    # Numeric threshold: extract captured value and compare
                    if threshold is not None:
                        try:
                            value = float(m.group(1))
                            triggered = (
                                value < threshold if direction == "below"
                                else value > threshold
                            )
                            if not triggered:
                                continue
                            # Annotate marker with actual value
                            match = _build_match(lab)
                            match["marker_label"] = f"{lab['marker_label']}: {value:.0f}"
                            matches.append(match)
                        except (IndexError, ValueError):
                            continue
                    else:
                        matches.append(_build_match(lab))
                    matched_ids.add(lab["id"])
                    break

        # Path 2: Structured lab input (EMR path)
        if structured_labs:
            for sl in structured_labs:
                test_name = sl.get("test_name", "").lower().strip()
                result_val = sl.get("result", "").lower().strip()
                for lab in lab_result_patterns:
                    if lab["id"] in matched_ids:
                        continue
                    name_match = any(
                        sn in test_name or test_name in sn
                        for sn in lab.get("structured_names", [])
                    )
                    if not name_match:
                        continue
                    threshold = lab.get("numeric_threshold")
                    direction = lab.get("threshold_direction", "below")
                    if threshold is not None:
                        # Numeric lab: parse value and compare
                        try:
                            value = float(result_val)
                            triggered = (
                                value < threshold if direction == "below"
                                else value > threshold
                            )
                            if triggered:
                                match = _build_match(lab)
                                match["marker_label"] = f"{lab['marker_label']}: {value:.0f}"
                                matches.append(match)
                                matched_ids.add(lab["id"])
                        except ValueError:
                            pass
                    else:
                        # Binary lab: check positive keywords
                        result_match = any(
                            kw in result_val
                            for kw in lab.get("positive_keywords", [])
                        )
                        if result_match:
                            matches.append(_build_match(lab))
                            matched_ids.add(lab["id"])
                    break

        return matches

    async def _inject_lab_conditions(
        self, conditions: list, lab_results: list[dict],
        patient_age: int | None = None,
    ) -> list:
        """
        Inject/boost conditions based on confirmed lab results.
        Mirrors _inject_vitals_conditions() pattern exactly.

        - force_rank_one=True: "lab-confirmed" marker → forced to #1 in synthesis
        - force_rank_one=False: "noted" marker → included but LLM ranks naturally
        """
        if not lab_results:
            return conditions

        existing_ids = {c["id"] for c in conditions}
        injected = False

        for lab in lab_results:
            target_code = lab.get("target_code")
            if not target_code:
                continue

            async with self.pool.acquire() as conn:
                if patient_age is not None:
                    row = await conn.fetchrow(
                        "SELECT id, stg_code, name, chapter_name, extraction_confidence, "
                        "referral_required, care_setting, source_tag, duration_profile "
                        "FROM conditions WHERE stg_code = $1 "
                        "AND min_age_years <= $2 AND max_age_years >= $2",
                        target_code, patient_age,
                    )
                else:
                    row = await conn.fetchrow(
                        "SELECT id, stg_code, name, chapter_name, extraction_confidence, "
                        "referral_required, care_setting, source_tag, duration_profile "
                        "FROM conditions WHERE stg_code = $1",
                        target_code,
                    )

            if not row:
                logger.warning(f"Lab injection: condition {target_code} not found in DB")
                continue

            cid = row["id"]
            score = lab["score_boost"]
            force = lab["force_rank_one"]
            marker = "lab-confirmed" if force else "noted"
            feat = f"{lab['marker_label']} → {row['name']} ({marker})"

            if cid in existing_ids:
                for c in conditions:
                    if c["id"] == cid:
                        if score > c.get("adjusted_score", 0):
                            c["adjusted_score"] = score
                            c["raw_score"] = max(c.get("raw_score", 0), score)
                        if feat not in c.get("matched_features", []):
                            c.setdefault("matched_features", []).append(feat)
                        break
            else:
                conditions.append({
                    "id": cid,
                    "stg_code": row["stg_code"],
                    "name": row["name"],
                    "chapter_name": row["chapter_name"] or "",
                    "extraction_confidence": float(row["extraction_confidence"] or 1.0),
                    "duration_profile": row.get("duration_profile"),
                    "match_count": 1,
                    "raw_score": score,
                    "matched_features": [feat],
                    "symptom_groups_matched": 1,
                    "adjusted_score": score,
                    "referral_required": row.get("referral_required"),
                    "care_setting": row.get("care_setting"),
                    "source_tag": row.get("source_tag"),
                })
                existing_ids.add(cid)

            injected = True
            logger.info(
                f"Lab injection: {row['name']} ({target_code}, "
                f"score={score}, force={force})"
            )

        if injected:
            conditions.sort(
                key=lambda c: (c.get("adjusted_score", 0), c.get("raw_score", 0)),
                reverse=True,
            )

        return conditions

    # ── Duration-Aware Scoring ─────────────────────────────────────
    # Applies multipliers to adjusted_score based on symptom duration.
    # Penalises self-limiting conditions (common cold) for long durations,
    # boosts chronic/infectious conditions (TB, COPD) when duration matches.

    @staticmethod
    def _apply_duration_modifiers(conditions: list, core_history: dict) -> list:
        """Apply duration-based score modifiers to matched conditions.

        Parses core_history.onset → duration category (acute/subacute/chronic),
        then multiplies adjusted_score using the condition's duration_profile
        from the DB (via DURATION_PROFILE_MULTIPLIERS).
        """
        if not core_history:
            return conditions

        onset = (core_history.get("onset") or "").strip().lower()
        if not onset:
            return conditions

        from agents.scoring_config import DURATION_CATEGORIES, DURATION_PROFILE_MULTIPLIERS

        category = DURATION_CATEGORIES.get(onset)
        if not category:
            return conditions

        modified = False
        for c in conditions:
            profile = c.get("duration_profile")
            if not profile or profile not in DURATION_PROFILE_MULTIPLIERS:
                continue
            multiplier = DURATION_PROFILE_MULTIPLIERS[profile].get(category, 1.0)
            if multiplier != 1.0:
                old_score = c.get("adjusted_score", 0)
                c["adjusted_score"] = old_score * multiplier

                # Add feature marker explaining the adjustment
                if multiplier > 1.0:
                    label = f"{onset} duration → {c.get('name', '')} more likely ({multiplier}x boost)"
                else:
                    label = f"{onset} duration → {c.get('name', '')} less likely ({multiplier}x penalty)"
                c.setdefault("matched_features", []).append(label)
                modified = True
                logger.info(f"Duration modifier: {c.get('name', '')} {old_score:.3f} → {c['adjusted_score']:.3f} ({profile}/{category}, {multiplier}x)")

        if modified:
            conditions.sort(
                key=lambda c: (c.get("adjusted_score", 0), c.get("raw_score", 0)),
                reverse=True,
            )

        return conditions

    def _build_analyze_prompt(self, complaint, patient, vitals, core_history):
        parts = [f"Chief complaint: {complaint}"]
        if patient:
            age = patient.get("age", "unknown")
            sex = patient.get("sex", "unknown")
            preg = patient.get("pregnancy_status", "unknown")
            parts.append(f"Patient: age {age}, sex {sex}, pregnancy status: {preg}")
        if vitals:
            vitals_str = ", ".join(f"{k}={v}" for k, v in vitals.items() if v is not None)
            parts.append(f"Vitals: {vitals_str}")
        if core_history:
            onset = core_history.get("onset", "unknown")
            recurrence = core_history.get("recurrence", "unknown")
            meds = core_history.get("medications", "none")
            parts.append(f"History: onset={onset}, recurrence={recurrence}, current medications={meds}")
        parts.append("\nRun the full triage analysis using all available tools in sequence.")
        return "\n".join(parts)

    def _extract_verified_conditions(self, tool_results: dict) -> dict:
        """
        Build a lookup of verified conditions from tool results.
        Returns {stg_code: {name, chapter, source_pages, stg_code}} from DB data only.
        """
        verified = {}

        # From search_conditions — include score and matched features for ranking
        for c in tool_results.get("search_conditions", {}).get("conditions", []):
            code = c.get("stg_code", "")
            if code:
                verified[code] = {
                    "stg_code": code,
                    "name": c.get("name", ""),
                    "chapter": c.get("chapter_name", ""),
                    "score": c.get("adjusted_score", c.get("raw_score", 0)),
                    "matched_features": c.get("matched_features", []),
                    "groups_matched": c.get("symptom_groups_matched", 0),
                }

        # From score_differential
        for c in tool_results.get("score_differential", {}).get("scored_conditions", []):
            code = c.get("stg_code", "")
            if code and code not in verified:
                verified[code] = {
                    "stg_code": code,
                    "name": c.get("name", ""),
                    "chapter": c.get("chapter", ""),
                }

        # From get_condition_detail (most authoritative — has source_pages)
        detail = tool_results.get("get_condition_detail", {})
        if isinstance(detail, dict) and not detail.get("error"):
            code = detail.get("stg_code", "")
            if code:
                verified[code] = {
                    "stg_code": code,
                    "name": detail.get("name", ""),
                    "chapter": detail.get("chapter", ""),
                    "source_pages": detail.get("source_pages", []),
                }

        return verified

    def _scrub_references(self, result: dict, verified: dict, strict: bool = True) -> dict:
        """
        Post-process synthesis output to remove hallucinated references.
        Only keeps references traceable to a verified stg_code.

        strict=True  (analyze path): drop unverified conditions, fix names to DB values
        strict=False (refine path):  keep all conditions, only scrub citation strings
        """
        valid_codes = set(verified.keys())

        # Scrub conditions
        scrubbed_conditions = []
        for c in result.get("conditions", []):
            code = c.get("condition_code", "")
            name = c.get("condition_name", "")

            if strict:
                # In strict mode, condition must exist in DB
                if code not in valid_codes:
                    # Try to find it by name — use partial matching to handle
                    # LLM shortening names (e.g. "Malaria" vs "Malaria, Non-Severe/Uncomplicated")
                    matched_code = None
                    name_lower = name.lower()
                    # 1. Exact match
                    for vc, vdata in verified.items():
                        if vdata["name"].lower() == name_lower:
                            matched_code = vc
                            break
                    # 2. Contains match (LLM name is substring of DB name, or vice versa)
                    if not matched_code:
                        for vc, vdata in verified.items():
                            db_name = vdata["name"].lower()
                            if name_lower in db_name or db_name in name_lower:
                                matched_code = vc
                                break
                    # 3. Significant word overlap (at least 2 words match)
                    if not matched_code:
                        name_words = set(w for w in name_lower.split() if len(w) > 3)
                        best_overlap = 0
                        for vc, vdata in verified.items():
                            db_words = set(w for w in vdata["name"].lower().split() if len(w) > 3)
                            overlap = len(name_words & db_words)
                            if overlap > best_overlap and overlap >= 2:
                                best_overlap = overlap
                                matched_code = vc
                    if matched_code:
                        c["condition_code"] = matched_code
                        code = matched_code
                    else:
                        logger.warning(f"Dropping unverified condition: {code} / {name}")
                        continue

                # Fix name to match DB exactly
                if code in verified:
                    c["condition_name"] = verified[code]["name"]

            # Scrub source_references — only allow "STG <valid_code>"
            raw_refs = c.get("source_references", [])
            clean_refs = []
            for ref in raw_refs:
                for vc in valid_codes:
                    if vc in ref:
                        clean_refs.append(f"STG {vc}")
                        break
            # Always include the condition's own STG code if it's verified
            if code in valid_codes:
                own_ref = f"STG {code}"
                if own_ref not in clean_refs:
                    clean_refs.insert(0, own_ref)
            c["source_references"] = clean_refs

            scrubbed_conditions.append(c)

        result["conditions"] = scrubbed_conditions

        # Scrub acuity_sources
        raw_sources = result.get("acuity_sources", [])
        clean_sources = []
        for src in raw_sources:
            if "vital signs" in src.lower() or "standard:" in src.lower():
                clean_sources.append("Standard: vital signs assessment")
            else:
                for vc in valid_codes:
                    if vc in src:
                        clean_sources.append(f"STG {vc}")
                        break
        result["acuity_sources"] = list(dict.fromkeys(clean_sources))  # dedupe, preserve order

        # Scrub assessment_questions source_citation + grounding
        for q in result.get("assessment_questions", []):
            citation = q.get("source_citation", "")
            verified_citation = ""
            for vc in valid_codes:
                if vc in citation:
                    verified_citation = f"STG {vc}"
                    break
            q["source_citation"] = verified_citation
            q["grounding"] = "verified" if verified_citation else "unverified"

        # Scrub condition_symptoms keys — must match verified names (partial matching)
        raw_cs = result.get("condition_symptoms", {})
        clean_cs = {}
        for key, questions in raw_cs.items():
            matched_name = None
            key_lower = key.lower()
            for v in verified.values():
                db_name = v["name"].lower()
                if db_name == key_lower or key_lower in db_name or db_name in key_lower:
                    matched_name = v["name"]
                    break
            if matched_name:
                clean_cs[matched_name] = questions
        result["condition_symptoms"] = clean_cs

        return result

    async def _synthesize_analyze(self, complaint, patient, vitals, core_history, tool_results):
        """Use Haiku to produce the final AnalyzeResponse-shaped JSON."""
        # Build verified conditions lookup from DB-sourced tool results
        verified = self._extract_verified_conditions(tool_results)
        # Include search scores + matched STG features so LLM ranks from evidence
        verified_sorted = sorted(
            verified.values(),
            key=lambda v: v.get("score", 0),
            reverse=True,
        )
        # Promote vitals-injected conditions to the top.
        # Vitals are direct measurement — more definitive than symptom-graph
        # matching (e.g., BP 170/100 IS hypertension, regardless of what
        # "headache + dizziness" matches in the graph).
        for i, v in enumerate(verified_sorted):
            features = v.get("matched_features", [])
            if any("vitals-based" in f for f in features):
                if i > 0:
                    verified_sorted.insert(0, verified_sorted.pop(i))
                break  # only one vitals-injected condition expected
        # Promote lab-confirmed conditions above vitals
        for i, v in enumerate(verified_sorted):
            features = v.get("matched_features", [])
            if any("lab-confirmed" in f for f in features):
                if i > 0:
                    verified_sorted.insert(0, verified_sorted.pop(i))
                break
        details = tool_results.get("condition_details", {})
        verified_lines = []
        for rank, v in enumerate(verified_sorted, 1):
            features = v.get("matched_features", [])
            # Show clean feature names without "(STG text match)" suffixes
            clean_feats = []
            for f in features[:5]:
                # Strip source annotations like "(condition name match)", "(STG text in ...)"
                base = f.split(" (")[0] if " (" in f else f
                if base not in clean_feats:
                    clean_feats.append(base)
            feat_str = ", ".join(clean_feats) if clean_feats else "name/text match"
            line = f"#{rank}. {v['stg_code']}: {v['name']} (matched: {feat_str})"

            # For top 3 conditions, include danger signs and description so the LLM
            # can generate clinically specific verification questions
            if rank <= 3:
                for cid, d in details.items():
                    if d.get("stg_code") == v["stg_code"]:
                        danger = d.get("danger_signs", "")
                        if danger:
                            line += f"\n   DANGER SIGNS: {danger[:200]}"
                        desc = d.get("description", "")
                        if desc:
                            line += f"\n   KEY FEATURES: {desc[:250]}"
                        break

            verified_lines.append(line)
        verified_text = "\n".join(verified_lines) or "No conditions found."

        # Build compact safety summary
        safety = tool_results.get("check_safety_flags", {})
        vitals_acuity = tool_results.get("vitals_acuity", {})
        safety_data = json.dumps({
            "red_flags": safety.get("red_flags_triggered", []),
            "vitals_acuity": vitals_acuity.get("acuity", "routine"),
            "vitals_reasons": vitals_acuity.get("reasons", []),
        }, default=str)

        prompt = SYNTHESIS_PROMPT.format(
            complaint=complaint,
            patient=json.dumps(patient) if patient else "none",
            vitals=json.dumps(vitals) if vitals else "none",
            verified_conditions=verified_text,
            safety_data=safety_data,
        )

        try:
            response = await self._call_with_fallback(
                max_tokens=4096,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            result = self._parse_json_response(text)
        except json.JSONDecodeError as e:
            logger.error(f"Synthesis JSON parse failed: {e}")
            logger.error(f"Raw text (first 500 chars): {text[:500]}")
            result = self._build_fallback_response(tool_results)
        except anthropic.APIError as e:
            logger.error(f"Synthesis API error: {e}")
            result = self._build_fallback_response(tool_results)

        # Post-process: scrub any references Sonnet hallucinated despite instructions
        result = self._scrub_references(result, verified)

        # Enforce lab-confirmed and vitals-injected conditions at top of results.
        lab_codes = set()
        for v in verified_sorted:
            if any("lab-confirmed" in f for f in v.get("matched_features", [])):
                lab_codes.add(v["stg_code"])

        vitals_codes = set()
        for v in verified_sorted:
            if any("vitals-based" in f for f in v.get("matched_features", [])):
                vitals_codes.add(v["stg_code"])

        if lab_codes or vitals_codes:
            conditions = result.get("conditions", [])
            lab_conds = [c for c in conditions if c.get("condition_code") in lab_codes]
            vitals_conds = [c for c in conditions
                           if c.get("condition_code") in vitals_codes
                           and c.get("condition_code") not in lab_codes]
            other_conds = [c for c in conditions
                          if c.get("condition_code") not in lab_codes
                          and c.get("condition_code") not in vitals_codes]

            # If vitals condition was dropped by the LLM, add it back from search data
            existing_vitals = {c.get("condition_code") for c in vitals_conds}
            for v in verified_sorted:
                code = v["stg_code"]
                if code in vitals_codes and code not in existing_vitals:
                    features = v.get("matched_features", [])
                    clean_feats = [f.split(" (")[0] if " (" in f else f for f in features[:3]]
                    vitals_conds.append({
                        "condition_code": code,
                        "condition_name": v["name"],
                        "confidence": 0.90,
                        "matched_symptoms": clean_feats,
                        "reasoning": "Vitals directly indicate this condition (STG criteria)",
                        "source_references": [f"STG {code}"],
                    })

            # If lab condition was dropped by the LLM, add it back
            existing_labs = {c.get("condition_code") for c in lab_conds}
            for v in verified_sorted:
                code = v["stg_code"]
                if code in lab_codes and code not in existing_labs:
                    features = v.get("matched_features", [])
                    clean_feats = [f.split(" (")[0] if " (" in f else f for f in features[:3]]
                    lab_conds.append({
                        "condition_code": code,
                        "condition_name": v["name"],
                        "confidence": 0.95,
                        "matched_symptoms": clean_feats,
                        "reasoning": "Lab-confirmed diagnosis — matched via deterministic lab result scan",
                        "source_references": [f"STG {code}"],
                        "lab_confirmed": True,
                    })

            result["conditions"] = lab_conds + vitals_conds + other_conds

        return result

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        """
        Robustly extract JSON from an LLM response.
        Handles: raw JSON, markdown-fenced JSON, JSON with trailing text.
        """
        text = text.strip()

        # Strip markdown fences (various formats)
        if text.startswith("```"):
            # Handle ```json\n...\n``` and ```\n...\n```
            first_newline = text.find("\n")
            if first_newline > 0:
                text = text[first_newline + 1:]
            # Strip trailing fence
            last_fence = text.rfind("```")
            if last_fence > 0:
                text = text[:last_fence]
            text = text.strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object boundaries (string-aware)
        start = text.find("{")
        if start == -1:
            raise json.JSONDecodeError("No JSON object found", text, 0)

        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])

        # Last resort: truncated JSON — try closing open braces/brackets
        # This handles the common case where max_tokens was hit mid-output
        truncated = text[start:]
        # Close any open strings
        if in_string:
            truncated += '"'
        # Close remaining brackets/braces
        for _ in range(depth):
            # Check if we're inside an array by scanning backwards
            last_open = max(truncated.rfind("["), truncated.rfind("{"))
            if last_open >= 0 and truncated[last_open] == "[":
                truncated += "]"
            truncated += "}"
        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            pass

        raise json.JSONDecodeError("Unbalanced JSON braces", text, start)

    def _build_fallback_response(self, tool_results: dict) -> dict:
        """Build a response from raw tool results when synthesis fails.
        All references come directly from DB-sourced tool results — no hallucination possible."""
        symptoms = tool_results.get("extract_symptoms", {}).get("symptoms", [])
        safety = tool_results.get("check_safety_flags", {})

        # Use search_conditions results directly — these already have gender/age
        # filtering, parent heading replacement, dedup, and non-disease exclusion
        # applied in handle_search_conditions
        search_data = tool_results.get("search_conditions", {})
        search_conditions = search_data.get("conditions", [])

        conditions = []
        for i, s in enumerate(search_conditions[:5]):
            code = s.get("stg_code", "")
            score = s.get("adjusted_score", s.get("raw_score", 0))
            # Convert score to confidence: normalize to 0-1 range, highest gets ~0.90
            confidence = round(min(score / max(search_conditions[0].get("adjusted_score", 1), 0.01) * 0.90, 0.95), 2)
            conditions.append({
                "condition_code": code,
                "condition_name": s.get("name", ""),
                "confidence": confidence,
                "matched_symptoms": s.get("matched_features", []),
                "reasoning": "Matched via STG knowledge graph",
                "source_references": [f"STG {code}"] if code else [],
            })

        acuity = "routine"
        vitals_acuity = tool_results.get("vitals_acuity", {})
        if vitals_acuity.get("acuity"):
            acuity = vitals_acuity["acuity"]
        if safety.get("requires_escalation"):
            acuity = "urgent"

        result = {
            "extracted_symptoms": symptoms,
            "acuity": acuity,
            "acuity_reasons": vitals_acuity.get("reasons", []),
            "acuity_sources": ["SATS (South African Triage Scale)"],
            "conditions": conditions,
            "condition_symptoms": {},
            "needs_assessment": True,
            "assessment_questions": [],
        }
        sats_colour = vitals_acuity.get("sats_colour")
        if sats_colour:
            result["sats_colour"] = sats_colour
            result["sats_priority"] = vitals_acuity.get("sats_priority", "")
            result["tews_score"] = vitals_acuity.get("tews_score", 0)
            result["sats_target_minutes"] = vitals_acuity.get("target_minutes", 240)
        return result

    def _build_full_response(
        self,
        tool_results: dict,
        condition_symptoms: dict,
        assessment_questions: list[dict],
        vitals: dict | None = None,
        symptoms: list[str] | None = None,
        labs: dict | None = None,
    ) -> dict:
        """Build the complete response deterministically from pre-computed data.

        All condition data comes from DB-sourced search results.
        condition_symptoms come from DB clinical_entities.
        assessment_questions come from the slim LLM call.
        vitals/symptoms/labs are passed through for reasoning rule matching.
        """
        extracted_symptoms_raw = tool_results.get("extract_symptoms", {}).get("symptoms", [])
        # Use parameter symptoms for rule matching, fallback to extracted
        rule_symptoms = symptoms or extracted_symptoms_raw
        safety = tool_results.get("check_safety_flags", {})
        vitals_acuity = tool_results.get("vitals_acuity", {})

        # Acuity: deterministic from vitals + red flags
        acuity = vitals_acuity.get("acuity", "routine")
        acuity_reasons = list(vitals_acuity.get("reasons", []))
        if safety.get("requires_escalation"):
            acuity = "urgent"
        for rf in safety.get("red_flags_triggered", []):
            reason = f"Red flag: {rf.get('flag', '')} ({rf.get('condition', '')})"
            if reason not in acuity_reasons:
                acuity_reasons.append(reason)

        # Conditions: from search results (same logic as _build_fallback_response)
        search_conditions = tool_results.get("search_conditions", {}).get("conditions", [])
        top_score = search_conditions[0].get("adjusted_score", 1) if search_conditions else 1
        conditions = []
        for s in search_conditions[:5]:
            code = s.get("stg_code", "")
            score = s.get("adjusted_score", s.get("raw_score", 0))
            confidence = round(min(score / max(top_score, 0.01) * 0.90, 0.95), 2)

            # Clean matched features for display
            raw_feats = s.get("matched_features", [])
            clean_feats = []
            for f in raw_feats[:5]:
                base = f.split(" (")[0] if " (" in f else f
                if base and base not in clean_feats:
                    clean_feats.append(base)

            cond_entry = {
                "condition_code": code,
                "condition_name": s.get("name", ""),
                "confidence": confidence,
                "matched_symptoms": clean_feats,
                "reasoning": "Matched via STG knowledge graph",
                "source_references": [f"STG {code}"] if code else [],
            }

            # Tag lab-confirmed conditions
            if any("lab-confirmed" in f for f in raw_feats):
                cond_entry["lab_confirmed"] = True
                cond_entry["reasoning"] = "Lab-confirmed diagnosis — matched via deterministic lab result scan"

            # Propagate care-setting fields for referral-only conditions
            if s.get("referral_required"):
                cond_entry["referral_required"] = True
                cond_entry["care_setting"] = s.get("care_setting", "hospital")
                cond_entry["source_tag"] = s.get("source_tag", "")
                cond_entry["reasoning"] = "REFER — identified via knowledge graph, requires higher-level care"

            conditions.append(cond_entry)

        # ── Wire reasoning rules into condition entries ───────────────────
        top_codes = [c["condition_code"] for c in conditions if c.get("condition_code")]

        # Referral triggers (830 STG rules)
        if top_codes:
            referral_map = get_referral_triggers(top_codes)
            for cond in conditions:
                cc = cond.get("condition_code", "")
                if cc in referral_map:
                    cond["referral_triggers"] = referral_map[cc]

        # Severity classification (80 STG rules)
        if top_codes and (vitals or rule_symptoms):
            for cond in conditions:
                cc = cond.get("condition_code", "")
                sev = classify_severity(cc, vitals=vitals, symptoms=rule_symptoms)
                if sev:
                    cond["severity_classification"] = sev

        # Lab threshold matching (57 STG rules)
        if top_codes and labs:
            lab_matches = match_lab_rules(labs, top_codes)
            if lab_matches:
                # Group by condition code
                for lm in lab_matches:
                    for cond in conditions:
                        if cond.get("condition_code") == lm["confirms_code"]:
                            cond.setdefault("lab_matches", []).append(lm)

        # Vital threshold alerts (49 condition-specific rules)
        if top_codes and vitals:
            vital_alerts = check_vital_rules(vitals, top_codes)
            if vital_alerts:
                for va in vital_alerts:
                    for cond in conditions:
                        if cond.get("condition_code") == va["condition_code"]:
                            cond.setdefault("vital_alerts", []).append(va)

        extracted_symptoms = tool_results.get("extract_symptoms", {}).get("symptoms", [])
        result = {
            "extracted_symptoms": extracted_symptoms,
            "acuity": acuity,
            "acuity_reasons": acuity_reasons,
            "acuity_sources": ["SATS (South African Triage Scale)"],
            "conditions": conditions,
            "condition_symptoms": condition_symptoms,
            "needs_assessment": True,
            "assessment_questions": assessment_questions,
        }

        # Add SATS-specific fields
        sats_colour = vitals_acuity.get("sats_colour")
        if sats_colour:
            result["sats_colour"] = sats_colour
            result["sats_priority"] = vitals_acuity.get("sats_priority", "")
            result["tews_score"] = vitals_acuity.get("tews_score", 0)
            result["sats_target_minutes"] = vitals_acuity.get("target_minutes", 240)

        # Match quality: "Does the STG actually cover this presentation?"
        from agents.scoring_config import (
            STRONG_MATCH_THRESHOLD, PARTIAL_MATCH_THRESHOLD,
            NO_CLEAR_MATCH_WARNING, PARTIAL_MATCH_WARNING,
        )
        top_adjusted = search_conditions[0].get("adjusted_score", 0) if search_conditions else 0

        # Lab-confirmed or vitals-based presentations are always strong matches
        has_lab = any(
            "lab-confirmed" in f
            for s in search_conditions[:1]
            for f in s.get("matched_features", [])
        )
        has_vitals = any(
            "vitals-based" in f
            for s in search_conditions[:1]
            for f in s.get("matched_features", [])
        )

        if has_lab or has_vitals or top_adjusted >= STRONG_MATCH_THRESHOLD:
            result["match_quality"] = "strong_match"
        elif top_adjusted >= PARTIAL_MATCH_THRESHOLD:
            result["match_quality"] = "partial_match"
            result["low_confidence_warning"] = PARTIAL_MATCH_WARNING
        else:
            result["match_quality"] = "no_clear_match"
            result["low_confidence_warning"] = NO_CLEAR_MATCH_WARNING

        return result

    def _build_condition_symptoms(
        self,
        verified_sorted: list[dict],
        features_by_condition: dict[int, list[dict]],
        condition_id_map: dict[str, int],
        reported_symptoms: set[str],
    ) -> dict:
        """Generate condition_symptoms from DB clinical features.

        For each condition (top 5), produce all verification questions from:
        1. RED_FLAG features (highest priority — determines urgency)
        2. diagnostic_feature not already in complaint (distinguishing)
        3. presenting_feature not already in complaint (confirming)

        Returns all features with metadata (is_red_flag, source_citation, grounding)
        so the frontend can render red flag badges and section role indicators.
        """
        condition_symptoms: dict[str, list[dict]] = {}
        reported_lower = {s.lower() for s in reported_symptoms}

        for v in verified_sorted[:5]:
            code = v["stg_code"]
            name = v["name"]
            cid = condition_id_map.get(code)
            if not cid:
                continue

            features = features_by_condition.get(cid, [])
            questions: list[dict] = []
            seen_names: set[str] = set()
            q_idx = 0

            for feat in features:
                feat_name = feat["name"]
                feat_lower = feat_name.lower()

                # Skip features already reported by the patient
                if any(feat_lower in r or r in feat_lower for r in reported_lower):
                    continue
                # Skip overly short/generic features
                if len(feat_name) < 4:
                    continue
                # Skip duplicates
                if feat_lower in seen_names:
                    continue
                seen_names.add(feat_lower)

                q_idx += 1
                question_text = self._feature_to_question(
                    feat_name, feat["relationship_type"]
                )
                is_red_flag = feat["relationship_type"] == "RED_FLAG"
                section_role = "danger_signs" if is_red_flag else feat.get("feature_type", "unknown")
                questions.append({
                    "id": f"cs_{code.replace('.', '_')}_{q_idx}",
                    "question": question_text,
                    "is_red_flag": is_red_flag,
                    "source_citation": f"graph:{code}:{feat_name}:{section_role}",
                    "grounding": "verified",
                })

            if questions:
                condition_symptoms[name] = questions

        return condition_symptoms

    @staticmethod
    def _build_deterministic_questions(
        patient: dict | None,
        complaint: str,
        symptoms: list[str],
    ) -> list[dict]:
        """Inject deterministic assessment questions based on demographics.

        Same pattern as asking about pregnancy status or vitals — these
        always fire for the right demographic + complaint combination.
        The LLM is not relied upon to remember to ask them.
        """
        if not patient:
            return []

        questions = []
        age = patient.get("age")
        sex = (patient.get("sex") or "").lower()

        if sex != "female" or not age:
            return []

        # Check if complaint is gynaecological
        gynae_keywords = _cache.keyword_sets.get("gynae_complaint", set()) if _cache else set()
        complaint_lower = complaint.lower()
        symptoms_lower = " ".join(s.lower() for s in symptoms)
        combined = complaint_lower + " " + symptoms_lower
        is_gynae = any(kw in combined for kw in gynae_keywords)

        if not is_gynae:
            return []

        # Women >50 with gynae complaint → ask menopausal status
        if age > 50:
            questions.append({
                "id": "det_menopause",
                "question": "Has the patient gone through menopause? (No menstrual periods for 12 or more months)",
                "type": "yes_no",
                "required": False,
                "round": 1,
                "source_citation": "",
                "grounding": "deterministic",
            })
            questions.append({
                "id": "det_pap_smear",
                "question": "When was the patient's last cervical cancer screening (Pap smear)?",
                "type": "free_text",
                "required": False,
                "round": 1,
                "source_citation": "STG 5.1",
                "grounding": "deterministic",
            })

        # Women 40-50 with gynae complaint → ask perimenopausal symptoms
        elif age >= 40:
            questions.append({
                "id": "det_perimenopause",
                "question": "Is the patient experiencing perimenopausal symptoms? (Irregular periods, hot flushes, night sweats, mood changes)",
                "type": "yes_no",
                "required": False,
                "round": 1,
                "source_citation": "",
                "grounding": "deterministic",
            })
            questions.append({
                "id": "det_last_period",
                "question": "When was the patient's last menstrual period?",
                "type": "free_text",
                "required": False,
                "round": 1,
                "source_citation": "",
                "grounding": "deterministic",
            })
            questions.append({
                "id": "det_pap_smear",
                "question": "When was the patient's last cervical cancer screening (Pap smear)?",
                "type": "free_text",
                "required": False,
                "round": 1,
                "source_citation": "STG 5.1",
                "grounding": "deterministic",
            })

        return questions

    _VERB_PREFIXES = ("can't", "cannot", "unable", "difficulty", "trouble",
                      "loss of", "no ", "not ", "absent")

    @staticmethod
    def _feature_to_question(feature_name: str, rel_type: str) -> str:
        """Convert a clinical feature name into a verification question."""
        name = feature_name.strip()
        if name.endswith("?"):
            return name
        lower = name.lower()
        if any(lower.startswith(v) for v in TriageAgent._VERB_PREFIXES):
            return f"Does the patient report {name}?"
        return f"Does the patient have {name}?"

    async def _generate_assessment_questions(
        self,
        complaint: str,
        patient: dict | None,
        verified_sorted: list[dict],
        features_by_condition: dict[int, list[dict]],
        condition_id_map: dict[str, int],
    ) -> tuple[list[str], list[dict]]:
        """Slim LLM call: re-rank conditions and generate assessment questions.

        Returns (ranked_codes, questions) where ranked_codes is the LLM's
        preferred ordering of condition stg_codes, and questions is the list
        of discriminating assessment questions.
        """
        # Build compact condition summary with distinguishing features
        lines = []
        for v in verified_sorted[:5]:
            code = v["stg_code"]
            cid = condition_id_map.get(code)
            feats = features_by_condition.get(cid, []) if cid else []
            feat_names = [f["name"] for f in feats[:6]]
            marker = ""
            if any("lab-confirmed" in f for f in v.get("matched_features", [])):
                marker = " (lab-confirmed)"
            elif any("vitals-based" in f for f in v.get("matched_features", [])):
                marker = " (vitals-based)"
            lines.append(f"- {code}: {v['name']}{marker} (features: {', '.join(feat_names)})")

        prompt = SLIM_ASSESSMENT_PROMPT.format(
            complaint=complaint,
            patient=json.dumps(patient) if patient else "none",
            conditions_summary="\n".join(lines),
        )

        try:
            response = await self._call_with_fallback(
                max_tokens=512,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = self._parse_json_response(text)

            # Extract ranked_codes and questions from response
            ranked_codes = parsed.get("ranked_codes", [])
            raw_questions = parsed.get("questions", [])
            if not isinstance(ranked_codes, list):
                ranked_codes = []
            if not isinstance(raw_questions, list):
                raw_questions = []

            # Convert string questions to full question dicts
            verified_codes = {v["stg_code"] for v in verified_sorted}
            questions = []
            for i, q in enumerate(raw_questions[:5]):
                if isinstance(q, str):
                    questions.append({
                        "id": f"hyp_{i+1}",
                        "question": q,
                        "type": "yes_no",
                        "required": False,
                        "round": 1,
                        "source_citation": "",
                        "grounding": "unverified",
                    })
                elif isinstance(q, dict):
                    # Handle if LLM still returns dict format
                    q.setdefault("id", f"hyp_{i+1}")
                    q.setdefault("type", "yes_no")
                    q.setdefault("required", False)
                    q.setdefault("round", 1)
                    q.setdefault("source_citation", "")
                    q.setdefault("grounding", "unverified")
                    questions.append(q)

            # Only keep ranked_codes that are actually in our verified set
            ranked_codes = [c for c in ranked_codes if c in verified_codes]

            return ranked_codes, questions
        except Exception as e:
            logger.error(f"Assessment question generation failed: {e}")
            return [], []
