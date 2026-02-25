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
import anthropic
import asyncpg
from typing import Optional

from agents.tools import TOOL_HANDLERS

logger = logging.getLogger(__name__)


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
Conditions: {verified_conditions}
Safety: {safety_data}

REQUIREMENTS:
- Return 4-5 conditions ranked by likelihood (top condition highest confidence)
- condition_symptoms: include 3-4 verification questions for EACH condition
- extracted_symptoms: include all identified symptoms as descriptive phrases
- assessment_questions: 4-5 discriminating questions across top conditions
- condition_code/condition_name MUST match verified list exactly
- source_references format "STG X.Y"
- confidence 0.0-1.0

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


class TriageAgent:

    def __init__(self, pool: asyncpg.Pool):
        self.client = anthropic.AsyncAnthropic()
        self.pool = pool
        self.haiku = "claude-haiku-4-5-20251001"
        self.sonnet = "claude-sonnet-4-6"

    # ── Main analyse endpoint ────────────────────────────────────────────────

    async def analyze(
        self,
        complaint: str,
        patient: Optional[dict] = None,
        vitals: Optional[dict] = None,
        core_history: Optional[dict] = None,
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
        if patient:
            age = patient.get("age")
            sex = patient.get("sex", "unknown")
            preg = patient.get("pregnancy_status", "unknown")
            patient_context = f"Patient: age {age}, sex {sex}, pregnancy: {preg}"
            if age and age < 12:
                is_child = True
            if preg == "pregnant":
                is_pregnant = True

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
            extract_response = await self.client.messages.create(
                model=self.haiku,
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

        logger.info(f"[{time.monotonic()-t0:.1f}s] Extracted symptoms: {symptoms}")
        tool_results = {
            "extract_symptoms": {"symptoms": symptoms, "count": len(symptoms)},
        }

        # ── Step 2: Direct DB calls (no LLM needed) ──────────────────────
        # 2a. Expand synonyms
        expand_result = await TOOL_HANDLERS["expand_synonyms"](
            {"clinical_terms": symptoms}, self.pool
        )
        tool_results["expand_synonyms"] = expand_result
        expanded_terms = expand_result.get("expanded_terms", symptoms)
        logger.info(f"[{time.monotonic()-t0:.1f}s] Expanded to {len(expanded_terms)} terms")

        # 2b. Search conditions
        search_result = await TOOL_HANDLERS["search_conditions"](
            {"symptoms": expanded_terms, "original_symptoms": symptoms,
             "patient_is_child": is_child, "patient_is_pregnant": is_pregnant},
            self.pool,
        )
        tool_results["search_conditions"] = search_result
        conditions = search_result.get("conditions", [])
        top_names = [f"{c.get('name','')} ({c.get('adjusted_score',0):.3f})" for c in conditions[:5]]
        logger.info(f"[{time.monotonic()-t0:.1f}s] Found {len(conditions)} conditions: {top_names}")

        # 2c. Parallel: safety flags + condition details + vitals acuity
        condition_ids = [c["id"] for c in conditions[:15]]
        top_ids = [c["id"] for c in conditions[:5]]

        async def _get_safety():
            if not condition_ids:
                return {}
            return await TOOL_HANDLERS["check_safety_flags"](
                {"symptoms": expanded_terms, "condition_ids": condition_ids,
                 "vitals": vitals or {}}, self.pool)

        async def _get_details():
            details = {}
            for cid in top_ids:
                d = await TOOL_HANDLERS["get_condition_detail"](
                    {"condition_id": cid}, self.pool)
                details[cid] = d
            return details

        safety_result, details = await asyncio.gather(_get_safety(), _get_details())
        tool_results["check_safety_flags"] = safety_result
        tool_results["condition_details"] = details
        acuity_info = self._compute_vitals_acuity(vitals or {})
        tool_results["vitals_acuity"] = acuity_info
        logger.info(f"[{time.monotonic()-t0:.1f}s] Safety + details done (parallel)")

        # ── Step 3: Synthesis + safety review in parallel ────────────────
        synthesis_task = asyncio.create_task(
            self._synthesize_analyze(complaint, patient, vitals, core_history, tool_results)
        )
        safety_review_task = asyncio.create_task(
            self._run_safety_review(complaint, patient, symptoms, conditions, acuity_info, vitals)
        )
        result, safety_review = await asyncio.gather(synthesis_task, safety_review_task)
        logger.info(f"[{time.monotonic()-t0:.1f}s] Synthesis + safety done (parallel)")

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
                    result["acuity"] = corrected
                    # Add each concern as a separate acuity reason (not one giant string)
                    for concern in safety_review.get("concerns", [])[:3]:
                        result.setdefault("acuity_reasons", []).append(concern)
                    result.setdefault("acuity_sources", []).append("Safety reviewer")

        # ── Add structured STG guidelines for each condition ───────────
        # Uses condition details already fetched in step 2c — no extra DB calls
        stg_guidelines = {}
        for cid, detail in details.items():
            if detail.get("error"):
                continue
            name = detail.get("name", "")
            stg_code = detail.get("stg_code", "")

            # Parse medicines into clean list
            meds = detail.get("medicines", [])
            medicine_list = []
            for m in (meds if isinstance(meds, list) else []):
                med_entry = {
                    "name": m.get("name", ""),
                    "treatment_line": m.get("treatment_line", ""),
                    "dose": m.get("dose_context", ""),
                    "special_notes": m.get("special_notes", ""),
                }
                if med_entry["name"]:
                    medicine_list.append(med_entry)

            # Parse referral criteria
            referral = detail.get("referral_criteria", [])
            if isinstance(referral, str):
                try:
                    referral = json.loads(referral)
                except (json.JSONDecodeError, TypeError):
                    referral = [referral] if referral else []

            stg_guidelines[name] = {
                "stg_code": stg_code,
                "description": detail.get("description", ""),
                "general_measures": detail.get("general_measures", ""),
                "danger_signs": detail.get("danger_signs", ""),
                "medicines": medicine_list,
                "referral_criteria": referral,
                "source_pages": detail.get("source_pages", []),
            }
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
            response = await self.client.messages.create(
                model=self.haiku,
                max_tokens=2048,
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

        # Don't generate more questions past round 5
        if current_round >= 5:
            result["next_round_questions"] = None

        if not request_next_round:
            result["next_round_questions"] = None

        return result

    # ── Private helpers ──────────────────────────────────────────────────────

    async def _run_safety_review(self, complaint, patient, symptoms, conditions, acuity_info, vitals=None):
        """Run safety review concurrently with synthesis using search results."""
        conditions_summary = [
            {"name": c.get("name", ""), "score": c.get("adjusted_score", 0)}
            for c in conditions[:5]
        ]
        review_input = json.dumps({
            "complaint": complaint,
            "patient": patient or {},
            "vitals": vitals or {},
            "acuity": acuity_info.get("acuity", "routine"),
            "acuity_reasons": acuity_info.get("reasons", []),
            "extracted_symptoms": symptoms,
            "conditions": conditions_summary,
        }, default=str)
        try:
            response = await self.client.messages.create(
                model=self.haiku,
                max_tokens=512,
                temperature=0,
                system=(
                    "Review triage for safety. Check for: missed red flags, dangerous omissions, "
                    "acuity appropriateness. Return JSON: {\"safe\":true} or "
                    "{\"safe\":false,\"concerns\":[\"...\"],\"corrected_acuity\":\"urgent\",\"missing_conditions\":[\"...\"]}"
                ),
                messages=[{"role": "user", "content": review_input}],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except Exception as e:
            logger.warning(f"Safety review error: {e}")
            return {"safe": True}

    @staticmethod
    def _compute_vitals_acuity(vitals: dict) -> dict:
        """Compute acuity from vitals thresholds — no LLM needed."""
        acuity = "routine"
        reasons = []
        if vitals.get("systolic") and vitals["systolic"] >= 180:
            acuity = "urgent"
            reasons.append(f"Severe hypertension: BP {vitals['systolic']}/{vitals.get('diastolic', '?')}")
        if vitals.get("oxygenSat") and vitals["oxygenSat"] < 92:
            acuity = "urgent"
            reasons.append(f"Low SpO2: {vitals['oxygenSat']}%")
        if vitals.get("temperature") and vitals["temperature"] >= 39.0:
            if acuity != "urgent":
                acuity = "priority"
            reasons.append(f"High fever: {vitals['temperature']}°C")
        if vitals.get("heartRate") and (vitals["heartRate"] > 120 or vitals["heartRate"] < 50):
            if acuity != "urgent":
                acuity = "priority"
            reasons.append(f"Abnormal heart rate: {vitals['heartRate']} bpm")
        if vitals.get("respiratoryRate") and vitals["respiratoryRate"] >= 30:
            acuity = "urgent"
            reasons.append(f"Tachypnoea: {vitals['respiratoryRate']}/min")
        return {"acuity": acuity, "reasons": reasons}

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

        # From search_conditions
        for c in tool_results.get("search_conditions", {}).get("conditions", []):
            code = c.get("stg_code", "")
            if code:
                verified[code] = {
                    "stg_code": code,
                    "name": c.get("name", ""),
                    "chapter": c.get("chapter_name", ""),
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
                    # Try to find it by name
                    matched_code = None
                    for vc, vdata in verified.items():
                        if vdata["name"].lower() == name.lower():
                            matched_code = vc
                            break
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

        # Scrub condition_symptoms keys — must match verified names
        raw_cs = result.get("condition_symptoms", {})
        clean_cs = {}
        for key, questions in raw_cs.items():
            # Find the verified name that matches
            matched_name = None
            for v in verified.values():
                if v["name"].lower() == key.lower():
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
        verified_text = "\n".join(
            f"- {v['stg_code']}: {v['name']}"
            for v in verified.values()
        ) or "No conditions found."

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
            response = await self.client.messages.create(
                model=self.haiku,
                max_tokens=2048,
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

        return result

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        """
        Robustly extract JSON from an LLM response.
        Handles: raw JSON, markdown-fenced JSON, JSON with trailing text.
        """
        text = text.strip()

        # Strip markdown fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object boundaries
        start = text.find("{")
        if start == -1:
            raise json.JSONDecodeError("No JSON object found", text, 0)

        # Find the matching closing brace
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])

        # Last resort: try parsing from the first { to the end
        raise json.JSONDecodeError("Unbalanced JSON braces", text, start)

    def _build_fallback_response(self, tool_results: dict) -> dict:
        """Build a response from raw tool results when synthesis fails.
        All references come directly from DB-sourced tool results — no hallucination possible."""
        symptoms = tool_results.get("extract_symptoms", {}).get("symptoms", [])
        score_data = tool_results.get("score_differential", {})
        scored = score_data.get("scored_conditions", [])
        safety = tool_results.get("check_safety_flags", {})

        conditions = []
        for s in scored[:5]:
            code = s.get("stg_code", "")
            conditions.append({
                "condition_code": code,
                "condition_name": s.get("name", ""),
                "confidence": s.get("confidence", 0),
                "matched_symptoms": s.get("matched_symptoms", []),
                "reasoning": "Matched via knowledge graph",
                "source_references": [f"STG {code}"] if code else [],
            })

        acuity = score_data.get("acuity", "routine")
        if safety.get("requires_escalation"):
            acuity = "urgent"

        return {
            "extracted_symptoms": symptoms,
            "acuity": acuity,
            "acuity_reasons": score_data.get("acuity_reasons", []),
            "acuity_sources": score_data.get("acuity_sources", []),
            "conditions": conditions,
            "condition_symptoms": {},
            "needs_assessment": True,
            "assessment_questions": [],
        }
