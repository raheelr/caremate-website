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

import json
import logging
import anthropic
import asyncpg
from typing import Optional

from agents.tools import TOOL_DEFINITIONS, TOOL_HANDLERS

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


SYNTHESIS_PROMPT = """Based on the triage analysis below, produce a JSON response matching this exact schema.

## Input
Complaint: {complaint}
Patient: {patient}
Vitals: {vitals}
Core history: {core_history}

## Tool Results
{tool_results}

## VERIFIED CONDITIONS (from the database — use ONLY these)
{verified_conditions}

## Required JSON Schema
Return a JSON object with these exact fields:
{{
  "extracted_symptoms": ["list of clinical terms identified"],
  "acuity": "routine" | "priority" | "urgent",
  "acuity_reasons": ["reasons for acuity level"],
  "acuity_sources": ["use ONLY references from the verified conditions list above"],
  "conditions": [
    {{
      "condition_code": "MUST be a stg_code from the verified list above",
      "condition_name": "MUST be the exact name from the verified list above",
      "confidence": 0.85,
      "matched_symptoms": ["symptoms that matched"],
      "reasoning": "why this condition fits",
      "source_references": ["MUST use format 'STG <stg_code>' from verified list ONLY"]
    }}
  ],
  "condition_symptoms": {{
    "condition name": [
      {{"id": "unique_id", "question": "Do you have X?"}}
    ]
  }},
  "needs_assessment": true,
  "assessment_questions": [
    {{
      "id": "unique_question_id",
      "question": "targeted follow-up question",
      "type": "yes_no",
      "required": false,
      "round": 1,
      "source_citation": "STG <stg_code> from verified list, or empty string if not traceable",
      "grounding": "verified"
    }}
  ]
}}

CRITICAL — CITATION RULES:
- You may ONLY cite condition_code values that appear in the VERIFIED CONDITIONS list
- You may ONLY use condition_name values that appear in the VERIFIED CONDITIONS list
- source_references MUST use the format "STG <stg_code>" where stg_code is from the verified list
- source_citation on questions MUST reference a verified stg_code, or be an empty string
- Do NOT invent STG section numbers, page numbers, or condition names not in the verified list
- Do NOT add descriptive suffixes to STG codes (e.g. "STG 11.1" is correct, "STG Section 11.1 – Tonsillitis" is NOT)
- acuity_sources: use "STG <stg_code>" for condition-derived reasons, "Standard: vital signs assessment" for vitals-derived reasons
- confidence is 0.0 to 1.0 (not percentage)
- acuity must be lowercase: "routine", "priority", or "urgent"
- Generate 3-5 discriminating assessment questions
- Each assessment question id should be unique like "hyp_conditioncode_1"
- grounding: "verified" if traceable to a verified stg_code, "unverified" otherwise
- Return ONLY valid JSON, no markdown fences or explanation
"""


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
        self.client = anthropic.Anthropic()
        self.pool = pool
        self.haiku = "claude-haiku-4-5-20251001"
        self.sonnet = "claude-sonnet-4-6"
        self.max_iterations = 8

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
        1. Haiku tool_use loop (extract → expand → search → score → check → detail)
        2. Sonnet synthesises structured JSON
        """
        user_message = self._build_analyze_prompt(complaint, patient, vitals, core_history)
        messages = [{"role": "user", "content": user_message}]
        tool_results = {}

        # ── Agent loop with Haiku ────────────────────────────────────────────
        for iteration in range(self.max_iterations):
            try:
                response = self.client.messages.create(
                    model=self.haiku,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=TOOL_DEFINITIONS,
                    messages=messages,
                )
            except anthropic.APIError as e:
                logger.error(f"Anthropic API error on iteration {iteration}: {e}")
                break

            if response.stop_reason == "tool_use":
                assistant_content = response.content
                tool_result_blocks = []

                for block in assistant_content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input
                        tool_use_id = block.id

                        logger.info(f"Tool call [{iteration}]: {tool_name}")

                        handler = TOOL_HANDLERS.get(tool_name)
                        if handler:
                            try:
                                result = await handler(tool_input, self.pool)
                            except Exception as e:
                                logger.error(f"Tool {tool_name} failed: {e}")
                                result = {"error": str(e)}
                        else:
                            result = {"error": f"Unknown tool: {tool_name}"}

                        tool_results[tool_name] = result
                        tool_result_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": json.dumps(result, default=str),
                        })

                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": tool_result_blocks})

            elif response.stop_reason == "end_turn":
                # Claude finished — capture any final text
                for block in response.content:
                    if hasattr(block, "text"):
                        tool_results["_agent_summary"] = block.text
                break

        # ── Synthesise with Sonnet ───────────────────────────────────────────
        return await self._synthesize_analyze(complaint, patient, vitals, core_history, tool_results)

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
            response = self.client.messages.create(
                model=self.haiku,
                max_tokens=2048,
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
        """Use Sonnet to produce the final AnalyzeResponse-shaped JSON."""
        # Build verified conditions lookup from DB-sourced tool results
        verified = self._extract_verified_conditions(tool_results)
        verified_text = "\n".join(
            f"- stg_code: {v['stg_code']}, name: {v['name']}, chapter: {v.get('chapter', '')}"
            for v in verified.values()
        ) or "No conditions found in database."

        # Clean tool_results for the prompt (trim overly large entries)
        clean_results = {}
        for k, v in tool_results.items():
            if k == "_agent_summary":
                clean_results[k] = v
                continue
            serialised = json.dumps(v, default=str)
            if len(serialised) > 4000:
                # Truncate large results by trimming list contents rather than raw string slice
                if isinstance(v, dict):
                    trimmed = {}
                    for dk, dv in v.items():
                        dv_str = json.dumps(dv, default=str)
                        if len(dv_str) > 2000:
                            if isinstance(dv, list):
                                trimmed[dk] = dv[:10]  # keep first 10 items
                            else:
                                trimmed[dk] = str(dv)[:500] + "...(truncated)"
                        else:
                            trimmed[dk] = dv
                    clean_results[k] = trimmed
                else:
                    clean_results[k] = v
            else:
                clean_results[k] = v

        prompt = SYNTHESIS_PROMPT.format(
            complaint=complaint,
            patient=json.dumps(patient) if patient else "Not provided",
            vitals=json.dumps(vitals) if vitals else "Not provided",
            core_history=json.dumps(core_history) if core_history else "Not provided",
            tool_results=json.dumps(clean_results, indent=2, default=str),
            verified_conditions=verified_text,
        )

        try:
            response = self.client.messages.create(
                model=self.sonnet,
                max_tokens=4096,
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
