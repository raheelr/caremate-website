"""
Safety Checker
--------------
Separate Haiku call that reviews the triage agent's output before
it reaches the nurse. Defense in depth — catches things the agent
might miss.

Checks:
1. Missed RED_FLAG danger signs
2. Dangerous omissions in the differential
3. Acuity appropriateness
4. Scope of practice issues
"""

import json
import logging
import anthropic

logger = logging.getLogger(__name__)


SAFETY_SYSTEM = """You are a clinical safety reviewer for a South African primary healthcare triage system.

Review the triage output and check for:

1. MISSED RED FLAGS: Are there symptoms suggesting urgent conditions (e.g. meningitis, sepsis,
   ectopic pregnancy, acute abdomen) that weren't flagged?
2. DANGEROUS OMISSIONS: Is a likely serious condition missing from the differential?
3. ACUITY APPROPRIATENESS: Should the urgency be higher given the symptom combination?
4. SCOPE OF PRACTICE: Any recommendations beyond PHC nurse scope?

If everything is safe, return EXACTLY: {"safe": true}

If there are concerns, return:
{
  "safe": false,
  "concerns": ["specific concern 1", "specific concern 2"],
  "corrected_acuity": "urgent" or null,
  "missing_conditions": ["condition that should be considered"]
}

Return ONLY valid JSON, no explanation."""


class SafetyChecker:

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.model = "claude-haiku-4-5-20251001"

    async def check(
        self,
        triage_output: dict,
        complaint: str,
        patient: dict = None,
    ) -> dict:
        """
        Review triage output for safety. Returns the output unchanged if safe,
        or annotated with safety concerns and escalated acuity if not.
        """
        # Build a concise review prompt (don't send entire condition details)
        conditions_summary = []
        for c in triage_output.get("conditions", [])[:5]:
            conditions_summary.append({
                "name": c.get("condition_name", ""),
                "confidence": c.get("confidence", 0),
                "matched_symptoms": c.get("matched_symptoms", []),
            })

        review_input = {
            "complaint": complaint,
            "patient": patient or {},
            "acuity": triage_output.get("acuity", "routine"),
            "acuity_reasons": triage_output.get("acuity_reasons", []),
            "extracted_symptoms": triage_output.get("extracted_symptoms", []),
            "conditions": conditions_summary,
        }

        review_prompt = (
            f"Review this triage output for clinical safety:\n\n"
            f"{json.dumps(review_input, indent=2, default=str)}"
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SAFETY_SYSTEM,
                messages=[{"role": "user", "content": review_prompt}],
            )

            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            review = json.loads(text)

        except (json.JSONDecodeError, anthropic.APIError, IndexError) as e:
            logger.warning(f"Safety check parse/API error: {e} — passing through")
            return triage_output

        if review.get("safe", True):
            return triage_output

        # Annotate output with safety concerns
        triage_output["safety_review"] = {
            "concerns": review.get("concerns", []),
            "missing_conditions": review.get("missing_conditions", []),
            "reviewed": True,
        }

        # Escalate acuity if safety checker says so
        if review.get("corrected_acuity"):
            original = triage_output.get("acuity", "routine")
            corrected = review["corrected_acuity"]
            # Only escalate, never downgrade
            acuity_rank = {"routine": 0, "priority": 1, "urgent": 2}
            if acuity_rank.get(corrected, 0) > acuity_rank.get(original, 0):
                triage_output["acuity"] = corrected
                concerns = review.get("concerns", ["Safety review"])
                triage_output.setdefault("acuity_reasons", []).append(
                    f"Safety review escalated: {'; '.join(concerns)}"
                )
                triage_output.setdefault("acuity_sources", []).append(
                    "Safety reviewer"
                )

        return triage_output
