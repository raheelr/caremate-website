"""
STG Condition Extractor
-----------------------
Two-pass extraction using Claude:

Pass 1 (Haiku — fast, cheap): 
  Extract structured data + flag ambiguous conditions

Pass 2 (Sonnet + extended thinking — only for flagged conditions):
  Resolve ambiguity, ensure edge type accuracy

This is the core intelligence of the ingestion pipeline.
"""

import json
import re
import anthropic
from dataclasses import dataclass, field
from typing import Optional, Union
from segmenter import ConditionSegment

# Import MergedConditionInput if available (backward compat)
try:
    from multi_source_merger import MergedConditionInput
except ImportError:
    MergedConditionInput = None


# ── Pydantic-style schemas (as dicts for JSON parsing) ───────────────────────
# We use structured output via Claude's tool_use

EXTRACTION_TOOL = {
    "name": "extract_condition",
    "description": "Extract structured clinical data from an STG condition section",
    "input_schema": {
        "type": "object",
        "properties": {
            
            # Basic identification
            "condition_name_normalised": {
                "type": "string",
                "description": "Clean condition name, title case. e.g. 'Candidiasis, Oral (Thrush)'"
            },
            "icd10_codes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "ICD-10 codes for this condition"
            },
            
            # Clinical features (for Knowledge Graph edges)
            "clinical_features": {
                "type": "array",
                "description": "Symptoms/signs that indicate this condition — standard presentation",
                "items": {
                    "type": "object",
                    "properties": {
                        "feature": {"type": "string", "description": "The clinical feature (specific, not generic)"},
                        "feature_type": {
                            "type": "string", 
                            "enum": ["diagnostic_feature", "presenting_feature", "associated_feature"],
                            "description": "diagnostic=pathognomonic, presenting=common, associated=may be present"
                        },
                        "source_section": {"type": "string", "enum": ["DESCRIPTION", "DANGER_SIGNS"]}
                    },
                    "required": ["feature", "feature_type", "source_section"]
                }
            },
            
            # Danger signs — these become RED_FLAG edges
            "danger_signs": {
                "type": "array",
                "description": "Symptoms/signs that trigger urgent escalation",
                "items": {
                    "type": "object",
                    "properties": {
                        "sign": {"type": "string"},
                        "triggers_referral": {"type": "boolean"}
                    },
                    "required": ["sign", "triggers_referral"]
                }
            },
            
            # Medicines
            "medicines": {
                "type": "array",
                "description": "All medicines mentioned for treatment",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Generic medicine name"},
                        "route": {"type": "string", "description": "oral, IM, IV, topical, etc."},
                        "dose_adults": {"type": "string"},
                        "dose_children": {"type": "string"},
                        "frequency": {"type": "string"},
                        "duration": {"type": "string"},
                        "treatment_line": {
                            "type": "string",
                            "enum": ["first_line", "second_line", "alternative", "adjunct"],
                        },
                        "age_group": {"type": "string", "enum": ["adults", "children", "all"]},
                        "special_notes": {"type": "string"}
                    },
                    "required": ["name", "route", "treatment_line"]
                }
            },
            
            # Referral criteria
            "referral_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Conditions under which patient must be referred to a doctor"
            },
            
            # Population applicability
            "applies_to_children": {"type": "boolean"},
            "applies_to_adults": {"type": "boolean"},
            "applies_to_pregnant": {
                "type": ["boolean", "null"],
                "description": "null if not specified"
            },
            
            # Prerequisite context (for discriminator-absence suppression)
            "prerequisite_context": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Context required for this condition to be plausible. e.g. 'antipsychotic_medication_use', 'tb_exposure', 'streptococcal_infection_history'"
            },
            
            # Synonym suggestions
            "patient_language_synonyms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "How patients might describe symptoms of this condition in everyday language"
            },
            
            # Self-reported ambiguity (key for two-pass system)
            "ambiguity_flags": {
                "type": "object",
                "properties": {
                    "overlapping_symptom_sets": {
                        "type": "boolean",
                        "description": "Symptoms overlap heavily with another condition"
                    },
                    "unclear_edge_type": {
                        "type": "boolean", 
                        "description": "Unsure if a feature is PRESENT vs RED_FLAG"
                    },
                    "conflicting_treatment_info": {
                        "type": "boolean",
                        "description": "Treatment steps unclear or seemingly contradictory"
                    },
                    "sparse_clinical_features": {
                        "type": "boolean",
                        "description": "Fewer than 3 clinical features could be extracted"
                    },
                    "ambiguity_score": {
                        "type": "number",
                        "description": "0.0 to 1.0. >0.6 triggers Pass 2 with extended thinking"
                    },
                    "ambiguity_notes": {
                        "type": "string",
                        "description": "Explain specifically what is ambiguous"
                    }
                },
                "required": ["ambiguity_score", "ambiguity_notes"]
            }
        },
        "required": [
            "condition_name_normalised",
            "clinical_features",
            "medicines",
            "referral_criteria",
            "ambiguity_flags"
        ]
    }
}


# ── Extractor class ──────────────────────────────────────────────────────────

class ConditionExtractor:
    
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.pass1_model = "claude-haiku-4-5-20251001"    # Fast, cheap
        self.pass2_model = "claude-sonnet-4-6"            # Thorough, for ambiguous cases
        self.ambiguity_threshold = 0.6
        
        # Track costs
        self.pass1_count = 0
        self.pass2_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def extract(self, segment) -> dict:
        """
        Extract structured data from a condition segment.
        Accepts ConditionSegment or MergedConditionInput.
        Uses two-pass approach when ambiguity is detected.
        """
        print(f"  Extracting: {segment.display_name} (pages {segment.start_page}-{segment.end_page})")
        
        # Pass 1: Fast extraction
        result = self._pass1_extract(segment)
        self.pass1_count += 1
        
        # Guard: if result is not a dict (e.g. Claude returned plain text), recover
        if not isinstance(result, dict):
            result = {
                "condition_name_normalised": segment.name.title(),
                "clinical_features": [],
                "medicines": [],
                "referral_criteria": [],
                "ambiguity_flags": {
                    "ambiguity_score": 0.9,
                    "ambiguity_notes": f"Pass 1 returned unexpected type: {type(result)}. Sending to Pass 2."
                }
            }
        
        ambiguity_score = result.get('ambiguity_flags', {}).get('ambiguity_score', 0)
        
        if ambiguity_score >= self.ambiguity_threshold:
            print(f"    ⚠️  Ambiguous (score: {ambiguity_score:.1f}) — running Pass 2 with extended thinking")
            print(f"    Reason: {result.get('ambiguity_flags', {}).get('ambiguity_notes', '')}")
            result = self._pass2_extract(segment, result)
            self.pass2_count += 1
            
            # Guard pass2 result too
            if not isinstance(result, dict):
                result = {
                    "condition_name_normalised": segment.name.title(),
                    "clinical_features": [],
                    "medicines": [],
                    "referral_criteria": [],
                    "needs_review": True,
                    "ambiguity_flags": {"ambiguity_score": 1.0, "ambiguity_notes": "Both passes failed to return structured data"}
                }
        
        # Add segment metadata
        result['stg_code'] = segment.stg_code
        result['chapter_number'] = segment.chapter_number
        result['chapter_name'] = segment.chapter_name
        result['source_pages'] = list(range(segment.start_page, segment.end_page + 1))
        result['raw_text'] = segment.raw_text

        # Track multi-source metadata if available
        is_merged = MergedConditionInput and isinstance(segment, MergedConditionInput)
        if is_merged:
            result['_primary_source'] = segment.primary_source
            result['_has_tables'] = segment.has_tables
            result['_has_vision'] = segment.has_vision
            result['_pdfplumber_chars'] = segment.pdfplumber_chars
            result['_docling_chars'] = segment.docling_chars
            result['_vision_chars'] = segment.vision_chars

        return result
    
    def _build_prompt(self, segment, is_pass2: bool = False) -> str:
        """Build the extraction prompt. Works with ConditionSegment or MergedConditionInput."""

        pass2_note = """
This is a SECOND PASS extraction. The first pass flagged this condition as ambiguous.
Please reason carefully through each clinical feature and its edge type.
For every feature, ask: "Is this a standard presenting symptom, OR would its presence
alone warrant urgent escalation?" Only RED_FLAG edges should trigger acuity escalation.
""" if is_pass2 else ""

        # Source quality context for MergedConditionInput
        source_context = ""
        is_merged = MergedConditionInput and isinstance(segment, MergedConditionInput)
        if is_merged:
            source_context = f"""
SOURCE QUALITY CONTEXT:
- Primary source: {segment.primary_source}
- pdfplumber chars: {segment.pdfplumber_chars}
- Docling chars: {segment.docling_chars}
- Vision chars: {segment.vision_chars}
"""
            if segment.has_tables:
                source_context += """
TABLES DETECTED: Structured tables were extracted by Docling (AI-powered table parser).
Extract EVERY medicine row from these tables as separate medicine entries with dose, route,
frequency, duration, and treatment line. Do NOT skip any row. Tables appear at the end
of the text under "--- STRUCTURED TABLES ---".
"""
            if segment.has_vision:
                source_context += """
VISUAL CONTENT: Flowcharts/diagrams were extracted by Claude Vision and appear at the end
of the text under "--- VISUAL CONTENT ---". This includes algorithm decision paths and
any text visible in images. Treat this content as authoritative STG text.
"""

        return f"""You are extracting structured clinical data from a section of the
South African Standard Treatment Guidelines (STG/EML) for use in a clinical
decision support system for nurse practitioners.
{pass2_note}{source_context}
CRITICAL RULES:
1. Only extract features explicitly stated in the text — never infer
2. DESCRIPTION section → clinical_features with source_section="DESCRIPTION"
3. DANGER SIGNS section → danger_signs list AND clinical_features with source_section="DANGER_SIGNS"
4. A feature is "diagnostic_feature" only if it would strongly differentiate this condition
5. Be conservative: if unsure about edge type, flag as ambiguous
6. Do NOT include generic symptoms like "pain" — only qualified terms like "epigastric pain"
7. Every medicine must include its route and treatment line
8. TREATMENT LINE CLASSIFICATION — be precise:
   - first_line: the default starting treatment
   - second_line: explicitly used when first-line fails or is not tolerated ("if X fails, use Y", "if not tolerated", "switch to")
   - alternative: a parallel option that can be used instead of first-line in specific populations or circumstances
   - adjunct: used alongside another medicine, not as standalone treatment
   NOTE: "If fluoxetine fails or is poorly tolerated, use citalopram" → citalopram is SECOND_LINE not alternative
9. DEDUPLICATION: Each feature must appear EXACTLY ONCE. Do not list the same symptom/sign twice even if it appears in multiple sections or is paraphrased.
9. CAUTION BOXES: Any text under "CAUTION" or "NOTE" must be extracted as:
   - Safety warnings about medicines → add to the medicine's special_notes field
   - Clinical warnings (e.g. "ask about suicidal ideation") → add as danger_signs with triggers_referral=False
   - Contraindications → add to the relevant medicine's special_notes
10. REFERRAL CRITERIA: Extract ALL items from the REFERRAL section as referral_criteria strings. These are critical for nurses to know when to escalate.
11. TABLES: If dose information appears in a table (e.g. "Step 1: hydrochlorothiazide 12.5mg"), extract each medicine row as a separate medicine entry with the dose from that table row.
12. RISK FACTORS: Items listed as risk factors in the DESCRIPTION belong in clinical_features as "associated_feature" type — they help match patients to conditions.

CONDITION TEXT TO EXTRACT:
{'-'*60}
{segment.raw_text}
{'-'*60}

Extract all structured clinical data using the extract_condition tool.
Flag your own uncertainty honestly in ambiguity_flags."""
    
    def _pass1_extract(self, segment) -> dict:
        """Fast extraction with Haiku. Retries with exponential backoff."""
        import time

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.pass1_model,
                    max_tokens=4000,
                    tools=[EXTRACTION_TOOL],
                    tool_choice={"type": "tool", "name": "extract_condition"},
                    messages=[{
                        "role": "user",
                        "content": self._build_prompt(segment, is_pass2=False)
                    }]
                )

                self.total_input_tokens += response.usage.input_tokens
                self.total_output_tokens += response.usage.output_tokens

                for block in response.content:
                    if block.type == "tool_use" and block.name == "extract_condition":
                        return block.input
                break

            except Exception as e:
                err_str = str(e).lower()
                is_retryable = ("overloaded" in err_str or "rate" in err_str
                                or "529" in err_str or "timeout" in err_str
                                or "connection" in err_str)
                if is_retryable and attempt < max_retries - 1:
                    wait = min(15 * (2 ** attempt), 120)  # 15, 30, 60, 120
                    print(f"    API error (attempt {attempt+1}/{max_retries}) — waiting {wait}s...", flush=True)
                    time.sleep(wait)
                    continue
                raise
        
        # Fallback if no tool use
        return {
            "condition_name_normalised": segment.name.title(),
            "clinical_features": [],
            "medicines": [],
            "referral_criteria": [],
            "_extraction_failed": True,
            "ambiguity_flags": {
                "ambiguity_score": 0.8,
                "ambiguity_notes": "Extraction failed — needs manual review"
            }
        }
    
    def _pass2_extract(self, segment, pass1_result: dict) -> dict:
        """Thorough extraction with Sonnet + extended thinking for ambiguous conditions."""
        
        pass1_summary = json.dumps({
            "pass1_features": pass1_result.get("clinical_features", []),
            "pass1_ambiguity": pass1_result.get("ambiguity_flags", {})
        }, indent=2)
        
        prompt = self._build_prompt(segment, is_pass2=True)
        prompt += f"\n\nPASS 1 RESULT FOR REFERENCE:\n{pass1_summary}"
        
        response = self.client.messages.create(
            model=self.pass2_model,
            max_tokens=8000,
            thinking={
                "type": "enabled",
                "budget_tokens": 3000  # Targeted thinking budget
            },
            tools=[EXTRACTION_TOOL],
            # Note: cannot force tool_choice when thinking is enabled
            # Claude will still use the tool — it's the only one available
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens
        
        for block in response.content:
            if block.type == "tool_use" and block.name == "extract_condition":
                result = block.input
                # Mark that this went through Pass 2
                result['ambiguity_flags']['resolved_by_pass2'] = True
                return result
        
        # If Pass 2 also fails, return Pass 1 result with review flag
        pass1_result['needs_review'] = True
        return pass1_result
    
    def print_cost_summary(self):
        """Print token usage summary."""
        print(f"\n{'='*50}")
        print(f"EXTRACTION COST SUMMARY")
        print(f"{'='*50}")
        print(f"Pass 1 (Haiku):   {self.pass1_count} conditions")
        print(f"Pass 2 (Sonnet):  {self.pass2_count} conditions")
        print(f"Total input tokens:  {self.total_input_tokens:,}")
        print(f"Total output tokens: {self.total_output_tokens:,}")
        
        # Rough cost estimate
        # Haiku: $0.25/M input, $1.25/M output
        # Sonnet: $3/M input, $15/M output (estimates)
        pass1_cost = (self.total_input_tokens * 0.00000025) + (self.total_output_tokens * 0.00000125)
        print(f"Estimated cost: ~${pass1_cost:.2f} (rough estimate)")


# ── Test on a single condition ───────────────────────────────────────────────

if __name__ == "__main__":
    from segmenter import STGSegmenter
    
    pdf_path = "/mnt/user-data/uploads/Primary-Healthcare-Standard-Treatment-Guidelines-and-Essential-Medicines-List-8th-Edition-2024-Updated-December-2025__1_.pdf"
    
    print("Segmenting PDF...")
    segmenter = STGSegmenter(pdf_path)
    segments = segmenter.segment()
    
    # Test on a few interesting conditions
    test_conditions = ["1.2", "2.9.1", "4.7.1"]  # Thrush, Diarrhoea in children, Hypertension
    
    extractor = ConditionExtractor()
    
    for stg_code in test_conditions:
        seg = next((s for s in segments if s.stg_code == stg_code), None)
        if not seg:
            print(f"Condition {stg_code} not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Testing extraction: {seg.display_name}")
        print(f"{'='*60}")
        
        result = extractor.extract(seg)
        
        print(f"\nName: {result.get('condition_name_normalised')}")
        print(f"Clinical features: {len(result.get('clinical_features', []))}")
        print(f"Danger signs: {len(result.get('danger_signs', []))}")
        print(f"Medicines: {len(result.get('medicines', []))}")
        print(f"Referral criteria: {len(result.get('referral_criteria', []))}")
        print(f"Ambiguity score: {result.get('ambiguity_flags', {}).get('ambiguity_score', 0)}")
        
        print("\nMedicines found:")
        for med in result.get('medicines', []):
            print(f"  • {med.get('name')} ({med.get('route')}) — {med.get('treatment_line')}")
            if med.get('dose_adults'):
                print(f"    Adults: {med.get('dose_adults')} {med.get('frequency', '')} {med.get('duration', '')}")
        
        print("\nClinical features:")
        for feat in result.get('clinical_features', [])[:5]:
            print(f"  • [{feat.get('feature_type')}] {feat.get('feature')} ({feat.get('source_section')})")
    
    extractor.print_cost_summary()
