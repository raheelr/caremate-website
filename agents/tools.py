"""
Triage Agent Tools
------------------
Six tools for the Anthropic tool_use loop:
1. extract_symptoms    — NL complaint → clinical terms (Claude does the extraction)
2. expand_synonyms     — clinical terms → patient language variants from DB
3. search_conditions   — symptoms → matching conditions from knowledge graph
4. score_differential  — rank conditions by feature weights + RED_FLAG bonus
5. get_condition_detail — full STG entry for one condition
6. check_safety_flags  — RED_FLAG features + vitals thresholds → escalation
"""

import json
import logging
import asyncpg
from db.database import (
    get_conditions_for_symptoms,
    get_conditions_for_medications,
    get_condition_detail,
    get_red_flag_matches,
    get_condition_red_flags,
    get_condition_prerequisites,
    get_condition_prerequisites_batch,
    resolve_to_canonical,
    vector_search_conditions,
)

logger = logging.getLogger(__name__)

# South African PHC prevalence tiers (based on SA burden of disease data).
# Conditions seen daily in SA PHC clinics get a small score boost (1.15x-1.25x)
# when multiple conditions score similarly. This breaks ties in favour of the
# epidemiologically more likely diagnosis.
SA_PREVALENCE_BOOST = {
    # Tier 1: Very high prevalence — seen multiple times daily in SA PHC
    "high": 1.25,
    # Tier 2: Common — seen regularly
    "moderate": 1.15,
    # Tier 3: Everything else — no boost
}
_PREVALENCE_TIER = {
    # Respiratory
    "17.2": "high",    # Asthma
    "17.3.3": "high",  # Pneumonia
    "17.3.1": "high",  # Influenza
    "17.1.5": "high",  # COPD
    "19.2": "high",    # Common cold
    "19.6": "high",    # Tonsillitis/Pharyngitis
    # Cardiovascular
    "4.7": "high",     # Hypertension
    "4.7.1": "high",   # Hypertension in adults
    # Metabolic
    "3.4": "high",     # Type 2 diabetes
    "3.4.1": "high",   # Diabetes
    # GI
    "2.2": "high",     # Dyspepsia/heartburn
    "2.9": "high",     # Diarrhoea
    "2.1": "moderate",  # Abdominal pain
    "2.3": "moderate",  # Peptic ulcer
    # Infectious
    "17.4.1": "high",  # Pulmonary TB
    "10.7": "moderate", # Malaria
    "8.2": "high",     # UTI
    # STI
    "12.1": "high",    # VDS
    "12.5": "moderate", # GUS
    # HIV
    "11.1": "high",    # ART
    # Mental health
    "16.4.1": "moderate", # Depression
    "16.3": "moderate",   # Anxiety
    # Musculoskeletal
    "14.5": "moderate", # Osteoarthritis
    "14.3": "moderate", # Gout
    # Dermatology
    "5.8.2": "moderate", # Eczema
    "5.5": "moderate",   # Fungal skin
    # Paediatric
    "2.9.1": "high",    # Paediatric diarrhoea
    "17.1.2": "moderate", # Paediatric asthma
}


def _apply_prevalence_boost(conditions: list[dict]) -> list[dict]:
    """Apply SA prevalence boost to conditions with matching STG codes."""
    for c in conditions:
        code = c.get("stg_code", "")
        # Check exact code and parent codes (e.g., 4.7.1 → check 4.7.1 then 4.7)
        tier = _PREVALENCE_TIER.get(code)
        if not tier and "." in code:
            parent = code.rsplit(".", 1)[0]
            tier = _PREVALENCE_TIER.get(parent)
        if tier:
            boost = SA_PREVALENCE_BOOST[tier]
            c["adjusted_score"] = round(c.get("adjusted_score", 0) * boost, 3)
    return conditions


async def _filter_parent_headings(conn, conditions: list[dict]) -> list[dict]:
    """
    Remove parent heading conditions when a populated child exists.
    E.g., "4.7 Hypertension" is removed if "4.7.1 Hypertension In Adults" exists
    with non-empty description_text.
    """
    codes = [c.get("stg_code", "") for c in conditions if c.get("stg_code")]
    if not codes:
        return conditions

    rows = await conn.fetch("""
        SELECT DISTINCT parent_code FROM (
            SELECT unnest($1::text[]) as parent_code
        ) p WHERE EXISTS (
            SELECT 1 FROM conditions c
            WHERE c.stg_code LIKE parent_code || '.%'
            AND c.stg_code != parent_code
            AND c.description_text IS NOT NULL
            AND TRIM(c.description_text) != ''
        )
    """, codes)
    parent_codes = {r["parent_code"] for r in rows}

    if not parent_codes:
        return conditions

    filtered = [c for c in conditions if c.get("stg_code", "") not in parent_codes]
    if len(filtered) < len(conditions):
        removed = len(conditions) - len(filtered)
        logger.info(f"Filtered {removed} parent heading condition(s) from results")
    return filtered


# ── Tool Definitions (Anthropic tool_use JSON schemas) ───────────────────────

EXTRACT_SYMPTOMS_TOOL = {
    "name": "extract_symptoms",
    "description": (
        "Extract standardised clinical symptoms from a nurse's free-text complaint. "
        "Return specific clinical terms (e.g. 'dysuria', 'epigastric pain', 'productive cough'), "
        "not vague terms like 'pain' or 'feeling unwell'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "symptoms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of standardised clinical terms extracted from the complaint"
            },
            "patient_context": {
                "type": "object",
                "properties": {
                    "is_child": {"type": "boolean", "description": "Patient is under 12 years"},
                    "is_pregnant": {"type": "boolean", "description": "Patient is pregnant"},
                    "age_relevant": {"type": "boolean", "description": "Age significantly changes the differential"}
                },
                "description": "Contextual flags derived from the complaint or patient info"
            }
        },
        "required": ["symptoms"]
    }
}

EXPAND_SYNONYMS_TOOL = {
    "name": "expand_synonyms",
    "description": (
        "Expand clinical terms with patient-language synonyms from the synonym database. "
        "This broadens the search to catch terms like 'burning when I pee' for 'dysuria'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "clinical_terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Clinical terms to expand with synonyms"
            }
        },
        "required": ["clinical_terms"]
    }
}

SEARCH_CONDITIONS_TOOL = {
    "name": "search_conditions",
    "description": (
        "Search the clinical knowledge graph for conditions matching a set of symptoms. "
        "Returns ranked conditions with weighted scores. "
        "ALWAYS use this with symptoms, NEVER search by condition name. "
        "Pass BOTH the expanded symptom list AND the original extracted symptoms — "
        "the original list is used to score conditions that match multiple distinct complaints higher."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "symptoms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Full expanded symptom list (from expand_synonyms)"
            },
            "original_symptoms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The original extracted symptoms BEFORE expansion (from extract_symptoms)"
            },
            "patient_is_child": {"type": "boolean", "default": False},
            "patient_is_pregnant": {"type": "boolean", "default": False},
            "limit": {"type": "integer", "default": 10}
        },
        "required": ["symptoms"]
    }
}

SCORE_DIFFERENTIAL_TOOL = {
    "name": "score_differential",
    "description": (
        "Score and rank conditions from the search results. "
        "Applies feature weights (diagnostic=0.18, presenting=0.12, associated=0.08), "
        "RED_FLAG bonus (+0.10), and vitals-based acuity."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "condition_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Condition IDs to score"
            },
            "symptoms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The matched symptoms for context"
            },
            "vitals": {
                "type": "object",
                "description": "Patient vitals for acuity scoring"
            }
        },
        "required": ["condition_ids", "symptoms"]
    }
}

GET_CONDITION_DETAIL_TOOL = {
    "name": "get_condition_detail",
    "description": (
        "Get the full STG guideline entry for a condition including description, "
        "danger signs, medicines, dosing, and referral criteria."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "condition_id": {"type": "integer", "description": "Database ID of the condition"}
        },
        "required": ["condition_id"]
    }
}

CHECK_SAFETY_FLAGS_TOOL = {
    "name": "check_safety_flags",
    "description": (
        "Check patient symptoms against RED_FLAG danger signs for all matched conditions. "
        "Also checks vitals thresholds. Returns any triggered safety escalations."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "symptoms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "All symptoms identified so far"
            },
            "condition_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Condition IDs from the current differential"
            },
            "vitals": {
                "type": "object",
                "description": "Patient vitals"
            }
        },
        "required": ["symptoms", "condition_ids"]
    }
}

# All tool definitions for the Anthropic API
TOOL_DEFINITIONS = [
    EXTRACT_SYMPTOMS_TOOL,
    EXPAND_SYNONYMS_TOOL,
    SEARCH_CONDITIONS_TOOL,
    SCORE_DIFFERENTIAL_TOOL,
    GET_CONDITION_DETAIL_TOOL,
    CHECK_SAFETY_FLAGS_TOOL,
]


# ── Tool Handlers (execute tool calls against the database) ──────────────────

async def handle_extract_symptoms(tool_input: dict, pool: asyncpg.Pool) -> dict:
    """
    Claude's tool call IS the extraction — we just normalise and pass through.
    """
    symptoms = tool_input.get("symptoms", [])
    symptoms = [s.lower().strip() for s in symptoms if s.strip()]
    context = tool_input.get("patient_context", {})
    return {
        "symptoms": symptoms,
        "count": len(symptoms),
        "patient_context": context,
    }


async def handle_expand_synonyms(tool_input: dict, pool: asyncpg.Pool) -> dict:
    """
    Expand clinical terms to match canonical entity names in the DB.

    Three expansion strategies:
      1. synonym_rings: forward + reverse lookup for patient language variants
      2. resolve_to_canonical: trigram similarity + word overlap against clinical_entities
         (bridges gaps like "sore throat" → "painful red throat")
      3. Filter: only keep terms that are actual SYMPTOM entities (not condition names)
    """
    terms = [t.lower().strip() for t in tool_input.get("clinical_terms", [])]
    if not terms:
        return {"original_terms": [], "expanded_terms": [], "canonical_matches": {}, "expansion_count": 0}

    async with pool.acquire() as conn:
        # 1. Synonym rings — forward + reverse
        forward = await conn.fetch("""
            SELECT canonical_term, synonym
            FROM synonym_rings
            WHERE canonical_term = ANY($1::text[])
        """, terms)

        reverse = await conn.fetch("""
            SELECT canonical_term, synonym
            FROM synonym_rings
            WHERE synonym = ANY($1::text[])
        """, terms)

        # Collect synonym candidates
        synonym_terms = set()
        for row in forward:
            synonym_terms.add(row["synonym"])
        for row in reverse:
            synonym_terms.add(row["canonical_term"])

        # Filter synonym results: keep only SYMPTOM entities, discard condition names
        if synonym_terms:
            symptom_entities = await conn.fetch("""
                SELECT canonical_name FROM clinical_entities
                WHERE canonical_name = ANY($1::text[])
                AND entity_type = 'SYMPTOM'
            """, list(synonym_terms))
            synonym_symptoms = {r["canonical_name"] for r in symptom_entities}
        else:
            synonym_symptoms = set()

        # 2. Fuzzy resolve: trigram + word overlap against clinical_entities
        canonical_map = await resolve_to_canonical(conn, terms)

    # Merge all expanded terms
    all_terms = set(terms)
    all_terms.update(synonym_symptoms)
    for matches in canonical_map.values():
        all_terms.update(matches)

    return {
        "original_terms": terms,
        "expanded_terms": sorted(all_terms),
        "canonical_matches": canonical_map,
        "expansion_count": len(all_terms) - len(terms),
    }


async def handle_search_conditions(tool_input: dict, pool: asyncpg.Pool) -> dict:
    """
    Search knowledge graph for conditions matching symptoms.

    Scoring: conditions matching features from MULTIPLE distinct original
    symptoms rank higher than those matching many variants of a single symptom.
    This prevents "fever" variants from drowning out the chief complaint.
    """
    symptoms = [s.lower().strip() for s in tool_input.get("symptoms", [])]
    original_symptoms = [s.lower().strip() for s in tool_input.get("original_symptoms", [])]
    is_child = tool_input.get("patient_is_child", False)
    is_pregnant = tool_input.get("patient_is_pregnant", False)
    patient_sex = tool_input.get("patient_sex")
    patient_age = tool_input.get("patient_age")
    medications = tool_input.get("medications", [])
    limit = tool_input.get("limit", 10)

    # If no original_symptoms provided, treat each symptom as its own group
    if not original_symptoms:
        original_symptoms = symptoms

    async with pool.acquire() as conn:
        # Build the original→canonical mapping for grouping
        # This tells us "sore throat" resolved to {"painful red throat", "throat irritation", ...}
        orig_canonical_map = await resolve_to_canonical(conn, original_symptoms)

        # Collect ALL search terms: expanded symptoms + canonical resolutions
        all_search_terms = set(symptoms)
        for matches in orig_canonical_map.values():
            all_search_terms.update(matches)
        all_search_terms = sorted(all_search_terms)

        # Search with the full term set, generous limit
        conditions = await get_conditions_for_symptoms(
            conn, all_search_terms, is_child, is_pregnant,
            patient_sex=patient_sex, patient_age=patient_age, limit=50
        )

        # Re-score with group awareness
        for c in conditions:
            c["matched_features"] = list(c.get("matched_features") or [])
            c["raw_score"] = float(c.get("raw_score", 0))
            c["extraction_confidence"] = float(c.get("extraction_confidence", 1.0))

            # Count distinct ORIGINAL symptom groups this condition covers
            groups_covered = set()
            for feat in c["matched_features"]:
                feat_lower = feat.lower()
                for orig_term, resolved_set in orig_canonical_map.items():
                    if feat_lower == orig_term or feat_lower in resolved_set:
                        groups_covered.add(orig_term)
                        break
                else:
                    # Direct match from expanded list — find closest original
                    for orig in original_symptoms:
                        if feat_lower == orig:
                            groups_covered.add(orig)
                            break

            c["symptom_groups_matched"] = len(groups_covered)
            # Boost: multiply raw_score by number of distinct symptom groups
            c["adjusted_score"] = round(
                c["raw_score"] * max(c["symptom_groups_matched"], 1), 3
            )

        # Prerequisite penalty: conditions requiring specific context (e.g., HIV+)
        # score lower when that context is unconfirmed by the patient
        # Batch query to avoid N+1 problem
        all_cids = [c["id"] for c in conditions if c.get("id")]
        prereqs_map = await get_condition_prerequisites_batch(conn, all_cids)
        for c in conditions:
            prereqs = prereqs_map.get(c["id"], [])
            if prereqs:
                prereq_names = [p["prerequisite"] for p in prereqs]
                c["adjusted_score"] = round(c.get("adjusted_score", 0) * 0.5, 3)
                c.setdefault("matched_features", []).append(
                    f"requires: {', '.join(prereq_names)}"
                )

        # SA prevalence boost: common PHC conditions get slight score boost
        conditions = _apply_prevalence_boost(conditions)

        # Sort by adjusted_score, then raw_score as tiebreak
        conditions.sort(key=lambda c: (c["adjusted_score"], c["raw_score"]), reverse=True)
        conditions = conditions[:limit]

        # Vector search (semantic safety net) — catches conditions where patient
        # phrasing doesn't match graph edges lexically. Gracefully skips if
        # embeddings not populated or VOYAGE_API_KEY not set.
        try:
            from agents.embeddings import get_embedding
            query_text = " ".join(original_symptoms)
            query_embedding = await get_embedding(query_text)
            if query_embedding:
                vector_results = await vector_search_conditions(
                    conn, query_embedding,
                    patient_sex=patient_sex, patient_age=patient_age,
                    limit=15, min_similarity=0.65,
                )
                existing_ids_vec = {c["id"] for c in conditions}
                existing_by_id_vec = {c["id"]: c for c in conditions}
                for vr in vector_results:
                    cid = vr["id"]
                    sim = vr["similarity"]
                    # Convert similarity to score: 0.65-1.0 → 0.15-0.85
                    vec_score = round((sim - 0.65) / 0.35 * 0.70 + 0.15, 3)
                    feat = f"semantic match ({sim:.2f}, {vr['best_section']})"
                    if cid in existing_by_id_vec:
                        # Boost existing if vector score is higher
                        existing = existing_by_id_vec[cid]
                        if vec_score > existing.get("adjusted_score", 0):
                            existing["adjusted_score"] = vec_score
                            existing["raw_score"] = max(existing.get("raw_score", 0), vec_score)
                        if feat not in existing.get("matched_features", []):
                            existing.setdefault("matched_features", []).append(feat)
                    elif cid not in existing_ids_vec:
                        conditions.append({
                            "id": vr["id"],
                            "stg_code": vr["stg_code"],
                            "name": vr["name"],
                            "chapter_name": vr["chapter_name"],
                            "extraction_confidence": vr["extraction_confidence"],
                            "match_count": 1,
                            "raw_score": vec_score,
                            "matched_features": [feat],
                            "symptom_groups_matched": 1,
                            "adjusted_score": vec_score,
                        })
                        existing_ids_vec.add(cid)
                logger.info(f"Vector search: {len(vector_results)} results merged")
        except Exception as e:
            # Graceful degradation — vector search is optional
            logger.debug(f"Vector search skipped: {e}")

        # Fallback 1: condition NAME matching — if search terms appear in the
        # condition name, include/boost it. Aggregates across ALL terms so multi-term
        # matches score higher (e.g. "vaginal discharge" → "Vaginal Discharge Syndrome").
        existing_ids = {c["id"] for c in conditions}
        name_matches = {}  # condition_id -> {row, terms}
        for orig_term in original_symptoms:
            if len(orig_term) < 4:
                continue
            name_rows = await conn.fetch("""
                SELECT c.id, c.stg_code, c.name, c.chapter_name, c.extraction_confidence
                FROM conditions c
                WHERE c.name ILIKE $1
                AND ($2::text IS NULL
                     OR ($2 = 'male' AND c.applies_to_male IS NOT FALSE)
                     OR ($2 = 'female' AND c.applies_to_female IS NOT FALSE))
                LIMIT 5
            """, f"%{orig_term}%", patient_sex)
            for row in name_rows:
                cid = row["id"]
                if cid not in name_matches:
                    name_matches[cid] = {"row": dict(row), "terms": set()}
                name_matches[cid]["terms"].add(orig_term)

        existing_by_id = {c["id"]: c for c in conditions}
        for cid, data in sorted(
            name_matches.items(),
            key=lambda x: len(x[1]["terms"]),
            reverse=True,
        ):
            num_terms = len(data["terms"])
            # Score: 0.40 per matching term, capped at 0.90
            name_score = min(0.40 * num_terms, 0.90)
            matched_terms = sorted(data["terms"])

            if cid in existing_by_id:
                # BOOST existing condition if name match score is higher
                existing = existing_by_id[cid]
                if name_score > existing.get("adjusted_score", 0):
                    existing["adjusted_score"] = name_score
                    existing["raw_score"] = max(existing.get("raw_score", 0), name_score)
                for t in matched_terms:
                    feat = f"{t} (condition name match)"
                    if feat not in existing.get("matched_features", []):
                        existing.setdefault("matched_features", []).append(feat)
            elif cid not in existing_ids:
                row = data["row"]
                conditions.append({
                    "id": row["id"],
                    "stg_code": row["stg_code"],
                    "name": row["name"],
                    "chapter_name": row["chapter_name"],
                    "extraction_confidence": float(row["extraction_confidence"] or 1.0),
                    "match_count": num_terms,
                    "raw_score": name_score,
                    "matched_features": [f"{t} (condition name match)" for t in matched_terms],
                    "symptom_groups_matched": num_terms,
                    "adjusted_score": name_score,
                })
                existing_ids.add(row["id"])

        # Fallback 2: Multi-term knowledge_chunks search
        # Searches STG text chunks for EACH symptom, then aggregates — conditions
        # matching multiple distinct symptoms get proportionally higher scores.
        # This catches conditions where the STG text describes the symptoms
        # even though the knowledge graph lacks edges.
        chunk_matches = {}  # condition_id -> {row, terms, best_section}
        for orig_term in original_symptoms:
            if len(orig_term) < 3:
                continue
            chunk_rows = await conn.fetch("""
                SELECT DISTINCT ON (c.id) c.id, c.stg_code, c.name, c.chapter_name,
                       c.extraction_confidence, kc.section_role
                FROM knowledge_chunks kc
                JOIN conditions c ON c.id = kc.condition_id
                WHERE kc.chunk_text ILIKE $1
                AND ($2::text IS NULL
                     OR ($2 = 'male' AND c.applies_to_male IS NOT FALSE)
                     OR ($2 = 'female' AND c.applies_to_female IS NOT FALSE))
                ORDER BY c.id,
                    CASE kc.section_role
                        WHEN 'CLINICAL_PRESENTATION' THEN 1
                        WHEN 'DANGER_SIGNS' THEN 2
                        WHEN 'MANAGEMENT' THEN 3
                        WHEN 'DOSING_TABLE' THEN 4
                        ELSE 5
                    END
                LIMIT 15
            """, f"%{orig_term}%", patient_sex)
            for row in chunk_rows:
                cid = row["id"]
                if cid not in chunk_matches:
                    chunk_matches[cid] = {
                        "row": dict(row),
                        "terms": set(),
                        "best_section": row["section_role"],
                    }
                chunk_matches[cid]["terms"].add(orig_term)
                # Track the best (most relevant) section across all term matches
                section_priority = {
                    "CLINICAL_PRESENTATION": 1, "DANGER_SIGNS": 2,
                    "MANAGEMENT": 3, "DOSING_TABLE": 4,
                }
                current_priority = section_priority.get(chunk_matches[cid]["best_section"], 5)
                new_priority = section_priority.get(row["section_role"], 5)
                if new_priority < current_priority:
                    chunk_matches[cid]["best_section"] = row["section_role"]

        # Score and add/boost conditions based on multi-term chunk matches
        section_base_scores = {
            "CLINICAL_PRESENTATION": 0.22,
            "DANGER_SIGNS": 0.18,
            "MANAGEMENT": 0.10,
            "DOSING_TABLE": 0.08,
        }
        existing_by_id = {c["id"]: c for c in conditions}
        for cid, data in sorted(
            chunk_matches.items(),
            key=lambda x: len(x[1]["terms"]),
            reverse=True,
        ):
            num_terms = len(data["terms"])
            base = section_base_scores.get(data["best_section"], 0.06)
            # Multi-term scoring: base * num_terms, capped at 0.90
            chunk_score = min(base * num_terms, 0.90)
            matched_terms = sorted(data["terms"])

            if cid in existing_by_id:
                # BOOST existing condition if chunk score is higher
                existing = existing_by_id[cid]
                if chunk_score > existing.get("adjusted_score", 0):
                    existing["adjusted_score"] = chunk_score
                    existing["raw_score"] = max(existing.get("raw_score", 0), chunk_score)
                    existing["symptom_groups_matched"] = max(
                        existing.get("symptom_groups_matched", 0), num_terms
                    )
                    for t in matched_terms:
                        feat = f"{t} (STG text match)"
                        if feat not in existing.get("matched_features", []):
                            existing.setdefault("matched_features", []).append(feat)
            elif cid not in existing_ids:
                row = data["row"]
                conditions.append({
                    "id": row["id"],
                    "stg_code": row["stg_code"],
                    "name": row["name"],
                    "chapter_name": row["chapter_name"],
                    "extraction_confidence": float(row["extraction_confidence"] or 1.0),
                    "match_count": num_terms,
                    "raw_score": chunk_score,
                    "matched_features": [
                        f"{t} (STG text in {data['best_section']})" for t in matched_terms
                    ],
                    "symptom_groups_matched": num_terms,
                    "adjusted_score": chunk_score,
                })
                existing_ids.add(row["id"])

        # Fallback 3: Search conditions.description_text for multi-symptom matches.
        # ALWAYS runs — can both ADD new conditions and BOOST existing low-scored
        # conditions when their STG description mentions the patient's symptoms.
        # This fixes cases where the knowledge graph lacks edges but the STG text
        # clearly describes the symptoms (e.g., Malaria: "fever, chills, rigors").
        desc_matches = {}  # condition_id -> {row, terms}
        search_terms = [t for t in original_symptoms if len(t) >= 4]
        for term in search_terms:
            desc_rows = await conn.fetch("""
                SELECT c.id, c.stg_code, c.name, c.chapter_name, c.extraction_confidence
                FROM conditions c
                WHERE c.description_text ILIKE $1
                AND ($2::text IS NULL
                     OR ($2 = 'male' AND c.applies_to_male IS NOT FALSE)
                     OR ($2 = 'female' AND c.applies_to_female IS NOT FALSE))
                LIMIT 20
            """, f"%{term}%", patient_sex)
            for row in desc_rows:
                cid = row["id"]
                if cid not in desc_matches:
                    desc_matches[cid] = {"row": dict(row), "terms": set()}
                desc_matches[cid]["terms"].add(term)

        # Boost existing or add new conditions where 2+ symptoms match
        existing_by_id = {c["id"]: c for c in conditions}
        for cid, data in sorted(
            desc_matches.items(),
            key=lambda x: len(x[1]["terms"]),
            reverse=True,
        ):
            if len(data["terms"]) < 2:
                continue
            matched_terms = sorted(data["terms"])
            # Score: 0.20 per matching symptom, capped at 0.90
            desc_score = min(0.20 * len(data["terms"]), 0.90)

            if cid in existing_by_id:
                # BOOST existing condition if description score is higher
                existing = existing_by_id[cid]
                if desc_score > existing.get("adjusted_score", 0):
                    existing["adjusted_score"] = desc_score
                    existing["raw_score"] = max(existing.get("raw_score", 0), desc_score)
                    existing["symptom_groups_matched"] = max(
                        existing.get("symptom_groups_matched", 0), len(data["terms"])
                    )
                    for t in matched_terms:
                        feat = f"{t} (description boost)"
                        if feat not in existing.get("matched_features", []):
                            existing.setdefault("matched_features", []).append(feat)
            else:
                # ADD new condition
                row = data["row"]
                conditions.append({
                    "id": row["id"],
                    "stg_code": row["stg_code"],
                    "name": row["name"],
                    "chapter_name": row["chapter_name"],
                    "extraction_confidence": float(row["extraction_confidence"] or 1.0),
                    "match_count": len(data["terms"]),
                    "raw_score": desc_score,
                    "matched_features": [
                        f"{t} (description match)" for t in matched_terms
                    ],
                    "symptom_groups_matched": len(data["terms"]),
                    "adjusted_score": desc_score,
                })
                existing_ids.add(row["id"])

        # Medication boost: if patient is on medications, boost conditions those meds treat
        if medications:
            med_conditions = await get_conditions_for_medications(conn, medications)
            existing_by_id = {c["id"]: c for c in conditions}
            for mc in med_conditions:
                cid = mc["id"]
                med_name = mc["medicine_name"]
                feat = f"current medication: {med_name} (medication-based)"
                if cid in existing_by_id:
                    existing = existing_by_id[cid]
                    existing["adjusted_score"] = existing.get("adjusted_score", 0) + 0.15
                    if feat not in existing.get("matched_features", []):
                        existing.setdefault("matched_features", []).append(feat)
                elif cid not in existing_ids:
                    conditions.append({
                        "id": mc["id"],
                        "stg_code": mc["stg_code"],
                        "name": mc["name"],
                        "chapter_name": mc.get("chapter_name", ""),
                        "extraction_confidence": 1.0,
                        "match_count": 1,
                        "raw_score": 0.40,
                        "matched_features": [feat],
                        "symptom_groups_matched": 0,
                        "adjusted_score": 0.40,
                    })
                    existing_ids.add(cid)

        # Re-sort and limit
        conditions.sort(key=lambda c: (c.get("adjusted_score", 0), c.get("raw_score", 0)), reverse=True)
        conditions = conditions[:limit]

        # Filter out parent headings (e.g., 4.7) when populated children exist (4.7.1)
        conditions = await _filter_parent_headings(conn, conditions)

        # Deduplicate by condition name — keep only highest-scoring entry
        # (e.g., collapses 7 "Candidiasis, Oral (Thrush), Recurrent" into 1)
        seen_names = {}
        deduped = []
        for c in conditions:
            name = c.get("name", "")
            if name not in seen_names:
                seen_names[name] = True
                deduped.append(c)
        conditions = deduped

        red_flags = await get_red_flag_matches(conn, all_search_terms)

    return {
        "conditions": conditions,
        "red_flags_found": red_flags,
        "total_matches": len(conditions),
    }


async def handle_score_differential(tool_input: dict, pool: asyncpg.Pool) -> dict:
    """Score and rank conditions, compute acuity from RED_FLAGS + vitals."""
    condition_ids = tool_input.get("condition_ids", [])
    symptoms = [s.lower() for s in tool_input.get("symptoms", [])]
    vitals = tool_input.get("vitals") or {}

    scored = []
    async with pool.acquire() as conn:
        for cid in condition_ids:
            # Get the condition's search result data
            row = await conn.fetchrow("""
                SELECT c.id, c.stg_code, c.name, c.chapter_name
                FROM conditions c WHERE c.id = $1
            """, cid)
            if not row:
                continue

            # Get matched features for this condition from the symptom set
            matched = await conn.fetch("""
                SELECT e.canonical_name, cr.feature_type, cr.relationship_type
                FROM clinical_relationships cr
                JOIN clinical_entities e ON e.id = cr.source_entity_id
                WHERE cr.condition_id = $1
                AND e.canonical_name = ANY($2::text[])
            """, cid, symptoms)

            raw_score = 0.0
            matched_symptoms = []
            has_red_flag = False
            for m in matched:
                ft = m["feature_type"]
                if ft == "diagnostic_feature":
                    raw_score += 0.18
                elif ft == "presenting_feature":
                    raw_score += 0.12
                else:
                    raw_score += 0.08
                if m["relationship_type"] == "RED_FLAG":
                    raw_score += 0.10
                    has_red_flag = True
                matched_symptoms.append(m["canonical_name"])

            # Normalise: divide by max possible, capping feature count
            # to prevent enrichment-bloated conditions from having artificially low scores
            total_features = await conn.fetchval(
                "SELECT COUNT(*) FROM clinical_relationships WHERE condition_id = $1",
                cid
            )
            capped_features = min(total_features, 12)  # cap denominator
            max_possible = max(capped_features * 0.18, 0.18)
            confidence = min(raw_score / max_possible, 1.0)

            # Check prerequisites
            prerequisites = await get_condition_prerequisites(conn, cid)
            prereq_flags = [p["prerequisite"] for p in prerequisites]

            scored.append({
                "condition_id": cid,
                "stg_code": row["stg_code"],
                "name": row["name"],
                "chapter": row["chapter_name"],
                "confidence": round(confidence, 3),
                "raw_score": round(raw_score, 3),
                "matched_symptoms": list(set(matched_symptoms)),
                "has_red_flag": has_red_flag,
                "prerequisite_flags": prereq_flags,
            })

    scored.sort(key=lambda x: x["confidence"], reverse=True)

    # Compute acuity from red flags + vitals
    acuity = "routine"
    acuity_reasons = []
    acuity_sources = []

    # Red flag escalation
    red_flag_conditions = [s for s in scored if s["has_red_flag"]]
    if red_flag_conditions:
        acuity = "urgent"
        for rfc in red_flag_conditions:
            acuity_reasons.append(f"Red flag features for {rfc['name']}")
            acuity_sources.append(f"STG {rfc['stg_code']}")

    # Vitals escalation
    if vitals.get("systolic") and vitals["systolic"] >= 180:
        acuity = "urgent"
        acuity_reasons.append(f"Severe hypertension: BP {vitals['systolic']}/{vitals.get('diastolic', '?')}")
        acuity_sources.append("Standard: vital signs assessment")
    if vitals.get("oxygenSat") and vitals["oxygenSat"] < 92:
        acuity = "urgent"
        acuity_reasons.append(f"Low SpO2: {vitals['oxygenSat']}%")
        acuity_sources.append("Standard: vital signs assessment")
    if vitals.get("temperature") and vitals["temperature"] >= 39.0:
        if acuity != "urgent":
            acuity = "priority"
        acuity_reasons.append(f"High fever: {vitals['temperature']}°C")
        acuity_sources.append("Standard: vital signs assessment")
    if vitals.get("heartRate") and (vitals["heartRate"] > 120 or vitals["heartRate"] < 50):
        if acuity != "urgent":
            acuity = "priority"
        acuity_reasons.append(f"Abnormal heart rate: {vitals['heartRate']} bpm")
        acuity_sources.append("Standard: vital signs assessment")

    return {
        "scored_conditions": scored[:10],
        "acuity": acuity,
        "acuity_reasons": acuity_reasons,
        "acuity_sources": acuity_sources,
    }


async def handle_get_condition_detail(tool_input: dict, pool: asyncpg.Pool) -> dict:
    """Fetch full STG entry for one condition."""
    cid = tool_input.get("condition_id")
    if not cid:
        return {"error": "condition_id required"}

    async with pool.acquire() as conn:
        detail = await get_condition_detail(conn, cid)

    if not detail:
        return {"error": f"Condition {cid} not found"}

    # Parse medicines_json
    meds = detail.get("medicines_json", [])
    if isinstance(meds, str):
        meds = json.loads(meds)

    # Parse referral_criteria
    referral = detail.get("referral_criteria", "[]")
    if isinstance(referral, str):
        try:
            referral = json.loads(referral)
        except (json.JSONDecodeError, TypeError):
            referral = [referral] if referral else []

    return {
        "condition_id": detail["id"],
        "stg_code": detail["stg_code"],
        "name": detail["name"],
        "chapter": detail.get("chapter_name", ""),
        "description": detail.get("description_text", ""),
        "general_measures": detail.get("general_measures", ""),
        "medicine_treatment": detail.get("medicine_treatment", ""),
        "danger_signs": detail.get("danger_signs", ""),
        "referral_criteria": referral,
        "medicines": meds,
        "source_pages": detail.get("source_pages", []),
    }


async def handle_check_safety_flags(tool_input: dict, pool: asyncpg.Pool) -> dict:
    """Check RED_FLAG features + vitals thresholds for escalation."""
    symptoms = [s.lower() for s in tool_input.get("symptoms", [])]
    condition_ids = tool_input.get("condition_ids", [])
    vitals = tool_input.get("vitals") or {}

    triggered = []
    async with pool.acquire() as conn:
        for cid in condition_ids:
            red_flags = await get_condition_red_flags(conn, cid)
            condition_name = await conn.fetchval(
                "SELECT name FROM conditions WHERE id = $1", cid
            )
            for rf in red_flags:
                if rf["canonical_name"].lower() in symptoms:
                    triggered.append({
                        "flag": rf["canonical_name"],
                        "condition": condition_name or "Unknown",
                        "condition_id": cid,
                        "action": "URGENT_REFERRAL",
                    })

    # Vitals-based flags
    vitals_flags = []
    if vitals.get("systolic") and vitals["systolic"] >= 180:
        vitals_flags.append({"flag": "Hypertensive crisis", "value": vitals["systolic"], "action": "URGENT_REFERRAL"})
    if vitals.get("oxygenSat") and vitals["oxygenSat"] < 90:
        vitals_flags.append({"flag": "Critical hypoxia", "value": vitals["oxygenSat"], "action": "URGENT_REFERRAL"})
    if vitals.get("temperature") and vitals["temperature"] >= 40.0:
        vitals_flags.append({"flag": "Hyperpyrexia", "value": vitals["temperature"], "action": "URGENT_REFERRAL"})
    if vitals.get("respiratoryRate") and vitals["respiratoryRate"] >= 30:
        vitals_flags.append({"flag": "Tachypnoea", "value": vitals["respiratoryRate"], "action": "URGENT_REFERRAL"})

    return {
        "red_flags_triggered": triggered,
        "vitals_flags": vitals_flags,
        "requires_escalation": len(triggered) > 0 or len(vitals_flags) > 0,
        "escalation_level": "urgent" if (triggered or vitals_flags) else "none",
    }


# ── Dispatch map ─────────────────────────────────────────────────────────────

TOOL_HANDLERS = {
    "extract_symptoms": handle_extract_symptoms,
    "expand_synonyms": handle_expand_synonyms,
    "search_conditions": handle_search_conditions,
    "score_differential": handle_score_differential,
    "get_condition_detail": handle_get_condition_detail,
    "check_safety_flags": handle_check_safety_flags,
}
