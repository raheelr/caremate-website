"""
Clinical Data Cache
====================

Loads all clinical knowledge data from the database at startup into an
in-memory dataclass. All runtime code reads from this cache — zero
per-request DB queries for clinical rules, drug classes, discriminators, etc.

Architecture: Server start → load_clinical_cache(conn) → ClinicalDataCache
              stored in app.state.clinical_cache, injected into agent modules.

Load time: ~100ms for ~400 rows. Memory: ~50KB.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import asyncpg

logger = logging.getLogger(__name__)


@dataclass
class ClinicalDataCache:
    """In-memory cache of all clinical knowledge data from the database.

    All fields are populated by load_clinical_cache() at server startup.
    Runtime code reads from these dicts/sets with O(1) lookups.
    """

    # ── Drug Classes ──
    # class_name → set of drug names
    drug_classes: dict[str, set[str]] = field(default_factory=dict)
    # class_name → class_id (for interaction rule resolution)
    drug_class_ids: dict[str, int] = field(default_factory=dict)
    # class_id → class_name (reverse)
    drug_class_names: dict[int, str] = field(default_factory=dict)

    # ── Allergy Cross-Reactivity ──
    # allergy_keyword → set of drug names (flattened from keyword→class→members)
    allergy_drug_map: dict[str, set[str]] = field(default_factory=dict)

    # ── Drug-Drug Interactions ──
    # List of (group_a: set[str], group_b: set[str], severity: str, message: str)
    interaction_rules: list[tuple[set[str], set[str], str, str]] = field(default_factory=list)

    # ── Pregnancy-Unsafe Drugs ──
    # drug_name → reason string
    pregnancy_unsafe: dict[str, str] = field(default_factory=dict)

    # ── Lab Result Patterns ──
    # List of lab pattern dicts (same structure as scoring_config.LAB_RESULT_PATTERNS)
    lab_result_patterns: list[dict] = field(default_factory=list)

    # ── Clinical Discriminators (SATS) ──
    # Keyed by (population, colour) → list of phrases
    adult_emergency_signs: list[str] = field(default_factory=list)
    adult_very_urgent_signs: list[str] = field(default_factory=list)
    adult_urgent_signs: list[str] = field(default_factory=list)
    paediatric_emergency_signs: list[str] = field(default_factory=list)

    # ── Clinical Opportunity Rules ──
    # List of rule dicts (same structure as opportunities.RULES)
    opportunity_rules: list[dict] = field(default_factory=list)

    # ── Keyword Sets ──
    # set_name → set of keywords
    keyword_sets: dict[str, set[str]] = field(default_factory=dict)

    # ── Condition-Level Data ──
    # stg_code → prevalence tier ("high" or "moderate")
    prevalence_tier: dict[str, str] = field(default_factory=dict)
    # Set of stg_codes that require pregnancy
    pregnancy_required_codes: set[str] = field(default_factory=set)
    # Set of chapter numbers that are non-disease
    non_disease_chapters: set[int] = field(default_factory=set)

    # ── Clinical Reasoning Rules ──
    # condition_stg_code → list of rule dicts (sorted by discriminating_power DESC)
    reasoning_rules: dict[str, list[dict]] = field(default_factory=dict)

    # ── Convenience Accessors (populated from drug_classes) ──
    # Named drug sets matching the old Python constants
    ace_inhibitors: set[str] = field(default_factory=set)
    nsaids: set[str] = field(default_factory=set)
    cns_depressants: set[str] = field(default_factory=set)
    cyp450_inducers: set[str] = field(default_factory=set)
    oral_contraceptives: set[str] = field(default_factory=set)


async def load_clinical_cache(conn: asyncpg.Connection) -> ClinicalDataCache:
    """Load all clinical data from DB into an in-memory cache.

    Called once at server startup. Returns populated ClinicalDataCache.
    """
    cache = ClinicalDataCache()

    # 1. Drug classes + members
    classes = await conn.fetch("SELECT id, class_name FROM drug_classes")
    for c in classes:
        cache.drug_class_ids[c["class_name"]] = c["id"]
        cache.drug_class_names[c["id"]] = c["class_name"]

    members = await conn.fetch("""
        SELECT dc.class_name, dcm.drug_name
        FROM drug_class_members dcm
        JOIN drug_classes dc ON dc.id = dcm.drug_class_id
    """)
    for m in members:
        cache.drug_classes.setdefault(m["class_name"], set()).add(m["drug_name"])

    # Populate named convenience sets
    cache.ace_inhibitors = cache.drug_classes.get("ace_inhibitors", set())
    cache.nsaids = cache.drug_classes.get("nsaids", set())
    cache.cns_depressants = cache.drug_classes.get("cns_depressants", set())
    cache.cyp450_inducers = cache.drug_classes.get("cyp450_inducers", set())
    cache.oral_contraceptives = cache.drug_classes.get("oral_contraceptives", set())

    # 2. Allergy cross-reactivity (keyword → drug class → members → flat set)
    acr_rows = await conn.fetch("""
        SELECT acr.allergy_keyword, dc.class_name
        FROM allergy_cross_reactivity acr
        JOIN drug_classes dc ON dc.id = acr.drug_class_id
    """)
    for row in acr_rows:
        drug_set = cache.drug_classes.get(row["class_name"], set())
        cache.allergy_drug_map[row["allergy_keyword"]] = drug_set

    # 3. Interaction rules (resolve class_id → drug set)
    ir_rows = await conn.fetch("""
        SELECT group_a_class_id, group_a_drug, group_b_class_id, group_b_drug,
               severity, message
        FROM drug_interaction_rules
        WHERE active = TRUE
    """)
    for row in ir_rows:
        # Resolve group A
        if row["group_a_class_id"]:
            class_name_a = cache.drug_class_names.get(row["group_a_class_id"])
            group_a = cache.drug_classes.get(class_name_a, set()) if class_name_a else set()
        else:
            group_a = set(d.strip() for d in (row["group_a_drug"] or "").split(",") if d.strip())

        # Resolve group B
        if row["group_b_class_id"]:
            class_name_b = cache.drug_class_names.get(row["group_b_class_id"])
            group_b = cache.drug_classes.get(class_name_b, set()) if class_name_b else set()
        else:
            group_b = set(d.strip() for d in (row["group_b_drug"] or "").split(",") if d.strip())

        cache.interaction_rules.append((group_a, group_b, row["severity"], row["message"]))

    # 4. Pregnancy-unsafe drugs
    pu_rows = await conn.fetch(
        "SELECT drug_name, reason FROM pregnancy_unsafe_rules WHERE active = TRUE"
    )
    for row in pu_rows:
        cache.pregnancy_unsafe[row["drug_name"]] = row["reason"]

    # 5. Lab result patterns
    lab_rows = await conn.fetch(
        "SELECT * FROM lab_result_patterns WHERE active = TRUE ORDER BY id"
    )
    for row in lab_rows:
        cache.lab_result_patterns.append({
            "id": row["lab_id"],
            "patterns": list(row["text_patterns"]),
            "structured_names": list(row["structured_names"]),
            "positive_keywords": list(row["positive_keywords"]) if row["positive_keywords"] else None,
            "numeric_threshold": row["numeric_threshold"],
            "threshold_direction": row["threshold_direction"],
            "condition_codes": json.loads(row["condition_codes"]) if isinstance(row["condition_codes"], str) else row["condition_codes"],
            "force_rank_one": row["force_rank_one"],
            "score_boost": row["score_boost"],
            "add_symptoms": list(row["add_symptoms"]) if row["add_symptoms"] else None,
            "marker_label": row["display_label"],
        })

    # 6. Clinical discriminators
    disc_rows = await conn.fetch(
        "SELECT phrase, acuity_colour, population FROM clinical_discriminators WHERE active = TRUE"
    )
    for row in disc_rows:
        phrase = row["phrase"]
        colour = row["acuity_colour"]
        pop = row["population"]

        if pop == "adult":
            if colour == "red":
                cache.adult_emergency_signs.append(phrase)
            elif colour == "orange":
                cache.adult_very_urgent_signs.append(phrase)
            elif colour == "yellow":
                cache.adult_urgent_signs.append(phrase)
        elif pop == "paediatric":
            if colour == "red":
                cache.paediatric_emergency_signs.append(phrase)

    # 7. Opportunity rules
    opp_rows = await conn.fetch(
        "SELECT * FROM clinical_opportunity_rules WHERE active = TRUE ORDER BY id"
    )
    for row in opp_rows:
        rule = {
            "id": row["rule_id"],
            "type": row["rule_type"],
            "title": row["title"],
            "description": row["description"],
            "action_label": row["action_label"],
            "priority": row["priority"],
            "stg_reference": row["stg_reference"] or "",
        }
        # Optional fields
        if row["min_age"] is not None:
            rule["min_age"] = row["min_age"]
        if row["max_age"] is not None:
            rule["max_age"] = row["max_age"]
        if row["sex"]:
            rule["sex"] = row["sex"]
        if row["exclude_pregnancy"]:
            rule["exclude_pregnancy"] = True
        if row["require_pregnancy"]:
            rule["require_pregnancy"] = True
        if row["dx_contains"]:
            rule["dx_contains"] = list(row["dx_contains"])
        if row["exclude_dx_contains"]:
            rule["exclude_dx_contains"] = list(row["exclude_dx_contains"])
        if row["dx_stg_prefix"]:
            rule["dx_stg_prefix"] = list(row["dx_stg_prefix"])
        if row["vitals_check"]:
            rule["vitals_check"] = row["vitals_check"]
        if row["suppress_if_dx_contains"]:
            rule["suppress_if_dx_contains"] = list(row["suppress_if_dx_contains"])
        if row["med_check"]:
            rule["med_check"] = row["med_check"]
        if row["dx_contains_any"]:
            rule["dx_contains_any"] = list(row["dx_contains_any"])
        if row["require_symptom_or_dx"]:
            rsod = row["require_symptom_or_dx"]
            rule["require_symptom_or_dx"] = json.loads(rsod) if isinstance(rsod, str) else rsod

        cache.opportunity_rules.append(rule)

    # 8. Keyword sets
    kw_rows = await conn.fetch(
        "SELECT set_name, keyword FROM clinical_keyword_sets WHERE active = TRUE"
    )
    for row in kw_rows:
        cache.keyword_sets.setdefault(row["set_name"], set()).add(row["keyword"])

    # 9. Condition-level data
    # Prevalence tier: build from keyword_sets (canonical source, includes parent codes)
    for set_name in ("prevalence_high", "prevalence_moderate"):
        tier = set_name.replace("prevalence_", "")
        codes = cache.keyword_sets.get(set_name, set())
        for code in codes:
            cache.prevalence_tier[code] = tier

    preg_rows = await conn.fetch(
        "SELECT stg_code FROM conditions WHERE pregnancy_required = TRUE"
    )
    cache.pregnancy_required_codes = {row["stg_code"] for row in preg_rows}

    nd_rows = await conn.fetch(
        "SELECT DISTINCT substring(stg_code from '^(\\d+)')::int as chapter_num FROM conditions WHERE is_non_disease = TRUE"
    )
    cache.non_disease_chapters = {row["chapter_num"] for row in nd_rows}

    # 10. Clinical reasoning rules (if table exists)
    try:
        rr_rows = await conn.fetch("""
            SELECT condition_stg_code, condition_name, rule_type, rule_data,
                   assessment_question, question_type, question_options,
                   discriminating_power, rules_out_codes, is_red_flag,
                   applies_to_age_min, applies_to_age_max, applies_to_sex,
                   source_file, source_tag
            FROM clinical_reasoning_rules
            WHERE active = TRUE
            ORDER BY condition_stg_code, discriminating_power DESC
        """)
        for row in rr_rows:
            code = row["condition_stg_code"]
            rule = {
                "condition_stg_code": code,
                "condition_name": row["condition_name"],
                "rule_type": row["rule_type"],
                "rule_data": json.loads(row["rule_data"]) if isinstance(row["rule_data"], str) else row["rule_data"],
                "assessment_question": row["assessment_question"],
                "question_type": row["question_type"] or "yes_no",
                "question_options": list(row["question_options"]) if row["question_options"] else None,
                "discriminating_power": float(row["discriminating_power"] or 0.5),
                "rules_out_codes": list(row["rules_out_codes"]) if row["rules_out_codes"] else [],
                "is_red_flag": bool(row["is_red_flag"]),
                "applies_to_age_min": row["applies_to_age_min"],
                "applies_to_age_max": row["applies_to_age_max"],
                "applies_to_sex": row["applies_to_sex"],
                "source_file": row["source_file"],
                "source_tag": row["source_tag"],
            }
            cache.reasoning_rules.setdefault(code, []).append(rule)
        rr_count = len(rr_rows)
    except asyncpg.UndefinedTableError:
        logger.info("clinical_reasoning_rules table not found — skipping (run migration 007)")
        rr_count = 0

    # Log summary
    logger.info(
        f"Clinical cache loaded: "
        f"{len(cache.drug_classes)} drug classes, "
        f"{sum(len(v) for v in cache.drug_classes.values())} members, "
        f"{len(cache.allergy_drug_map)} allergy keywords, "
        f"{len(cache.interaction_rules)} interaction rules, "
        f"{len(cache.pregnancy_unsafe)} pregnancy-unsafe drugs, "
        f"{len(cache.lab_result_patterns)} lab patterns, "
        f"{len(cache.adult_emergency_signs) + len(cache.adult_very_urgent_signs) + len(cache.adult_urgent_signs) + len(cache.paediatric_emergency_signs)} discriminators, "
        f"{len(cache.opportunity_rules)} opportunity rules, "
        f"{sum(len(v) for v in cache.keyword_sets.values())} keywords, "
        f"{rr_count} reasoning rules ({len(cache.reasoning_rules)} conditions)"
    )

    return cache
