"""
Scoring Configuration — Algorithm Constants
=============================================

Central configuration for scoring weights and algorithm parameters used
across the triage pipeline. All consumers import from here.

NOTE: Clinical data (drug classes, prevalence tiers, lab patterns,
discriminators, opportunity rules, etc.) has been moved to the database
and is loaded via ClinicalDataCache at startup. This file now contains
ONLY algorithm constants, thresholds, and penalties.

Weight Justification
--------------------
Feature type weights are informed by clinical diagnostic reasoning principles:

- **Diagnostic features (0.18)**: Pathognomonic or highly specific signs that
  narrow the differential most effectively. In diagnostic reasoning, these
  correspond to findings with high positive likelihood ratios (LR+ > 10).
  Example: nuchal rigidity for meningitis, Kernig's sign.

- **Presenting features (0.12)**: Chief complaint symptoms that define the
  clinical picture but are shared across multiple conditions.
  Example: fever (present in 200+ conditions), headache, cough.

- **Associated features (0.08)**: Supporting evidence that provides context
  but is non-discriminating on its own. Low LR+ (1-2).
  Example: immunosuppression as a risk factor, general malaise.

- **RED_FLAG bonus (+0.10)**: An additive urgency override. Danger signs
  demand attention regardless of base probability.

- **SA Prevalence boost (1.25x / 1.15x)**: Bayesian prior adjustment.
  Prevalence tier mappings are now in the database (conditions.prevalence_tier).

These weights were calibrated against 92 clinical test cases (deep_test.py).
"""

# ───────────────────────────────────────────────────────────────
# Feature Type Weights (for clinical_relationships scoring)
# ───────────────────────────────────────────────────────────────
FEATURE_WEIGHTS = {
    "diagnostic_feature": 0.18,
    "presenting_feature": 0.12,
    "associated_feature": 0.08,
}
DEFAULT_FEATURE_WEIGHT = 0.08  # Fallback for unknown feature types
RED_FLAG_BONUS = 0.10

# Maximum features used as denominator when normalising scores.
MAX_FEATURE_CAP = 12

# ───────────────────────────────────────────────────────────────
# SA Prevalence Boost Multipliers
# ───────────────────────────────────────────────────────────────
# The tier assignments (code → "high"/"moderate") are now in the DB.
# These multipliers are algorithm constants.
SA_PREVALENCE_BOOST = {
    "high": 1.25,       # Very high prevalence — seen multiple times daily in SA PHC
    "moderate": 1.15,   # Common — seen regularly
}

# ───────────────────────────────────────────────────────────────
# STG Text Chunk Section Weights
# ───────────────────────────────────────────────────────────────
SECTION_WEIGHTS = {
    "CLINICAL_PRESENTATION": 0.22,
    "DANGER_SIGNS": 0.18,
    "MANAGEMENT": 0.10,
    "DOSING_TABLE": 0.08,
}
DEFAULT_SECTION_WEIGHT = 0.06

# ───────────────────────────────────────────────────────────────
# Non-Disease Penalty
# ───────────────────────────────────────────────────────────────
# The chapter list and keywords are now in DB (conditions.is_non_disease
# + clinical_keyword_sets). This is the algorithm multiplier only.
NON_DISEASE_PENALTY = 0.4    # 60% reduction

# ───────────────────────────────────────────────────────────────
# Condition Name Match (Fallback)
# ───────────────────────────────────────────────────────────────
NAME_MATCH_SCORE_PER_TERM = 0.40
NAME_MATCH_SCORE_CAP = 0.90

# ───────────────────────────────────────────────────────────────
# Prerequisite Penalty
# ───────────────────────────────────────────────────────────────
PREREQUISITE_PENALTY = 0.5  # 50% reduction for unconfirmed prerequisites

# ───────────────────────────────────────────────────────────────
# Multi-Method & Multi-Group Boosts
# ───────────────────────────────────────────────────────────────
MULTI_METHOD_BOOST = 1.3  # Bonus if condition found by 2+ search methods

# Vector search thresholds
VECTOR_SIMILARITY_THRESHOLD = 0.65
VECTOR_SCORE_RANGE = 0.35        # (1.0 - threshold)
VECTOR_MAX_CONTRIBUTION = 0.70   # Max normalised vector score
VECTOR_MIN_CONTRIBUTION = 0.15   # Min normalised vector score

# ───────────────────────────────────────────────────────────────
# Pregnancy-Required Penalty
# ───────────────────────────────────────────────────────────────
# The code set is now in DB (conditions.pregnancy_required).
# This is the algorithm multiplier only.
PREGNANCY_REQUIRED_PENALTY = 0.05  # Near-zero: effectively removes from differential
PREGNANCY_CONTEXT_BOOST = 1.6     # Boost O&G conditions when patient is confirmed pregnant

# ───────────────────────────────────────────────────────────────
# Paediatric Context Boost
# ───────────────────────────────────────────────────────────────
# When patient_age < PAEDIATRIC_AGE_THRESHOLD, conditions with
# paediatric name markers (e.g. "Diarrhea, Acute (Paediatric)")
# are boosted so they outrank the generic/adult variant.
PAEDIATRIC_CONTEXT_BOOST = 1.5    # Boost paediatric-specific conditions for child patients
PAEDIATRIC_AGE_THRESHOLD = 13     # Age below which boost activates

# ───────────────────────────────────────────────────────────────
# Duration-Aware Scoring
# ───────────────────────────────────────────────────────────────
DURATION_CATEGORIES = {
    "sudden onset": "acute",
    "< 24 hours": "acute",
    "1-3 days": "acute",
    "1–3 days": "acute",       # em dash variant (frontend)
    "4-7 days": "acute",
    "4–7 days": "acute",       # em dash variant (frontend)
    "> 1 week": "subacute",
    "> 2 weeks": "subacute",
    "> 1 month": "chronic",
}

# {duration_profile: {duration_category: multiplier}}
DURATION_PROFILE_MULTIPLIERS = {
    "acute_self_limiting": {"acute": 1.0, "subacute": 0.5, "chronic": 0.3},
    "acute_severe":        {"acute": 1.0, "subacute": 1.0, "chronic": 1.0},
    "subacute_infectious": {"acute": 0.8, "subacute": 1.5, "chronic": 1.8},
    "chronic":             {"acute": 0.8, "subacute": 1.1, "chronic": 1.3},
    "variable":            {"acute": 1.0, "subacute": 1.0, "chronic": 1.0},
}

# ───────────────────────────────────────────────────────────────
# Match Quality Thresholds (Confidence Floor)
# ───────────────────────────────────────────────────────────────
STRONG_MATCH_THRESHOLD = 0.40
PARTIAL_MATCH_THRESHOLD = 0.20

NO_CLEAR_MATCH_WARNING = (
    "No strong STG match found for this presentation. "
    "Suggestions below are approximate matches."
)
PARTIAL_MATCH_WARNING = (
    "STG partially covers this presentation. "
    "Suggestions below may need clinical judgement."
)

# ───────────────────────────────────────────────────────────────
# Discriminating Power Scoring (from reasoning rules)
# ───────────────────────────────────────────────────────────────
# When enabled, feature scores are weighted by their discriminating
# power from extracted STG rules. E.g., "neck stiffness" for
# meningitis gets 0.18 × 0.92 vs "fever" gets 0.18 × 0.4.
#
# Default OFF — must run deep test to verify before enabling.
USE_DISCRIMINATING_POWER = False
