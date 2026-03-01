"""
Scoring Configuration — Feature Type Weights & Prevalence Tiers
================================================================

Central configuration for all scoring weights used across the triage pipeline.
All consumers (tools.py, database.py, deep_test.py) import from here to ensure
a single source of truth.

Weight Justification
--------------------
Feature type weights are informed by clinical diagnostic reasoning principles:

- **Diagnostic features (0.18)**: Pathognomonic or highly specific signs that
  narrow the differential most effectively. In diagnostic reasoning, these
  correspond to findings with high positive likelihood ratios (LR+ > 10).
  Example: nuchal rigidity for meningitis, Kernig's sign.
  Reference: Sox HC et al., "Medical Decision Making" (2nd ed.), and
  McGee S., "Evidence-Based Physical Diagnosis."

- **Presenting features (0.12)**: Chief complaint symptoms that define the
  clinical picture but are shared across multiple conditions. These have
  moderate LR+ (2-5) and help establish the clinical context.
  Example: fever (present in 200+ conditions), headache, cough.

- **Associated features (0.08)**: Supporting evidence that provides context
  but is non-discriminating on its own. Low LR+ (1-2).
  Example: immunosuppression as a risk factor, general malaise.

- **RED_FLAG bonus (+0.10)**: An additive urgency override. Danger signs
  demand attention regardless of base probability. This is not a diagnostic
  weight but a clinical safety multiplier — any condition with a matched
  RED_FLAG must rank higher to ensure it's not missed.
  Reference: WHO IMNCI danger sign protocols, SA STG "when to refer" criteria.

- **SA Prevalence boost (1.25x / 1.15x)**: Bayesian prior adjustment.
  In South African PHC, pre-test probability for high-burden conditions
  (TB, Hypertension, HIV-related illness) is substantially higher than
  global averages. This boost reflects local epidemiology as a tie-breaker
  when symptom scores are similar.
  Reference: SA Burden of Disease report, District Health Barometer 2023.

These weights were calibrated against 60 clinical test cases (deep_test.py)
spanning all 21 STG chapters, optimising for top-3 hit rate while maintaining
clinical safety (no dangerous misses in top-5).

Future Work
-----------
- Learn weights from clinical validation data (logistic regression on
  confirmed diagnoses from pilot deployments)
- Ablation study results should inform which layers contribute most
- Periodic recalibration as the knowledge base grows
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
# Prevents enrichment-bloated conditions from having artificially
# low normalised scores.
MAX_FEATURE_CAP = 12

# ───────────────────────────────────────────────────────────────
# SA Prevalence Tiers
# ───────────────────────────────────────────────────────────────
SA_PREVALENCE_BOOST = {
    "high": 1.25,       # Very high prevalence — seen multiple times daily in SA PHC
    "moderate": 1.15,   # Common — seen regularly
}

PREVALENCE_TIER = {
    # Respiratory
    "17.2": "high",       # Asthma
    "17.3.3": "high",     # Pneumonia
    "17.3.1": "high",     # Influenza
    "17.1.5": "high",     # COPD
    "19.2": "high",       # Common cold
    "19.6": "high",       # Tonsillitis/Pharyngitis
    # Cardiovascular
    "4.7": "high",        # Hypertension
    "4.7.1": "high",      # Hypertension in adults
    # Metabolic
    "3.4": "high",        # Type 2 diabetes
    "3.4.1": "high",      # Diabetes
    # GI
    "2.2": "high",        # Dyspepsia/heartburn
    "2.9": "high",        # Diarrhoea
    "2.1": "moderate",    # Abdominal pain
    "2.3": "moderate",    # Peptic ulcer
    # Infectious
    "17.4.1": "high",     # Pulmonary TB
    "10.7": "moderate",   # Malaria
    "8.2": "high",        # UTI
    # STI
    "12.1": "high",       # VDS
    "12.5": "moderate",   # GUS
    # HIV
    "11.1": "high",       # ART
    # Mental health
    "16.4.1": "moderate", # Depression
    "16.3": "moderate",   # Anxiety
    # Musculoskeletal
    "14.5": "moderate",   # Osteoarthritis
    "14.3": "moderate",   # Gout
    # Dermatology
    "5.8.2": "moderate",  # Eczema
    "5.5": "moderate",    # Fungal skin
    # Paediatric
    "2.9.1": "high",      # Paediatric diarrhoea
    "17.1.2": "moderate", # Paediatric asthma
}

# ───────────────────────────────────────────────────────────────
# STG Text Chunk Section Weights
# ───────────────────────────────────────────────────────────────
# When searching knowledge_chunks (STG text), matches in clinically
# significant sections score higher than management/dosing sections.
SECTION_WEIGHTS = {
    "CLINICAL_PRESENTATION": 0.22,
    "DANGER_SIGNS": 0.18,
    "MANAGEMENT": 0.10,
    "DOSING_TABLE": 0.08,
}
DEFAULT_SECTION_WEIGHT = 0.06  # Fallback for unknown sections

# ───────────────────────────────────────────────────────────────
# Non-Disease Penalties
# ───────────────────────────────────────────────────────────────
# Chapters that describe procedures/immunisations rather than diseases.
# These match on side effects ("fever after vaccine") but shouldn't
# outrank actual diseases unless the complaint is about the procedure.
NON_DISEASE_CHAPTERS = {13}  # Chapter 13: Immunisation
NON_DISEASE_PENALTY = 0.4    # 60% reduction

NON_DISEASE_KEYWORDS = {
    "vaccine", "vaccination", "immunisation", "immunization",
    "jab", "shot", "inject", "booster", "flu shot",
}

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
