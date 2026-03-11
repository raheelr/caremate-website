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
    "17.3.3": "high",     # Pneumonia (parent)
    "17.3.4": "high",     # Pneumonia (child)
    "17.3.4.1": "high",   # Pneumonia (specific)
    "17.3.1": "high",     # Influenza
    "17.1.5": "high",     # COPD
    "19.2": "high",       # Common cold
    "19.6": "high",       # Tonsillitis/Pharyngitis
    # Cardiovascular
    "4.5": "high",        # CCF
    "4.6.1": "high",      # CCF in adults
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
    "8.1": "moderate",    # CKD
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

# ───────────────────────────────────────────────────────────────
# Pregnancy-Required Conditions
# ───────────────────────────────────────────────────────────────
# Conditions that can ONLY be diagnosed if the patient is currently
# pregnant (or recently postpartum). When the patient explicitly
# reports "not pregnant", these conditions receive a near-zero
# penalty (0.05x) to effectively exclude them from the differential.
#
# This is a deterministic clinical logic filter — not relying on the
# LLM to figure out that "Threatened Miscarriage" doesn't apply to
# a non-pregnant 53-year-old.
PREGNANCY_REQUIRED_CODES = {
    # Chapter 6 — Obstetrics (STG Primary)
    "6.1.1",     # Miscarriage
    "6.2.1",     # Threatened Miscarriage
    "6.2.2",     # Miscarriage - Medical Management
    "6.3",       # Termination of Pregnancy
    "6.3.1",     # Termination of Pregnancy (specific)
    "6.4",       # Antenatal Supplements
    "6.4.2",     # Hypertensive Disorders in Pregnancy
    "6.4.2.1",   # Chronic HTN in Pregnancy
    "6.4.2.4",   # Pre-eclampsia
    "6.4.2.5",   # Eclampsia
    "6.4.3",     # Anaemia in Pregnancy
    "6.4.4",     # Syphilis in Pregnancy
    "6.4.5.2",   # Cystitis in Pregnancy
    "6.4.7",     # Preterm Labour
    "6.4.7.2",   # PPROM
    "6.4.7.3",   # PPROM (variant)
    "6.5",       # Intrapartum Care / Labour Management
    "6.6.4",     # Perinatal Hep B Transmission
    "6.7.1",     # Postpartum Haemorrhage (PPH)
    "6.7.2",     # Puerperal Sepsis
    "6.8",       # HIV in Pregnancy
    "6.9",       # Perinatal Depression/Anxiety
    "6.9.2",     # Bipolar Disorder Perinatal
    "6.11",      # Ectopic Pregnancy
    # Referral-only O&G
    "REF.OG.1",  # Foetal Distress
    "REF.OG.2",  # Placental Abruption
    "REF.OG.3",  # Placenta Praevia
    "REF.OG.5",  # Molar Pregnancy
    "REF.OG.8",  # Hyperemesis Gravidarum
    "REF.OG.9",  # Chorioamnionitis
}
PREGNANCY_REQUIRED_PENALTY = 0.05  # Near-zero: effectively removes from differential

# ───────────────────────────────────────────────────────────────
# Gynaecological Complaint Keywords
# ───────────────────────────────────────────────────────────────
# Used to trigger deterministic menopausal status questions for
# women 40+ presenting with gynaecological symptoms.
GYNAE_COMPLAINT_KEYWORDS = {
    "vaginal bleeding", "vaginal discharge", "bleeding",
    "menstrual", "period", "menopause", "spotting",
    "pelvic pain", "lower abdominal pain", "dysmenorrhoea",
    "amenorrhoea", "irregular periods",
}

# ───────────────────────────────────────────────────────────────
# Lab Result → Condition Mapping
# ───────────────────────────────────────────────────────────────
# Deterministic mapping of confirmed lab results to STG conditions.
# Follows the same injection pattern as vitals → conditions.
#
# Two input paths feed into the same injection function:
#   1. Text parsing: regex scan on complaint text (nurses typing "HIV reactive")
#   2. Structured input: lab_results API parameter (EMR integration)
#
# Adding a new lab result = adding one dict entry. No code changes needed.
# ───────────────────────────────────────────────────────────────
# Duration-Aware Scoring
# ───────────────────────────────────────────────────────────────
# Maps UI onset strings → duration category, then applies per-condition
# multipliers to adjusted_score. Two mechanisms:
#   1. Penalty (mult < 1.0) for self-limiting conditions when duration
#      exceeds their natural course (e.g. common cold lasting > 1 week)
#   2. Boost (mult > 1.0) for chronic/infectious conditions when duration
#      matches their clinical profile (e.g. TB with > 2 weeks cough)

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
# Profiles are stored in conditions.duration_profile column (DB-driven).
# NULL profile = no modifier applied.
DURATION_PROFILE_MULTIPLIERS = {
    "acute_self_limiting": {"acute": 1.0, "subacute": 0.5, "chronic": 0.3},
    "acute_severe":        {"acute": 1.0, "subacute": 1.0, "chronic": 1.0},
    "subacute_infectious": {"acute": 0.8, "subacute": 1.5, "chronic": 1.8},
    "chronic":             {"acute": 0.8, "subacute": 1.1, "chronic": 1.3},
    "variable":            {"acute": 1.0, "subacute": 1.0, "chronic": 1.0},
}

# ───────────────────────────────────────────────────────────────
# Lab Result → Condition Mapping
# ───────────────────────────────────────────────────────────────
LAB_RESULT_PATTERNS = [
    {
        "id": "hiv_positive",
        "patterns": [
            r"hiv\s+(screening\s+)?reactive",
            r"hiv\s+(test\s+)?positive",
            r"hiv\s+confirmed",
            r"hiv\+",
            r"hiv\s+rapid\s+test\s+positive",
            r"hiv\s+elisa\s+positive",
        ],
        "structured_names": ["hiv", "hiv antibody", "hiv elisa", "hiv rapid test"],
        "positive_keywords": ["positive", "reactive", "confirmed", "detected"],
        "condition_codes": {"pregnant": "6.8", "default": "11.1"},
        "force_rank_one": True,
        "score_boost": 2.0,
        "add_symptoms": ["hiv positive"],
        "marker_label": "HIV screening reactive",
    },
    {
        "id": "syphilis_positive",
        "patterns": [
            r"rpr\s+(test\s+)?positive",
            r"rpr\s+reactive",
            r"vdrl\s+(test\s+)?positive",
            r"vdrl\s+reactive",
            r"syphilis\s+(test\s+)?positive",
            r"syphilis\s+confirmed",
        ],
        "structured_names": ["rpr", "vdrl", "syphilis", "syphilis serology"],
        "positive_keywords": ["positive", "reactive", "detected"],
        "condition_codes": {"pregnant": "6.4.4", "default": "12.5"},
        "force_rank_one": True,
        "score_boost": 2.0,
        "add_symptoms": ["syphilis positive"],
        "marker_label": "Syphilis screening positive",
    },
    {
        "id": "tb_positive",
        "patterns": [
            r"genexpert\s+positive",
            r"xpert\s+positive",
            r"sputum\s+(smear\s+)?positive",
            r"tb\s+(test\s+)?positive",
            r"afb\s+(smear\s+)?positive",
        ],
        "structured_names": ["genexpert", "xpert", "sputum", "afb", "tb test"],
        "positive_keywords": ["positive", "detected", "confirmed"],
        "condition_codes": {"default": "17.4.1"},
        "force_rank_one": True,
        "score_boost": 2.0,
        "add_symptoms": ["tuberculosis confirmed", "sputum positive"],
        "marker_label": "TB test positive",
    },
    {
        "id": "malaria_positive",
        "patterns": [
            r"malaria\s+(rdt|rapid\s+test|smear)\s+positive",
            r"malaria\s+(test\s+)?positive",
        ],
        "structured_names": ["malaria rdt", "malaria smear", "malaria rapid test"],
        "positive_keywords": ["positive", "detected"],
        "condition_codes": {"default": "10.7"},
        "force_rank_one": True,
        "score_boost": 2.0,
        "add_symptoms": ["malaria positive"],
        "marker_label": "Malaria test positive",
    },
    {
        "id": "diabetes_confirmed",
        "patterns": [
            r"hba1c\s+(elevated|raised|high|abnormal)",
            r"diabetes\s+confirmed",
            r"diabetic",
        ],
        "structured_names": ["hba1c", "fasting glucose", "random glucose", "ogtt"],
        "positive_keywords": ["elevated", "raised", "high", "abnormal", "confirmed"],
        "condition_codes": {"default": "9.2"},
        "force_rank_one": True,
        "score_boost": 2.0,
        "add_symptoms": ["diabetes confirmed", "hyperglycaemia"],
        "marker_label": "Diabetes confirmed",
    },
]
