-- ============================================================
-- Phase II Clinician Survey — Vignettes + Responses
-- ============================================================
-- Supports the "reasonable doctor norm" benchmark:
-- Same standardised vignettes given to clinicians AND CareMate,
-- then compared for concordance.

-- Standardised clinical vignettes (created by Tasleem)
CREATE TABLE IF NOT EXISTS clinical_vignettes (
    id                  SERIAL PRIMARY KEY,
    vignette_code       TEXT NOT NULL UNIQUE,       -- e.g. "PII-001"
    title               TEXT NOT NULL,              -- e.g. "58M chest pain on exertion"
    domain              TEXT,                       -- e.g. "Cardiovascular", "Respiratory"

    -- Patient presentation (same fields as AnalyzeRequest)
    complaint           TEXT NOT NULL,              -- chief complaint text
    patient_age         INTEGER,
    patient_sex         TEXT,                       -- "male" | "female"
    pregnancy_status    TEXT,                       -- "pregnant" | "not_pregnant" | "unknown"
    vitals              JSONB DEFAULT '{}',         -- {systolic, diastolic, heartRate, temperature, respiratoryRate, oxygenSat}
    core_history        JSONB DEFAULT '{}',         -- {onset, recurrence, medications}
    additional_info     TEXT,                       -- extra clinical context

    -- Expected answers (for automated scoring)
    expected_conditions JSONB DEFAULT '[]',         -- [{condition_name, stg_code, must_include: bool}]
    expected_acuity     TEXT,                       -- "routine" | "priority" | "urgent"
    expected_sats_colour TEXT,                      -- "green" | "yellow" | "orange" | "red"

    -- Metadata
    difficulty          TEXT DEFAULT 'medium',      -- "easy" | "medium" | "hard"
    active              BOOLEAN DEFAULT TRUE,
    created_by          TEXT,                       -- "Tasleem" etc.
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Clinician AND CareMate responses to vignettes
CREATE TABLE IF NOT EXISTS vignette_responses (
    id                      SERIAL PRIMARY KEY,
    vignette_id             INTEGER NOT NULL REFERENCES clinical_vignettes(id) ON DELETE CASCADE,

    -- Who responded
    respondent_type         TEXT NOT NULL,           -- "clinician" | "caremate"
    respondent_name         TEXT,                    -- clinician name or "caremate_v1.2"
    respondent_credentials  TEXT,                    -- "GP" | "nurse_practitioner" | "specialist"

    -- Clinical assessment (same fields CareMate outputs)
    differential_diagnosis  JSONB DEFAULT '[]',     -- [{rank, condition_name, condition_code, confidence, reasoning}]
    triage_level            TEXT,                    -- "routine" | "priority" | "urgent"
    sats_colour             TEXT,                    -- "green" | "yellow" | "orange" | "red"
    investigations          JSONB DEFAULT '[]',     -- [{name, urgency, reasoning}]
    treatment_plan          JSONB DEFAULT '[]',     -- [{medication, dose, duration, reasoning}]
    referral_decision       TEXT,                    -- "refer" | "manage" | "conditional"
    referral_reason         TEXT,
    red_flags_identified    JSONB DEFAULT '[]',     -- [string]
    notes                   TEXT,                    -- free-text clinical reasoning

    -- Metadata
    time_taken_seconds      INTEGER,                -- how long they took
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_vignette_responses_vignette ON vignette_responses(vignette_id);
CREATE INDEX IF NOT EXISTS idx_vignette_responses_type ON vignette_responses(respondent_type);
CREATE INDEX IF NOT EXISTS idx_vignettes_active ON clinical_vignettes(active) WHERE active = TRUE;
