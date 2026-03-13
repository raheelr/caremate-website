-- Migration 007: Clinical Reasoning Rules
-- Stores structured clinical rules extracted from STG/KB markdown files.
-- Used by the question engine to generate deterministic, STG-grounded
-- assessment questions during triage.

CREATE TABLE IF NOT EXISTS clinical_reasoning_rules (
    id                    SERIAL PRIMARY KEY,
    condition_stg_code    TEXT NOT NULL,
    condition_name        TEXT NOT NULL,
    source_file           TEXT NOT NULL,
    source_section        TEXT,
    source_text           TEXT,

    rule_type             TEXT NOT NULL CHECK (rule_type IN (
        'lab_threshold',
        'vital_threshold',
        'examination_finding',
        'investigation_rec',
        'drug_condition_mod',
        'referral_trigger',
        'severity_classifier',
        'clinical_sign',
        'history_discriminator'
    )),

    rule_data             JSONB NOT NULL,
    assessment_question   TEXT,
    question_type         TEXT DEFAULT 'yes_no' CHECK (question_type IN (
        'yes_no', 'numeric', 'select', 'free_text'
    )),
    question_options      TEXT[],

    discriminating_power  FLOAT DEFAULT 0.5,
    rules_out_codes       TEXT[],
    is_red_flag           BOOLEAN DEFAULT FALSE,

    applies_to_age_min    INTEGER,
    applies_to_age_max    INTEGER,
    applies_to_sex        TEXT CHECK (applies_to_sex IS NULL OR applies_to_sex IN ('male', 'female')),
    country_code          TEXT DEFAULT 'ZA',
    source_tag            TEXT DEFAULT 'stg_primary_za',
    active                BOOLEAN DEFAULT TRUE,
    extraction_confidence FLOAT DEFAULT 0.8,
    reviewed              BOOLEAN DEFAULT FALSE,
    created_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_crr_condition ON clinical_reasoning_rules(condition_stg_code);
CREATE INDEX IF NOT EXISTS idx_crr_type ON clinical_reasoning_rules(rule_type);
CREATE INDEX IF NOT EXISTS idx_crr_active ON clinical_reasoning_rules(active) WHERE active = TRUE;
