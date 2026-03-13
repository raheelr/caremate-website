-- Migration 006: Clinical Knowledge Tables
-- Moves ~400 hardcoded clinical data entries from Python files to database.
-- All data loads into ClinicalDataCache at startup (~400 rows, <100ms).
-- No per-request DB queries added. Triage latency unaffected.

-- ═══════════════════════════════════════════════════════════════════════
-- 1. drug_classes — pharmacological class definitions
-- ═══════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS drug_classes (
    id           SERIAL PRIMARY KEY,
    class_name   TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    class_type   TEXT NOT NULL,  -- 'allergy_class' | 'interaction_group' | 'pharmacological'
    country_code TEXT DEFAULT 'ZA',
    source_tag   TEXT DEFAULT 'stg_primary_za',
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- ═══════════════════════════════════════════════════════════════════════
-- 2. drug_class_members — which drugs belong to which class
-- ═══════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS drug_class_members (
    id            SERIAL PRIMARY KEY,
    drug_class_id INTEGER NOT NULL REFERENCES drug_classes(id) ON DELETE CASCADE,
    medicine_id   INTEGER REFERENCES medicines(id),
    drug_name     TEXT NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (drug_class_id, drug_name)
);

CREATE INDEX IF NOT EXISTS idx_dcm_class ON drug_class_members(drug_class_id);
CREATE INDEX IF NOT EXISTS idx_dcm_name ON drug_class_members(drug_name);

-- ═══════════════════════════════════════════════════════════════════════
-- 3. allergy_cross_reactivity — keyword → drug class mapping
-- ═══════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS allergy_cross_reactivity (
    id            SERIAL PRIMARY KEY,
    allergy_keyword TEXT NOT NULL,
    drug_class_id INTEGER NOT NULL REFERENCES drug_classes(id) ON DELETE CASCADE,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (allergy_keyword, drug_class_id)
);

CREATE INDEX IF NOT EXISTS idx_acr_keyword ON allergy_cross_reactivity(allergy_keyword);

-- ═══════════════════════════════════════════════════════════════════════
-- 4. drug_interaction_rules — drug-drug interaction pairs
-- ═══════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS drug_interaction_rules (
    id               SERIAL PRIMARY KEY,
    group_a_class_id INTEGER REFERENCES drug_classes(id),
    group_a_drug     TEXT,
    group_b_class_id INTEGER REFERENCES drug_classes(id),
    group_b_drug     TEXT,
    severity         TEXT NOT NULL CHECK (severity IN ('critical', 'warning', 'info')),
    message          TEXT NOT NULL,
    bidirectional    BOOLEAN DEFAULT TRUE,
    country_code     TEXT DEFAULT 'ZA',
    active           BOOLEAN DEFAULT TRUE,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    CHECK (group_a_class_id IS NOT NULL OR group_a_drug IS NOT NULL),
    CHECK (group_b_class_id IS NOT NULL OR group_b_drug IS NOT NULL)
);

-- ═══════════════════════════════════════════════════════════════════════
-- 5. pregnancy_unsafe_rules — class/individual drug pregnancy safety
-- ═══════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS pregnancy_unsafe_rules (
    id            SERIAL PRIMARY KEY,
    drug_class_id INTEGER REFERENCES drug_classes(id),
    medicine_id   INTEGER REFERENCES medicines(id),
    drug_name     TEXT NOT NULL,
    reason        TEXT NOT NULL,
    severity      TEXT DEFAULT 'critical' CHECK (severity IN ('critical', 'warning')),
    active        BOOLEAN DEFAULT TRUE,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (drug_name)
);

CREATE INDEX IF NOT EXISTS idx_pur_name ON pregnancy_unsafe_rules(drug_name);

-- ═══════════════════════════════════════════════════════════════════════
-- 6. lab_result_patterns — deterministic lab → condition mapping
-- ═══════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS lab_result_patterns (
    id                  SERIAL PRIMARY KEY,
    lab_id              TEXT NOT NULL UNIQUE,
    display_label       TEXT NOT NULL,
    text_patterns       TEXT[] NOT NULL,
    structured_names    TEXT[] NOT NULL,
    positive_keywords   TEXT[],
    numeric_threshold   FLOAT,
    threshold_direction TEXT CHECK (threshold_direction IS NULL OR threshold_direction IN ('above', 'below')),
    condition_codes     JSONB NOT NULL,
    force_rank_one      BOOLEAN DEFAULT TRUE,
    score_boost         FLOAT DEFAULT 2.0,
    add_symptoms        TEXT[],
    country_code        TEXT DEFAULT 'ZA',
    active              BOOLEAN DEFAULT TRUE,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ═══════════════════════════════════════════════════════════════════════
-- 7. clinical_discriminators — SATS triage discriminator phrases
-- ═══════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS clinical_discriminators (
    id            SERIAL PRIMARY KEY,
    phrase        TEXT NOT NULL,
    acuity_colour TEXT NOT NULL CHECK (acuity_colour IN ('red', 'orange', 'yellow')),
    population    TEXT NOT NULL CHECK (population IN ('adult', 'paediatric', 'all')),
    category      TEXT,
    country_code  TEXT DEFAULT 'ZA',
    source_tag    TEXT DEFAULT 'sats_triage',
    active        BOOLEAN DEFAULT TRUE,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (phrase, population)
);

-- ═══════════════════════════════════════════════════════════════════════
-- 8. clinical_opportunity_rules — screening, dx-triggered, etc.
-- ═══════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS clinical_opportunity_rules (
    id                  SERIAL PRIMARY KEY,
    rule_id             TEXT NOT NULL UNIQUE,
    rule_type           TEXT NOT NULL,
    title               TEXT NOT NULL,
    description         TEXT NOT NULL,
    action_label        TEXT NOT NULL,
    priority            TEXT NOT NULL CHECK (priority IN ('urgent', 'warning', 'info')),
    stg_reference       TEXT,
    min_age             INTEGER,
    max_age             INTEGER,
    sex                 TEXT CHECK (sex IS NULL OR sex IN ('male', 'female')),
    exclude_pregnancy   BOOLEAN DEFAULT FALSE,
    require_pregnancy   BOOLEAN DEFAULT FALSE,
    dx_contains         TEXT[],
    exclude_dx_contains TEXT[],
    dx_stg_prefix       TEXT[],
    vitals_check        TEXT,
    suppress_if_dx_contains TEXT[],
    med_check           TEXT,
    dx_contains_any     TEXT[],
    require_symptom_or_dx JSONB,
    country_code        TEXT DEFAULT 'ZA',
    active              BOOLEAN DEFAULT TRUE,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ═══════════════════════════════════════════════════════════════════════
-- 9. clinical_keyword_sets — generic keyword groupings
-- ═══════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS clinical_keyword_sets (
    id         SERIAL PRIMARY KEY,
    set_name   TEXT NOT NULL,
    keyword    TEXT NOT NULL,
    active     BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (set_name, keyword)
);

CREATE INDEX IF NOT EXISTS idx_cks_name ON clinical_keyword_sets(set_name);

-- ═══════════════════════════════════════════════════════════════════════
-- 3 New columns on conditions
-- ═══════════════════════════════════════════════════════════════════════

ALTER TABLE conditions ADD COLUMN IF NOT EXISTS prevalence_tier TEXT
    CHECK (prevalence_tier IS NULL OR prevalence_tier IN ('high', 'moderate'));

ALTER TABLE conditions ADD COLUMN IF NOT EXISTS pregnancy_required BOOLEAN DEFAULT FALSE;

ALTER TABLE conditions ADD COLUMN IF NOT EXISTS is_non_disease BOOLEAN DEFAULT FALSE;
