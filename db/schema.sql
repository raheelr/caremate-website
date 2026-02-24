-- ============================================================
-- CareMate AI — PostgreSQL Schema
-- Replaces Neo4j + Supabase edge functions
-- ============================================================

-- Enable extensions (pgvector may not be available on all hosts)
DO $$ BEGIN
    CREATE EXTENSION IF NOT EXISTS vector;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pgvector not available — vector search will be disabled';
END $$;

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================
-- KNOWLEDGE BASE — The digitised STG
-- ============================================================

-- Every clinical condition from the STG
CREATE TABLE conditions (
    id                  SERIAL PRIMARY KEY,
    stg_code            TEXT NOT NULL UNIQUE,   -- e.g. "1.2", "4.7.1"
    icd10_codes         TEXT[],                 -- e.g. ["B37.0"]
    name                TEXT NOT NULL,          -- e.g. "Candidiasis, Oral (Thrush)"
    chapter             INTEGER,                -- 1-23
    chapter_name        TEXT,                   -- e.g. "Dental and Oral Conditions"
    
    -- Clinical content (raw text, preserved faithfully)
    description_text    TEXT,
    general_measures    TEXT,
    medicine_treatment  TEXT,
    danger_signs        TEXT,
    referral_criteria   TEXT,
    
    -- Extraction metadata
    source_pages        INTEGER[],              -- which PDF pages this came from
    extraction_confidence  FLOAT DEFAULT 1.0,  -- 0-1, lower = more ambiguous
    ambiguity_flags     JSONB DEFAULT '{}',     -- flags from Pass 1
    needs_review        BOOLEAN DEFAULT FALSE,
    
    -- Population applicability
    applies_to_children   BOOLEAN DEFAULT TRUE,
    applies_to_adults     BOOLEAN DEFAULT TRUE,
    applies_to_pregnant   BOOLEAN,              -- NULL = not specified
    
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Prerequisite context for discriminator-absence suppression
-- e.g. NMS requires antipsychotic medication in profile
CREATE TABLE condition_prerequisites (
    id              SERIAL PRIMARY KEY,
    condition_id    INTEGER REFERENCES conditions(id),
    prerequisite    TEXT NOT NULL,  -- e.g. "antipsychotic_medication"
    description     TEXT            -- human-readable explanation
);

-- ============================================================
-- KNOWLEDGE GRAPH — Relationships (replaces Neo4j)
-- ============================================================

CREATE TABLE clinical_entities (
    id              SERIAL PRIMARY KEY,
    canonical_name  TEXT NOT NULL UNIQUE,   -- normalised name
    entity_type     TEXT NOT NULL,          -- SYMPTOM | DRUG | CONDITION | RISK_FACTOR
    aliases         TEXT[],                 -- all known names/spellings
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE clinical_relationships (
    id                  SERIAL PRIMARY KEY,
    source_entity_id    INTEGER REFERENCES clinical_entities(id),
    target_entity_id    INTEGER REFERENCES clinical_entities(id),
    
    -- The relationship type
    relationship_type   TEXT NOT NULL,
    -- INDICATES    = symptom → condition (standard presentation)
    -- RED_FLAG     = symptom → condition (danger sign — triggers escalation)
    -- TREATS       = drug → condition
    -- CONTRAINDICATES = drug → condition/population
    -- RISK_FACTOR  = condition/factor → condition
    
    -- Scoring weights (from STG structure)
    feature_type    TEXT,
    -- diagnostic_feature   (weight 0.18) — pathognomonic
    -- presenting_feature   (weight 0.12) — common presentation  
    -- associated_feature   (weight 0.08) — may be present
    
    -- Source provenance — critical for auditability
    condition_id        INTEGER REFERENCES conditions(id),
    source_section      TEXT,   -- DESCRIPTION | DANGER_SIGNS | MANAGEMENT | REFERRAL
    source_page         INTEGER,
    
    -- Quality
    confidence          FLOAT DEFAULT 1.0,
    
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Prevent duplicate feature→condition edges (same feature, same condition, same type)
CREATE UNIQUE INDEX idx_unique_relationship 
    ON clinical_relationships(source_entity_id, condition_id, relationship_type, feature_type);

-- ============================================================
-- SYNONYM RINGS — Bridging patient language to clinical terms
-- ============================================================

CREATE TABLE synonym_rings (
    id                  SERIAL PRIMARY KEY,
    canonical_term      TEXT NOT NULL,          -- clinical term: "dysuria"
    synonym             TEXT NOT NULL,          -- patient term: "burning when I pee"
    relationship_type   TEXT DEFAULT 'synonym', -- synonym | abbreviation | child_symptom
    parent_concept      TEXT,                   -- for child symptoms: "dehydration"
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_synonym_unique ON synonym_rings(canonical_term, synonym);
CREATE INDEX idx_synonym_lookup ON synonym_rings USING gin(synonym gin_trgm_ops);

-- ============================================================
-- MEDICINES — The EML database  
-- ============================================================

CREATE TABLE medicines (
    id                  SERIAL PRIMARY KEY,
    name                TEXT NOT NULL UNIQUE,   -- generic name
    trade_names         TEXT[],
    schedule            INTEGER,                -- 0-6, enforces scope of practice
    routes              TEXT[],                 -- oral, IM, IV, topical, etc.
    
    -- Standard dosing (adults)
    adult_dose          TEXT,
    adult_frequency     TEXT,
    adult_duration      TEXT,
    adult_max_dose      TEXT,
    
    -- Paediatric dosing
    paediatric_dose_mg_per_kg  FLOAT,
    paediatric_frequency       TEXT,
    paediatric_note            TEXT,
    
    -- Safety
    contraindications   TEXT[],
    pregnancy_safe      BOOLEAN,
    pregnancy_notes     TEXT,
    
    source_page         INTEGER,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Which medicines treat which conditions (with dosing context)
CREATE TABLE condition_medicines (
    id              SERIAL PRIMARY KEY,
    condition_id    INTEGER REFERENCES conditions(id),
    medicine_id     INTEGER REFERENCES medicines(id),
    treatment_line  TEXT,       -- first_line | second_line | alternative | adjunct
    dose_context    TEXT,       -- the full dosing text from the STG for this condition
    age_group       TEXT,       -- adults | children | all
    special_notes   TEXT
);

-- ============================================================
-- VECTOR SEARCH — Semantic similarity on guideline chunks
-- ============================================================

CREATE TABLE knowledge_chunks (
    id              SERIAL PRIMARY KEY,
    condition_id    INTEGER REFERENCES conditions(id),
    
    -- The actual text chunk
    chunk_text      TEXT NOT NULL,
    section_role    TEXT NOT NULL,
    -- CLINICAL_PRESENTATION | DANGER_SIGNS | MANAGEMENT | 
    -- REFERRAL | DOSING_TABLE | DECISION_TREE | CAUTIONS
    
    -- Structural flags
    is_table        BOOLEAN DEFAULT FALSE,
    is_algorithm    BOOLEAN DEFAULT FALSE,
    
    -- Vector embedding (768-dim) — requires pgvector extension
    -- embedding       vector(768),  -- uncomment when pgvector is available
    
    source_page     INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Uncomment when pgvector is available:
-- CREATE INDEX idx_chunks_vector ON knowledge_chunks 
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_chunks_condition ON knowledge_chunks(condition_id);
CREATE INDEX idx_chunks_section ON knowledge_chunks(section_role);

-- ============================================================
-- CLINICAL ONTOLOGY — Chief complaints → conditions routing
-- ============================================================

CREATE TABLE chief_complaints (
    id          SERIAL PRIMARY KEY,
    complaint   TEXT NOT NULL UNIQUE,   -- "chest pain", "burning when I pee"
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE complaint_condition_routes (
    complaint_id    INTEGER REFERENCES chief_complaints(id),
    condition_id    INTEGER REFERENCES conditions(id),
    confidence_boost FLOAT DEFAULT 0.0,  -- up to +0.10
    PRIMARY KEY (complaint_id, condition_id)
);

-- ============================================================
-- CLINICAL SESSIONS — Patient encounters
-- ============================================================

CREATE TABLE clinical_sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Patient context (de-identified)
    age_years       INTEGER,
    sex             TEXT,       -- male | female
    is_pregnant     BOOLEAN,
    weight_kg       FLOAT,
    
    -- Complaint
    chief_complaint         TEXT NOT NULL,
    complaint_hash          TEXT,   -- PHI-scrubbed hash for monitoring
    clinical_terms_extracted TEXT[],
    
    -- Vitals
    vitals                  JSONB,  -- bp_systolic, bp_diastolic, temp, spo2, pulse, rr
    
    -- Triage outcome
    triage_level            TEXT,   -- URGENT | PRIORITY | ROUTINE
    red_flags_triggered     TEXT[],
    
    -- Differential diagnosis
    differential            JSONB,  -- list of {condition_id, condition_name, confidence}
    confirmed_diagnosis_id  INTEGER REFERENCES conditions(id),
    confirmed_diagnosis_name TEXT,
    
    -- Encounter state
    session_state           TEXT DEFAULT 'TRIAGE',
    -- TRIAGE | QUEUE | ENCOUNTER | PRESCRIBING | DISCHARGE
    
    -- Prescriptions issued
    prescriptions           JSONB,
    
    -- Documentation
    soap_note               TEXT,
    referral_document       TEXT,
    
    -- Monitoring
    pivot_count             INTEGER DEFAULT 0,
    assessment_rounds       INTEGER DEFAULT 0,
    nurse_overrides         JSONB DEFAULT '[]',
    
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    updated_at              TIMESTAMPTZ DEFAULT NOW()
);

-- Question/answer history within a session
CREATE TABLE session_answers (
    id              SERIAL PRIMARY KEY,
    session_id      UUID REFERENCES clinical_sessions(id),
    condition_id    INTEGER REFERENCES conditions(id),
    question_text   TEXT NOT NULL,
    answer          BOOLEAN,        -- TRUE = yes/present, FALSE = no/absent
    is_red_flag     BOOLEAN DEFAULT FALSE,
    answered_at     TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- MONITORING & AUDIT
-- ============================================================

CREATE TABLE triage_events (
    id                  SERIAL PRIMARY KEY,
    session_id          UUID,
    complaint_hash      TEXT,
    top_condition       TEXT,
    top_confidence      FLOAT,
    triage_level        TEXT,
    pivot_count         INTEGER,
    assessment_rounds   INTEGER,
    is_no_match         BOOLEAN DEFAULT FALSE,
    red_flags_triggered TEXT[],
    duration_ms         INTEGER,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- INGESTION TRACKING
-- ============================================================

CREATE TABLE ingestion_runs (
    id              SERIAL PRIMARY KEY,
    started_at      TIMESTAMPTZ DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    source_file     TEXT,
    pages_processed INTEGER DEFAULT 0,
    conditions_extracted INTEGER DEFAULT 0,
    conditions_needing_review INTEGER DEFAULT 0,
    status          TEXT DEFAULT 'running'  -- running | complete | failed
);

CREATE TABLE ingestion_progress (
    page_number     INTEGER PRIMARY KEY,
    status          TEXT,   -- pending | complete | failed
    condition_name  TEXT,
    processed_at    TIMESTAMPTZ DEFAULT NOW()
);
