-- Migration 005: Add duration_profile column to conditions
-- Enables DB-driven duration-aware scoring (replaces hardcoded DURATION_MODIFIERS)
-- 5 profiles: acute_self_limiting, acute_severe, subacute_infectious, chronic, variable
-- NULL = no modifier (default)

-- Add column
ALTER TABLE conditions ADD COLUMN IF NOT EXISTS duration_profile TEXT DEFAULT NULL;

-- CHECK constraint for valid values
ALTER TABLE conditions DROP CONSTRAINT IF EXISTS chk_duration_profile;
ALTER TABLE conditions ADD CONSTRAINT chk_duration_profile
    CHECK (duration_profile IS NULL OR duration_profile IN (
        'acute_self_limiting',
        'acute_severe',
        'subacute_infectious',
        'chronic',
        'variable'
    ));

-- Index for potential future filtering
CREATE INDEX IF NOT EXISTS idx_conditions_duration_profile ON conditions (duration_profile);

-- ───────────────────────────────────────────────────────────
-- Bulk UPDATE by chapter_name (22 chapters)
-- ───────────────────────────────────────────────────────────

-- Emergencies and Injuries → acute_severe
UPDATE conditions SET duration_profile = 'acute_severe'
WHERE chapter_name = 'Emergencies and Injuries' AND duration_profile IS NULL;

-- Cardiovascular Conditions → chronic
UPDATE conditions SET duration_profile = 'chronic'
WHERE chapter_name = 'Cardiovascular Conditions' AND duration_profile IS NULL;

-- Central Nervous System Conditions → variable
UPDATE conditions SET duration_profile = 'variable'
WHERE chapter_name = 'Central Nervous System Conditions' AND duration_profile IS NULL;

-- Dental and Oral Conditions → acute_self_limiting
UPDATE conditions SET duration_profile = 'acute_self_limiting'
WHERE chapter_name = 'Dental and Oral Conditions' AND duration_profile IS NULL;

-- Dermatological Conditions → variable
UPDATE conditions SET duration_profile = 'variable'
WHERE chapter_name = 'Dermatological Conditions' AND duration_profile IS NULL;

-- Ear, Nose and Throat Conditions → acute_self_limiting
UPDATE conditions SET duration_profile = 'acute_self_limiting'
WHERE chapter_name = 'Ear, Nose and Throat Conditions' AND duration_profile IS NULL;

-- Endocrine Conditions (ch6 O&G in DB) → NULL (leave as-is)
-- No update needed for this chapter

-- Eye Conditions → variable
UPDATE conditions SET duration_profile = 'variable'
WHERE chapter_name = 'Eye Conditions' AND duration_profile IS NULL;

-- Gastro-Intestinal Conditions → variable
UPDATE conditions SET duration_profile = 'variable'
WHERE chapter_name = 'Gastro-Intestinal Conditions' AND duration_profile IS NULL;

-- HIV and AIDS → subacute_infectious
UPDATE conditions SET duration_profile = 'subacute_infectious'
WHERE chapter_name = 'HIV and AIDS' AND duration_profile IS NULL;

-- Immunisation → NULL (leave as-is)
-- No update needed for this chapter

-- Infectious Conditions → acute_self_limiting
UPDATE conditions SET duration_profile = 'acute_self_limiting'
WHERE chapter_name = 'Infectious Conditions' AND duration_profile IS NULL;

-- Mental Health Conditions → variable
UPDATE conditions SET duration_profile = 'variable'
WHERE chapter_name = 'Mental Health Conditions' AND duration_profile IS NULL;

-- Musculoskeletal Conditions → chronic
UPDATE conditions SET duration_profile = 'chronic'
WHERE chapter_name = 'Musculoskeletal Conditions' AND duration_profile IS NULL;

-- Nutrition and Anaemia → chronic
UPDATE conditions SET duration_profile = 'chronic'
WHERE chapter_name = 'Nutrition and Anaemia' AND duration_profile IS NULL;

-- Obstetrics and Gynaecology (ch9 endo in DB) → chronic
UPDATE conditions SET duration_profile = 'chronic'
WHERE chapter_name = 'Obstetrics and Gynaecology' AND duration_profile IS NULL;

-- Pain → variable
UPDATE conditions SET duration_profile = 'variable'
WHERE chapter_name = 'Pain' AND duration_profile IS NULL;

-- Palliative Care → chronic
UPDATE conditions SET duration_profile = 'chronic'
WHERE chapter_name = 'Palliative Care' AND duration_profile IS NULL;

-- Respiratory Conditions → variable
UPDATE conditions SET duration_profile = 'variable'
WHERE chapter_name = 'Respiratory Conditions' AND duration_profile IS NULL;

-- Sexually Transmitted Infections → subacute_infectious
UPDATE conditions SET duration_profile = 'subacute_infectious'
WHERE chapter_name = 'Sexually Transmitted Infections' AND duration_profile IS NULL;

-- Urinary Conditions → variable
UPDATE conditions SET duration_profile = 'variable'
WHERE chapter_name = 'Urinary Conditions' AND duration_profile IS NULL;

-- Blood and Blood-Forming Organs → acute_severe
UPDATE conditions SET duration_profile = 'acute_severe'
WHERE chapter_name = 'Blood and Blood-Forming Organs' AND duration_profile IS NULL;

-- ───────────────────────────────────────────────────────────
-- Individual overrides (where condition differs from chapter)
-- ───────────────────────────────────────────────────────────

-- Respiratory overrides
UPDATE conditions SET duration_profile = 'subacute_infectious' WHERE stg_code LIKE '17.4%';  -- TB variants
UPDATE conditions SET duration_profile = 'chronic' WHERE stg_code = '17.1.5';               -- COPD
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code LIKE '17.3.4%';      -- Pneumonia (child)
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code LIKE '17.3.3%';      -- Pneumonia (adult)
UPDATE conditions SET duration_profile = 'acute_self_limiting' WHERE stg_code = '17.3.1';    -- Influenza
UPDATE conditions SET duration_profile = 'acute_self_limiting' WHERE stg_code = '19.2';      -- Common Cold
UPDATE conditions SET duration_profile = 'chronic' WHERE stg_code = '17.2';                  -- Asthma (chronic)
UPDATE conditions SET duration_profile = 'chronic' WHERE stg_code = '17.1.3';               -- Chronic Asthma

-- Infectious Conditions overrides
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code LIKE '10.7%';        -- Malaria
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code LIKE '10.5%';        -- Meningitis
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code LIKE '10.6%';        -- Meningitis variants

-- GI overrides
UPDATE conditions SET duration_profile = 'acute_self_limiting' WHERE stg_code = '2.9';      -- Acute diarrhoea
UPDATE conditions SET duration_profile = 'acute_self_limiting' WHERE stg_code = '2.9.1';    -- Paediatric diarrhoea
UPDATE conditions SET duration_profile = 'chronic' WHERE stg_code = '2.5';                  -- Chronic liver disease
UPDATE conditions SET duration_profile = 'acute_self_limiting' WHERE stg_code = '2.2';      -- Dyspepsia

-- O&G overrides (Endocrine Conditions chapter in DB = ch6 O&G)
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code = '6.4.2.5';         -- Eclampsia
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code = '6.7.1';           -- PPH
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code = '6.11';            -- Ectopic Pregnancy
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code = '6.4.7';           -- Preterm Labour
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code = '6.4.2.4';         -- Pre-eclampsia
UPDATE conditions SET duration_profile = NULL WHERE stg_code LIKE '7.%';                     -- Contraception → NULL

-- Endocrine overrides (Obstetrics and Gynaecology chapter in DB = ch9 endocrine)
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code = '9.3';             -- DKA
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code = '9.3.1';           -- DKA variant
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code = '9.1';             -- Hypoglycaemia
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code = '9.1.1';           -- Hypoglycaemia variant

-- Urinary overrides
UPDATE conditions SET duration_profile = 'acute_self_limiting' WHERE stg_code = '8.4';      -- UTI
UPDATE conditions SET duration_profile = 'acute_self_limiting' WHERE stg_code = '8.4.1';    -- UTI variant
UPDATE conditions SET duration_profile = 'chronic' WHERE stg_code = '8.1';                  -- CKD
UPDATE conditions SET duration_profile = 'chronic' WHERE stg_code = '8.1.1';                -- CKD variant

-- ENT overrides
UPDATE conditions SET duration_profile = 'chronic' WHERE stg_code = '19.1';                 -- Allergic Rhinitis

-- Referral-only overrides
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code = 'REF.OG.1';       -- Foetal Distress
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code = 'REF.OG.2';       -- Placental Abruption
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code = 'REF.OG.3';       -- Placenta Praevia
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code = 'REF.H.1';        -- Acute MI
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code = 'REF.H.2';        -- Stroke
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code = 'REF.H.3';        -- Status Epilepticus
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code = 'REF.H.4';        -- Acute Abdomen
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code = 'REF.H.5';        -- DKA (hospital)
UPDATE conditions SET duration_profile = 'acute_severe' WHERE stg_code = 'REF.H.6';        -- Pulmonary Embolism

-- Comment
COMMENT ON COLUMN conditions.duration_profile IS 'Duration scoring profile: acute_self_limiting, acute_severe, subacute_infectious, chronic, variable. NULL = no modifier. Used by triage to adjust scores based on symptom duration.';
