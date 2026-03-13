-- Migration 004: Add care_setting, source_tag, and referral_required columns
-- Enables multi-care-setting support (primary → hospital → specialist)
-- and referral-only conditions that triage can identify but nurses don't treat

-- Add columns
ALTER TABLE conditions ADD COLUMN IF NOT EXISTS source_tag TEXT DEFAULT 'stg_primary_za';
ALTER TABLE conditions ADD COLUMN IF NOT EXISTS care_setting TEXT DEFAULT 'primary';
ALTER TABLE conditions ADD COLUMN IF NOT EXISTS referral_required BOOLEAN DEFAULT FALSE;

-- Index for care_setting filtering (triage queries will filter by this)
CREATE INDEX IF NOT EXISTS idx_conditions_care_setting ON conditions (care_setting);
CREATE INDEX IF NOT EXISTS idx_conditions_source_tag ON conditions (source_tag);

-- Backfill existing conditions
UPDATE conditions SET source_tag = 'stg_primary_za', care_setting = 'primary', referral_required = FALSE
WHERE source_tag IS NULL OR source_tag = 'stg_primary_za';

COMMENT ON COLUMN conditions.source_tag IS 'Knowledge source: stg_primary_za, eml_hospital_za, og_guidelines_za, eml_paeds_za, maternal_perinatal_za, sats_triage';
COMMENT ON COLUMN conditions.care_setting IS 'Care level: primary, hospital, specialist. Determines which conditions appear in triage based on facility type.';
COMMENT ON COLUMN conditions.referral_required IS 'If true, triage identifies this condition but instructs REFER instead of TREAT. Used for conditions beyond primary care scope.';
