# CareMate — Master To-Do List

> **Source of truth** for all open action items. Referenced from CLAUDE.md.
> Update status inline: `[ ]` pending, `[x]` done, `[-]` won't do.

---

## EVAH RFP (Deadline: April 1, 2026)

### Submissions & Deadlines
- [x] Submit 5 clarifying questions to evah@povertyactionlab.org (due March 6) — DONE
- [ ] Finalize and submit full Pathway A application (due **April 1, 10am EDT**) — **~50% drafted**

### Partners & Eligibility
- [ ] Secure SA-based lead applicant entity (Dr Tasleem's institution or SA university)
- [ ] Identify evaluation/research partner (UCT, Stellenbosch, or Wits — health sciences faculty)
- [ ] Formalize clinical implementation partner — letter of support from clinic or district health office
- [ ] Confirm 80% funds flow to SSA-based entities (budget structure)

### Proposal Deliverables
- [ ] Final review of proposal narrative (word counts, consistency check)
- [ ] Budget finalization ($620K breakdown across personnel, equipment, operations)
- [ ] Team CVs / biosketches for all named investigators
- [ ] Letters of support from partner institutions
- [ ] Attach CareMate whitepaper as pilot findings addendum
- [ ] Ethics approval plan (which IRB, timeline)

### Commitments Made in Proposal (must deliver if funded)
- [ ] Commission legal opinion on liability for AI-guided clinical decisions
- [ ] SAHPRA pre-submission consultation for SaMD classification
- [ ] POPIA Data Protection Impact Assessment (DPIA)
- [ ] ISO 14971 risk management documentation
- [ ] HPCSA Booklet 20 informed consent — culturally appropriate, multilingual
- [ ] Pre-register evaluation protocol (registry TBD)
- [ ] Set up Data Safety Monitoring Board (DSMB)
- [ ] Develop standardised consultation observation tool
- [ ] Performance drift monitoring infrastructure (Stream 6)

---

## Architecture — Knowledge Base Restructure (NEW — from March 7 decisions)

> **Decision**: Adopt Max Kaplan's `.claude-plugin/` pattern for Clinical Assistant + Encounter Agent knowledge. Triage agent keeps structured DB (the moat). This enables easy KB management — see what's in it, update it, add to it.

### Knowledge Base Architecture
- [x] Design `.claude-plugin/` directory structure for CareMate — DONE (2026-03-07)
- [x] Extract existing STG knowledge chunks into organized markdown files — DONE (22 chapters in `knowledge-base/stg-primary/`)
- [x] Build search/retrieval tool for Clinical Assistant to query markdown KB — DONE (`agents/kb_search.py`, tool #9)
- [x] Ensure triage agent continues using structured DB — CONFIRMED (triage unchanged, KB search is Clinical Assistant only)

### New Knowledge Sources — ALL COMPLETE (2026-03-08)
- [x] **Hospital Level EML (2019 v2.0)** — 26/26 chapters extracted to `knowledge-base/hospital-eml/`
- [x] **Paediatric EML (2017)** — 24/24 chapters extracted to `knowledge-base/paediatric-eml/`
- [x] **O&G / SASOG Guidelines** — 2 files extracted to `knowledge-base/obstetrics-gynae/`
- [x] **Maternal & Perinatal Care guidelines** — 20 files extracted to `knowledge-base/maternal-perinatal/`
- [x] **Road to Health Book** — 1 file extracted to `knowledge-base/road-to-health/`
- [x] **SATS Triage Manual** — 3 files extracted to `knowledge-base/sats-triage/`
- [x] Build multi-source ingestion framework with `source_tag` — DONE (migration 004: `source_tag`, `care_setting`, `referral_required` columns)
- [x] Define care-level hierarchy — DONE (Primary STG → Hospital EML → Specialist, enforced in system prompt + DB flags)

### Referral-Only Conditions — DONE (2026-03-08)
- [x] 15 conditions added: 9 O&G + 6 Hospital EML critical
- [x] Each has symptom edges (triage can FIND them) + `referral_required=TRUE` (triage says REFER)
- [x] Script: `ingestion/add_referral_conditions.py`
- [x] Care level boundaries enforced in Clinical Assistant system prompt

### Knowledge Base Admin / Visibility
- [ ] Dashboard showing: all conditions, edge counts, source documents, coverage gaps
- [ ] Ability to see which conditions have few edges (weak coverage)
- [ ] Source document management: which PDFs ingested, when, what they contributed
- [ ] Multi-source visibility: STG primary vs Hospital EML vs O&G vs Paediatric EML
- [ ] Simple way for clinical team (Tasleem) to flag gaps or corrections

---

## Product — Knowledge Base Gaps (from Phase I Validation)

### Fixable — ALL FIXED (2026-03-08)
- [x] Foetal distress — DONE. Added as referral-only condition with 19 symptom edges. Ranks #1 for "baby not moving" complaints.
- [x] CKD clinical symptom edges — DONE (fatigue, oedema, nocturia, pruritus). Verified working in deep test.
- [x] Hyperthyroidism age limit — DONE (max_age 18→999). Verified in deep test.
- [x] Post-menopausal bleeding — DONE. Added as referral-only condition. Ranks #1 with REFER flag. Tasleem's sister should re-test.

### Out of Scope (Not in Primary Care STG)
- [-] Learning difficulty / ADHD — not a primary care STG condition
- [-] Falls risk / orthostatic hypotension — not a discrete STG condition

---

## Product — Feature Roadmap

### Recently Completed (March 2026)
- [x] Investigation recommendations as structured output — DONE via Clinical Opportunities Engine (25 dx-triggered workup rules)
- [x] Non-pharma interventions as structured output — DONE via care plan (lifestyle advice, danger signs as structured arrays) + SDOH rules in Opportunities Engine
- [x] Drug interaction / prescription clash checking — DONE via `check_drug_safety` tool in Clinical Assistant + medication safety rules in Opportunities Engine
- [x] Multi-language care plans — 11 SA official languages at Grade 8 literacy (en, zu, xh, af, nso, tn, st, ts, ss, ve, nr)
- [x] Print-ready care plans — formatted for printing with clinician/patient signature lines
- [x] Clinical Assistant — 9-tool agentic loop (guidelines, conditions, red flags, meds, drug safety, alternatives, referral drafting, condition search, KB search)
- [x] Clinical Opportunities Engine — 27 deterministic rules (screening, dx-workups, vitals, SDOH, med safety)
- [x] Encounter Agent — SOAP note, care plan, discharge summary generation (all STG-grounded)
- [x] Referral letter generation — Clinical Assistant drafts referral letters with encounter context, asks for referral reason
- [x] Proactive referral triggers — eGFR <45, HbA1c >10, resistant hypertension flagged (fixed live during March 7 demo)
- [x] Clear STG vs non-STG demarcation — responses split into grounded STG section + "Clinical consideration (not from STG)" with disclaimer (fixed live during March 7 demo)
- [x] Knowledge Base Architecture — 98 markdown files across 7 sources in `.claude-plugin/knowledge-base/` (2026-03-08)
- [x] KB Search Tool — Clinical Assistant tool #9 (`search_knowledge_base`) searches Hospital EML, Paediatric EML, O&G, Maternal, SATS (2026-03-08)
- [x] Referral-Only Conditions — 15 conditions (9 O&G + 6 Hospital EML critical) with symptom edges + referral flags (2026-03-08)
- [x] Care Level Boundaries — Primary→Hospital→Specialist hierarchy enforced. Hospital EML = referral context only (Tasleem's rule, 2026-03-08)
- [x] Multi-care-setting DB — `source_tag`, `care_setting`, `referral_required` columns on conditions table. Same DB serves primary care + hospital staff (2026-03-08)
- [x] Clinical Assistant API endpoint — `POST /api/assistant/chat` with DB-persisted multi-turn conversations (2026-03-10)
- [x] Encounter Agent API endpoints — SOAP, care plan, discharge summary, clinical opportunities (2026-03-10)
- [x] Guidelines + prescribing endpoints — `POST /api/guidelines/lookup` (structured STG sections), `POST /api/prescribing/recommended-drugs` (by treatment line) (2026-03-10)
- [x] Structured dosing — parsed `dose_mg`, standard frequency abbreviations (od/bd/tds/qds/prn), route extraction (2026-03-10)
- [x] Triage model fallback — `_call_with_fallback()` retries Haiku → Sonnet on 429/529 errors, `max_retries=3` (2026-03-10)
- [x] Referral-only condition propagation — triage returns `referral_required`, `care_setting`, `source_tag` to frontend (2026-03-10)
- [x] Full condition_symptoms — all features returned with `is_red_flag` + `source_citation` metadata (was capped at 4) (2026-03-10)
- [x] CORS regex — switched from explicit allow_origins list to regex pattern covering all Lovable subdomains + caremateai.health (2026-03-10)
- [x] Edge enrichment script — `ingestion/enrich_missing_edges.py` for targeted edge additions (measles, CKD, hyperthyroidism, etc.) (2026-03-10)
- [x] Deep test updates — updated STG codes after dedup, added `alt_codes` support, added measles test case (2026-03-10)
- [x] Pipeline manifest verification — warns on missing extractions, suggests retry command (2026-03-10)
- [x] Prevalence tiers expanded — added pneumonia child variants, CCF, CKD (2026-03-10)
- [x] Proactive context awareness — Clinical Assistant now automatically uses ALL patient context (pregnancy, allergies, vitals, meds, age, sex, chronic conditions) without being asked (2026-03-10)
- [x] Context propagation fix — pregnancy/vitals/allergies/meds from triage now flow to Clinical Assistant via API normalization (2026-03-10)
- [x] Pregnancy safety data — 159/337 medicines (47%) populated with `pregnancy_safe` + `pregnancy_notes` (was 0%). Includes ARVs, TB drugs, insulin, top 30 drugs (2026-03-10)
- [x] Rifampicin + OCP interaction — CYP450 induction reduces oral contraceptive efficacy. Added to Opportunities Engine + Clinical Assistant (2026-03-10)
- [x] NSAID in pregnancy rule — added to Opportunities Engine as urgent safety alert (2026-03-10)
- [x] Drug interaction sets — `CYP450_INDUCERS`, `ORAL_CONTRACEPTIVES` exported from `opportunities.py`, imported by Clinical Assistant (2026-03-10)
- [x] Competitive landscape analysis — `docs/COMPETITIVE_LANDSCAPE.md` + Word doc `docs/CareMate_Competitive_Landscape.docx` (2026-03-10)
- [x] Proactive Prescription Safety Checker — `agents/prescription_safety.py`: deterministic batch safety checking (no LLM), endpoint `POST /api/prescriptions/safety-check`, frontend hook + PrescribeTab inline alerts + summary banner + formulary "Contraindicated" badges + "Ask CareMate for alternatives" auto-chat (2026-03-11)
- [x] Generic allergy matching — 2-phase: direct name match (any allergy) + 8 cross-reactivity classes (penicillins, cephalosporins, sulfonamides, macrolides, tetracyclines, fluoroquinolones, statins, opioids, NSAIDs) (2026-03-11)
- [x] Drug interaction rules expanded — 17 pairs (was 5): added methotrexate+NSAID, methotrexate+co-trimoxazole, lithium+ACE/NSAID, SSRI+tramadol, digoxin+amiodarone/verapamil, theophylline+ciprofloxacin, phenytoin+fluconazole, carbamazepine+macrolide, valproate+carbamazepine (2026-03-11)
- [x] Pregnancy-unsafe drug classes expanded — 40+ drugs in-memory: added tetracyclines, fluoroquinolones, statins, anticonvulsants, lithium, misoprostol, sulfonylureas, chloramphenicol (2026-03-11)
- [x] Pregnancy safety data expanded — 281/337 (83%, was 47%). 215 safe, 66 unsafe. Remaining 56 are non-prescribable entries (2026-03-11)
- [x] Single source of truth for safety rules — `INTERACTION_RULES` + `PREGNANCY_UNSAFE_CLASSES` defined in `prescription_safety.py`, imported by `clinical_assistant.py` (2026-03-11)
- [x] Demo walkthrough updated — 18 new screenshots from Emma Triage 2, 11 steps including proactive safety alerts + Ask CareMate auto-chat (2026-03-11)

### Tier 2 (High Value, Near-Term)
- [ ] Care setting context switch — `care_setting` parameter (primary/hospital/emergency) filtering knowledge corpus
- [ ] Per-province prevalence granularity (Western Cape vs KZN vs Limpopo)
- [ ] Populate contraindications + paediatric dosing columns in medicines table (currently 0% populated)
- [ ] Add `breastfeeding_safe` column to medicines table (not yet created)
- [ ] SOAP note quality improvement — Numaan: "needs a little more clinical detail" (compare against existing systems)
- [ ] South African Medicine Formulary integration — Tasleem suggested adding SAMF for deeper drug knowledge

### Tier 3 (Strategic, Medium-Term)
- [ ] Longitudinal patient record / EHR layer (see EMR section below)
- [ ] API-first / embeddable mode (CareMate as API inside other EHRs — for private sector integration with Alphabet/Discovery etc.)

### Tier 4 (Vision, Long-Term)
- [ ] Multi-country expansion (countries + guideline_sets tables, source_tag filtering)
- [ ] Multi-care-setting (primary + hospital + emergency + community)
- [ ] SA Hospital STG ingestion (first multi-source target)
- [ ] Real-time clinical oversight dashboard
- [ ] Paramedic / pre-hospital triage
- [ ] Multimodal input (audio, images)
- [ ] Ambient listening — automatically determining clinical needs from conversation (discussed March 7)
- [ ] ICD-10 + NAPI coding — premature for now, connect with Morgan (AI + coding). For primary care: ICD-10 diagnoses + NAPI medications would suffice. ICHI being rolled out nationally — monitor.

---

## EMR / EHR (from March 7 demo discussion)

> **Context**: Tasleem validated that SOAP note "meets all of the legal requirements" for a medical record. Only missing: clinician credentials and entry validation.

### SHAWCO Pilot (potential first deployment)
- [ ] Determine SHAWCO's current workflow (paper-based, confirmed by Tasleem)
- [ ] Define minimal EHR for SHAWCO: patient registration, clinician/student registration, encounter persistence, longitudinal access
- [ ] Explore funding via Allan Gray E2 (Numaan has contact, interested in demo after Ramadan)

### Build vs Integrate Decision
- [ ] Investigate National DoH electronic health record (Tasleem to check with Professor Chris — is it open source?)
- [ ] Evaluate OpenMRS integration path vs building lightweight from scratch
- [ ] Look at Tasleem's wife's site for flow/UX inspiration (private sector, more advanced)
- [ ] Look at CGM and Discovery's records system (demo with Tasleem's wife scheduled)
- [ ] Keep interoperability open-ended (Tasleem's request) — API-first so CareMate can plug into different EHRs

### Clinician Credentials (legal compliance)
- [ ] Record treating clinician's credentials (HPCSA/SANC registration) on every encounter
- [ ] Validation step before encounter data goes live in patient record

### Full EHR Plan
- See `docs/EHR_PLAN.md` for detailed feature list, DB schema, and implementation sequence

---

## Connectivity & Offline (from March 7 demo — raised repeatedly)

> **Context**: Tasleem: "connectivity was the biggest challenge" from published research on digital adoption in Africa. Some clinics in rural areas have no reliable network. Power cuts, LTE outages. If system doesn't work offline, clinics revert to paper.

### Immediate
- [ ] Numan to connect with Vodacom/MTN to understand rural connectivity and telco health programs
- [ ] Diagnose current deployment issue — Raheel noted things work locally but not reliably in production

### Architecture
- [ ] Evaluate offline-first options: ElectricSQL vs PowerSync vs custom sync (see EHR_PLAN.md Phase 0)
- [ ] Knowledge graph is ~3 MB — can be cached on device for offline triage
- [ ] Define what works offline vs what requires connectivity:
  - **Offline**: triage (with cached graph), care plan generation (if model cached), print
  - **Online**: LLM calls (extraction, re-ranking, assistant chat), DB sync, WhatsApp delivery
- [ ] Investigate Limba Health — local app with trained language model for SA dialects, reportedly works offline. Could use their model for speech understanding (Tasleem suggested connecting)

### Patient Communication
- [ ] WhatsApp care plan delivery — needs phone number update flow (low-income patients change numbers frequently; Tasleem: "depending on how they have money or not, they change their phone numbers regularly")
- [ ] Print fallback for patients without phones/data (DONE — print function built March 7)
- [ ] SMS fallback option (simpler than WhatsApp, works on feature phones)

---

## Validation & Monitoring

- [ ] Phase II clinician survey — frontend form (backend done, 6 endpoints live)
- [ ] Phase II vignettes — awaiting Tasleem's standardised vignettes
- [ ] Model version pinning + regression testing before any LLM update
- [ ] Automated performance drift detection (monthly reports against 92-case benchmark)
- [ ] TRIPOD-aligned reporting for validation results
- [ ] Plan future CONSORT-AI compliant stepped-wedge RCT (post Pathway A)
- [ ] O&G validation — get Tasleem's sister to test again after O&G knowledge added

---

## Infrastructure & Compliance

- [ ] Migrate LLM inference to AWS Bedrock Cape Town (af-south-1) for POPIA data residency
- [ ] Load testing for sustained real-world clinical use
- [ ] Evaluate alternative LLMs for abstraction layer (reduce single-vendor dependency)
- [ ] Monitoring dashboards for production system health
- [ ] Mobile responsive UI — Numaan couldn't use properly on phone ("I wasn't able to prescribe a treatment though... will do laptop test tmrw")

---

## Frontend & Marketing

- [ ] Build caremate.co.za landing page (domain registered 2026-03-02)
- [ ] Phase II clinician survey form in frontend app
- [ ] Clinic onboarding flow / training materials
- [ ] Allan Gray Ventures demo — Numaan has contact at mosque, keen to see demo after Ramadan

---

## Partnerships & Outreach (from March 7)

- [ ] **Morgan** — Tasleem suggested connecting re: AI + medical coding (ICD-10, NAPI, ICHI)
- [ ] **Limba Health** — local mental health app with SA dialect language model. Potential partner for speech/language understanding
- [ ] **Allan Gray Ventures** — Numaan's contact, demo after Ramadan
- [ ] **Vodacom/MTN** — Numaan to connect re: rural connectivity + health programs
- [ ] **Professor Chris** — Tasleem to check re: National DoH electronic health record (open source?)
- [ ] **SHAWCO** — university student health clinic, potential pilot site
- [ ] **Tasleem's wife** — demo of CGM + Discovery records system (scheduled for March 8, 4pm SA time)

---

*Last updated: 2026-03-11*
