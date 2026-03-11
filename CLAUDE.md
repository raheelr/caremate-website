# CareMate Backend — Project Context

## What This Project Is
Clinical decision support tool for South African primary healthcare nurses. Matches patient symptoms to conditions in the SA Standard Treatment Guidelines (STG), returning ranked differentials, first-line medicines, danger signs, and referral criteria. Built by Raheel Retiwalla with clinical guidance from Dr Tasleem Ras and Numaan.

## Team
- **Raheel Retiwalla** — Product & engineering (builds everything with Claude Code)
- **Dr Tasleem Ras** — Clinical lead, quality improvement expert, 20 years SA public health
- **Numaan** — Business development, clinic access, stakeholder management

## Project Location
`~/Downloads/caremate-backend-v2`

## Domains
- **caremate.co.za** — public landing page / marketing website (registered 2026-03-02)
- **caremateai.health** — the live app (clinical tool)

## Current System Status (as of 2026-03-11)

### What's Built and Live
- **Backend**: FastAPI on Railway (`https://caremate-api-production.up.railway.app`), healthy
- **Frontend**: Lovable React app (GitHub: `raheelr/caremateaihealth`), connected to backend
- **Database**: Supabase + pgvector, 350 conditions (335 STG primary + 15 referral-only), 12,150+ clinical edges, 1,532 knowledge chunks
- **Triage Agent**: Full pipeline — extract → expand → search → score → safety → synthesise. Model fallback chain (Haiku → Sonnet) on 429/529 errors.
- **Encounter Agent**: SOAP note, care plan (multi-language + print), discharge summary generation — all STG-grounded
- **Clinical Assistant**: Multi-turn conversational agent with 9 tools (guidelines, drugs, safety, referrals, KB search). **Proactive context awareness** — automatically uses all patient context (pregnancy, allergies, vitals, current meds, age, sex, chronic conditions) without being asked. **Deterministic context injection** — tool handlers receive ground-truth patient data injected at dispatch time, never relying on LLM to relay patient facts correctly.
- **Knowledge Base**: 98 markdown files across 7 sources in `.claude-plugin/knowledge-base/` — STG (22), Hospital EML (26), Paediatric EML (24), O&G (2), Maternal (20), SATS (3), Road to Health (1)
- **Referral-Only Conditions**: 15 conditions (9 O&G + 6 Hospital EML critical) with `referral_required=TRUE` — triage FINDS them but says REFER, not TREAT
- **Care Level Boundaries**: Primary STG (treat) → Hospital EML (referral context only) → Specialist (flag + refer). Enforced in Clinical Assistant system prompt and knowledge model.
- **Clinical Opportunities Engine**: 27 deterministic rules — screening, dx-triggered workups, vitals nudges, SDOH, med safety
- **Proactive Prescription Safety**: Deterministic batch safety checker (`agents/prescription_safety.py`). No LLM. Checks: pregnancy (DB + 40+ in-memory drug classes), allergy cross-reactivity (8 drug classes + generic direct-name matching for ANY allergy), 17 drug-drug interaction rules, CNS stacking, paediatric dosing. Pre-screens formulary drugs with "Contraindicated" badges before nurse prescribes. Frontend: inline alerts on prescribed drugs, summary banner, "Ask CareMate for alternatives" link → auto-chat.
- **Pregnancy Safety Data**: 281/337 medicines (83%) have `pregnancy_safe` + `pregnancy_notes` populated in DB. 215 safe, 66 unsafe. Remaining 56 are non-prescribable entries (breast milk, gauze, sugar water, etc.).
- **Context Propagation**: Patient context (pregnancy, vitals, allergies, meds) flows from triage → encounter → Clinical Assistant automatically. **Deterministic injection** at tool dispatch — 6 tools get patient data injected directly, not via LLM params.
- **Demo Presentation**: Interactive clinician demo at `presentation/demo.html` — 18 screenshots, Emma Thompson walkthrough (triage → proactive safety alerts → Ask CareMate → prescribing → care plan in isiXhosa), targeted at Unjani/SHAWCO CEOs
- **Deep Test**: 92/92 conditions found (97.8% top-5, 92.4% Top-1), EXCELLENT grade
- **Performance**: 9-10s production triage, 5-6s local; 2-3s per encounter generation; <1ms opportunities
- **Deploy command**: `cd ~/Downloads/caremate-backend-v2 && railway up`
- **Presentation**: Architecture deck at `public/presentation/index.html` in frontend repo (Lovable-deployed)
- **Architecture Diagram**: Solution architecture at `public/architecture/index.html` — 5 tabs (Stack, Agents, Flow, Status, Scaling)
- **EHR Prototype**: Clickable prototype at `public/prototype/index.html` — patient queue, chart, encounter flow, AI assistant

### Folder Structure
```
caremate-backend-v2/
  agents/          ← 3 agents (triage, encounter, clinical assistant) + tools + SATS + opportunities + KB search
  api/             ← FastAPI server (BUILT, 19+ endpoints)
  safety/          ← safety checker (BUILT)
  db/              ← database queries + schema + migrations (001-004)
  ingestion/       ← STG PDF → knowledge graph pipeline (COMPLETE)
  .claude-plugin/  ← knowledge base architecture
    knowledge-base/  ← 98 markdown files across 7 clinical sources
      stg-primary/       ← 22 STG chapters
      hospital-eml/      ← 26 Hospital Level EML chapters
      paediatric-eml/    ← 24 Paediatric EML chapters
      obstetrics-gynae/  ← 2 O&G guideline files
      maternal-perinatal/ ← 20 maternal care files
      sats-triage/       ← 3 SATS manual files
      road-to-health/    ← 1 under-5 development file
    agents/            ← agent behaviour definitions (markdown)
  presentation/    ← architecture deck + clinician overview + live demo walkthrough
    demo.html          ← interactive clinician demo (Emma Thompson, 18 screenshots)
    demo-images/       ← 18 screenshots for the demo walkthrough
  docs/            ← TODO.md, EHR_PLAN.md, COMPETITIVE_LANDSCAPE.md, planning docs
  venv/            ← Python virtual environment
```

### Key Files
- `api/main.py` — FastAPI app, 19+ endpoints (triage + encounter + assistant + vignette survey + guidelines + prescribing), CORS, connection pool
- `api/models.py` — Pydantic models matching frontend contracts
- `agents/triage_agent.py` — TriageAgent class (analyze + refine), deterministic synthesis
- `agents/encounter_agent.py` — Encounter documentation: `generate_soap_note()`, `generate_care_plan()`, `generate_discharge_summary()` — all STG-grounded
- `agents/clinical_assistant.py` — ClinicalAssistant class, 9-tool agentic loop (guidelines, drugs, safety, referrals, KB search), multi-turn DB-persisted conversations
- `agents/kb_search.py` — File-based markdown KB search engine for Clinical Assistant tool #9, searches 98 files across 7 knowledge sources
- `agents/opportunities.py` — ClinicalOpportunitiesEngine: 27 deterministic rules across 5 categories + drug interaction sets (CYP450_INDUCERS, ORAL_CONTRACEPTIVES, ACE_INHIBITORS, NSAIDS, CNS_DEPRESSANTS)
- `agents/prescription_safety.py` — Proactive prescription safety checker: batch DB + in-memory rules, 8 allergy classes, 17 interaction pairs, 40+ pregnancy-unsafe drugs. Single source of truth for INTERACTION_RULES + PREGNANCY_UNSAFE_CLASSES (imported by clinical_assistant.py)
- `agents/tools.py` — 6 triage tool handlers, batch DB queries, prevalence boost
- `agents/sats.py` — SATS triage: `compute_sats_acuity()`, TEWS vital scoring (adult + child), clinical discriminators
- `agents/scoring_config.py` — centralised scoring constants, feature weights, prevalence tiers, non-disease penalties
- `agents/embeddings.py` — Voyage AI wrapper (voyage-3-lite, 512 dims)
- `safety/checker.py` — SafetyChecker (defense-in-depth review)
- `db/database.py` — all query functions incl `get_condition_features_batch()`, `get_vitals_mappings()`, assistant conversation persistence
- `db/migrations/004_care_setting_and_referral.sql` — adds `source_tag`, `care_setting`, `referral_required` columns to conditions table
- `ingestion/enrich_missing_edges.py` — targeted edge enrichment for conditions failing deep test (measles, CKD, hyperthyroidism, etc.)
- `ingestion/add_referral_conditions.py` — script to add referral-only conditions (15 O&G + Hospital EML critical) with symptom edges

### API Endpoints (Live)
```
Triage:
POST /api/triage/analyze     ← main triage (complaint + vitals + patient → differential)
POST /api/triage/refine      ← follow-up with assessment answers → re-ranked conditions
POST /api/triage/enrich      ← condition treatment details
GET  /api/health             ← health check (public, no API key)

Encounter Agent:
POST /api/encounter/generate-soap              ← SOAP note from encounter data
POST /api/encounter/generate-care-plan         ← patient-facing care plan (multi-language, print-ready)
POST /api/encounter/generate-discharge-summary ← clinician-facing discharge summary
POST /api/encounter/clinical-opportunities     ← proactive screening/safety/SDOH nudges

Clinical Assistant:
POST /api/assistant/chat     ← multi-turn conversational AI (9 tools, DB-persisted)

Clinical Reference:
POST /api/rag/query                    ← RAG-based clinical question answering
POST /api/guidelines/lookup            ← structured STG guideline sections
POST /api/prescribing/suggest-dosing   ← medication dosing
POST /api/prescribing/recommended-drugs ← drugs for condition by treatment line

Prescription Safety:
POST /api/prescriptions/safety-check   ← batch safety check (pregnancy, allergies, interactions, paediatric)

Phase II Clinician Survey:
GET  /api/vignettes                  ← list all vignettes
POST /api/vignettes                  ← create a vignette (admin)
GET  /api/vignettes/:id              ← get vignette (strips expected answers)
POST /api/vignettes/:id/respond      ← submit clinician/CareMate response
GET  /api/vignettes/:id/results      ← compare clinician vs CareMate
POST /api/vignettes/:id/run-caremate ← auto-run CareMate on a vignette
```
- API Key header: `X-API-Key`

### Clinical Assistant — 9 Tools + Deterministic Context Injection
1. `search_guidelines` — search STG knowledge base with optional condition filter (DB)
2. `lookup_condition` — full STG entry: description, danger signs, medicines, referral criteria (DB). **Injects**: pregnancy flag, allergies → annotates medicines with warnings
3. `check_red_flags` — match symptoms against danger signs (DB)
4. `search_medications` — drug dosing, pregnancy safety, paediatric dosing, routes (DB). **Injects**: pregnancy flag → annotates results with pregnancy warnings
5. `find_conditions` — differential diagnosis from symptoms, age/sex filtered (DB). **Injects**: age, sex from encounter context
6. `check_drug_safety` — patient-specific safety: pregnancy, interactions, age concerns (in-memory). **Injects**: pregnancy (overrides LLM), age, sex, current meds (all overridden)
7. `suggest_alternative` — find alternative drugs when one is contraindicated (DB). **Injects**: pregnancy flag, allergies → filters unsafe drugs, sorts safe first
8. `draft_referral_letter` — generate referral letter with encounter context (LLM). **Injects**: full patient demographics, vitals, diagnosis, allergies, meds
9. `search_knowledge_base` — search extended markdown KB beyond STG: Hospital EML, Paediatric EML, O&G, Maternal, SATS (file-based)

**Deterministic Context Injection** (`_inject_patient_context()` in `clinical_assistant.py`): Patient context is extracted ONCE from the encounter and injected into tool parameters at dispatch time. This ensures tools get ground-truth patient data regardless of what the LLM decides to pass. Critical fix: LLMs often omit or misreport patient context (e.g., pregnancy status) when calling tools.

### Clinical Opportunities Engine — 27 Rules, 5 Categories
- **Screening** (7): cervical cancer, breast self-exam, BP, glucose, HIV, TB, antenatal panel
- **Dx-Triggered Workups** (6): diabetes foot/eye/HbA1c, HTN target organ damage, HIV baseline, depression PHQ-9, STI partner notification
- **Vitals Safety Nudges** (4): incidental elevated BP, unexplained tachycardia, low SpO2, hypothermia
- **SDOH & Social Assistance** (4): SASSA disability grant, TB DOTS, free maternal care, SADAG helpline
- **Medication Safety** (6): ACE in pregnancy, warfarin+NSAID, CNS stacking, ACE+NSAID, **rifampicin+OCP** (CYP450 induction), **NSAID in pregnancy**

### Multi-Language Care Plans
Supports all 11 SA official languages at Grade 8 literacy level (print-ready):
`en` English, `zu` isiZulu, `xh` isiXhosa, `af` Afrikaans, `nso` Sepedi, `tn` Setswana, `st` Sesotho, `ts` Xitsonga, `ss` siSwati, `ve` Tshivenda, `nr` isiNdebele

### Triage Pipeline (2 LLM calls + DB queries)
1. **Haiku** (with fallback to Sonnet on 429/529): Extract symptoms from complaint (temperature=0)
2. **DB only**: expand_synonyms (batch LATERAL) → search_conditions (batch CTE, no vector search) → safety + details + features (all parallel, batch queries)
3. **Haiku** (parallel, with fallback): Slim re-rank + assessment questions (~160 output tokens, max_tokens=512) + safety review (~20 tokens)
   - Condition_symptoms generated from DB clinical_entities (no LLM) — all features returned with `is_red_flag` + `source_citation` metadata
   - Acuity computed deterministically via SATS/TEWS (`agents/sats.py`)
   - Referral-only conditions propagate `referral_required`, `care_setting`, `source_tag` to frontend
   - Vector search skipped (zero unique results vs graph+synonym search)

### Current Acuity System — SATS (South African Triage Scale)
- **Red** (Emergency, TEWS 7+) — immediate
- **Orange** (Very Urgent, TEWS 5-6) — < 10 minutes
- **Yellow** (Urgent, TEWS 3-4) — < 60 minutes
- **Green** (Routine, TEWS 0-2) — < 4 hours
- TEWS scores 7 components: HR, RR, SBP, Temp, consciousness (AVPU), mobility, trauma (each 0-3 points)
- Clinical discriminators (e.g. active seizure, chest pain at rest) override TEWS and assign colour directly
- Age-stratified thresholds (adult vs child HR/RR tables)
- Backward-compatible `acuity` field: Green→routine, Yellow→priority, Orange/Red→urgent
- Computed in `agents/sats.py: compute_sats_acuity()` + safety review escalation guard

## Environment Variables (.env)
```
DATABASE_URL=postgresql://...  ← Supabase connection string
ANTHROPIC_API_KEY=...
VOYAGE_API_KEY=...             ← Voyage AI for embeddings
API_KEY=...                    ← Backend API key
```

## Python Environment
```bash
cd ~/Downloads/caremate-backend-v2 && source venv/bin/activate
```

## Open Priorities (see `docs/TODO.md` for full list)
- **EVAH RFP**: Secure partners, finalize Pathway A proposal by Apr 1. Google/Heimerl team may also apply.
- **Competitive**: 6-12 month window as only working SA-STG CDST. See `docs/COMPETITIVE_LANDSCAPE.md`
- **Partnerships**: SHAWCO pilot first (March-April, EVAH data) → Unjani scale deployment (Q2, complement EMG)
- **Knowledge Base**: COMPLETE — 98 files across 7 sources. 15 referral-only conditions added. Care level boundaries enforced.
- **Validation**: Phase II survey frontend + Tasleem's vignettes, O&G re-test with Tasleem's sister
- **Compliance**: AWS Cape Town migration, SAHPRA consultation, POPIA DPIA
- **Product**: Care setting context switch, EHR layer, deploy to clinics, landing page (caremate.co.za)

---

## Roadmap — Feb 28, 2026 Team Call Decisions

### Validation Framework (3 phases, agreed with Tasleem)
- **Phase I**: Tasleem writes clinical vignettes across 7 domains, run blind against CareMate, results on live dashboard. Confirmed.
- **Phase II**: Tasleem creates standardised vignettes given to BOTH 5-10 clinicians AND CareMate. Clinician responses establish the "reasonable doctor norm" — CareMate must meet or exceed that norm. Same simulated cases for everyone.
- **Phase III**: Feasibility testing in real nurse clinics (focus on nurse clinics, not GPs). Numaan managing clinic access via Tutuk.

### Tier 1 — Immediate / Required for Validation
1. **SATS integration** — DONE (commit 341a589). `agents/sats.py` implements TEWS vital sign scoring (adult + child tables) + clinical discriminator checking. Replaces hard-coded vitals thresholds with nationally standardised SATS colour system (Red/Orange/Yellow/Green). Backward-compatible `acuity` field preserved.
2. **Phase I vignette validation** — DONE (2026-03-01). Tasleem provided 30 vignettes across 6 domains. **Results: 24/30 Top-1 correct (80%), adjusted 96% excluding KB gaps.** 5 misses all due to knowledge base gaps, not algorithm failures. See "Phase I Results" section below.
3. **Phase II clinician survey form** — BACKEND DONE. DB tables `clinical_vignettes` + `vignette_responses` created. 6 API endpoints: list/create/get vignettes, submit responses, compare results, auto-run CareMate. 3 seed vignettes loaded. **Awaiting**: frontend survey form + Tasleem's vignettes.
4. **Regional/local prevalence tuning** — SCORING CONFIG MERGED. `agents/scoring_config.py` centralises all feature weights, prevalence tiers, non-disease penalties. SA-wide prevalence tiers active (28 conditions). **Remaining**: per-province granularity (Western Cape vs KZN vs Limpopo) not yet implemented.

### Tier 2 — High Value, Near-Term
4. ~~**Investigation recommendations as structured output**~~ — DONE. Dx-triggered workup rules in `agents/opportunities.py` surface labs/investigations proactively.
5. **Care setting context switch** — `care_setting` parameter (primary/hospital/emergency) filtering knowledge corpus
6. ~~**Non-pharma interventions as structured output**~~ — DONE. Lifestyle advice + danger signs returned as structured arrays in care plan. SDOH assistance surfaced by opportunities engine.

### Tier 3 — Strategic / Medium-Term
7. **Longitudinal patient record (EHR layer)** — Patient history across encounters, continuity of care
8. ~~**Drug interaction / prescription clash checking**~~ — DONE. Proactive prescription safety checker (`agents/prescription_safety.py`) with 17 interaction rules, 8 allergy cross-reactivity classes, 40+ pregnancy-unsafe drugs. Also: `check_drug_safety` tool in Clinical Assistant (imports same rules) + medication safety rules in Opportunities Engine.
9. **API-first / embeddable mode** — CareMate as embeddable API inside other EHRs

### Tier 4 — Vision / Long-Term
10. Real-time clinical oversight dashboard
11. **Multi-country expansion** — `countries` + `guideline_sets` tables, `source_tag` on all knowledge corpus, country-adaptive patient IDs, triage system dispatch. Architecture designed for SSA, South Asia, Southeast Asia (aligned with EVAH RFP). Full plan: `docs/EHR_PLAN.md`
12. **Multi-care-setting** — primary + hospital + emergency + community. Same condition → different protocols per setting. SA Hospital STG = first multi-source ingestion after Tier 1 MVP.
13. Paramedic / pre-hospital triage
14. Multimodal input (glasses/audio)

### Key Architectural Decisions from Call
- **Multi-country from the architecture up** — `countries` + `guideline_sets` tables determine which clinical guidelines, triage system, formulary, and languages are active per facility
- **Knowledge corpus is multi-source** — `source_tag` column (stg_primary_za, sats_triage, stg_hospital_za, eml_ke, etc.) on conditions, knowledge_chunks, clinical_entities. All queries filter by facility context.
- **Multi-care-setting** — `care_setting` on facilities + triage requests. Primary care and hospital protocols differ for same conditions.
- **Scoring weights need clinician validation** — Phase II will reveal if 0.18/0.12/0.08 weights match how clinicians think
- **Output format needs to expand** — add: structured investigations, non-pharma interventions, clearer refer yes/no, follow-up plan

### Phase I Validation Results (2026-03-01)

**Test**: 30 clinical vignettes by Dr Tasleem Ras across 6 domains (Pregnancy, Under 5, Schoolgoing, Adolescent, Adult Health, Geriatrics — 5 each). Run blind through CareMate triage agent, system knowledge only.

**Files**: `Tasleem Tests March 1/run_vignettes.py` (runner), `vignette_results.json` (raw output), `*_FILLED.xlsx` (formatted results + comparison)

**Overall: 24/30 Top-1 correct (80%). Adjusted: 24/25 = 96% excluding KB gaps.**

| Domain | Top-1 | Missed |
|--------|:-----:|--------|
| Pregnancy | 4/5 | Case 4: foetal distress (not in DB) |
| Under 5 | 5/5 | — |
| Schoolgoing | 4/5 | Case 1: learning difficulty (not in DB) |
| Adolescent | 4/5 | Case 5: hyperthyroidism (age-filtered, 22yo vs 0-18 limit) |
| Adult Health | 5/5 | — |
| Geriatrics | 2-3/5 | Case 2: CKD (lab-only edges), Case 4: falls risk (not in DB) |

**5 Knowledge Base Gaps Identified — 3 FIXED, 2 unfixable**:
1. ~~**Foetal distress**~~ — FIXED. Added as referral-only condition with 19 symptom edges.
2. **Learning difficulty/ADHD** — not in STG DB. Out of scope for primary care STG.
3. ~~**Adult hyperthyroidism**~~ — FIXED. Extended max_age from 18→999 via `enrich_missing_edges.py`.
4. ~~**CKD symptom edges**~~ — FIXED. Added clinical presentation edges (fatigue, oedema, nocturia, pruritus) via `enrich_missing_edges.py`.
5. **Falls risk / orthostatic hypotension / polypharmacy** — not in STG DB. Out of scope.

### Reference Documents
- Meeting transcript: `Meeting started 2026_02_28 09_50 EST - Notes by Gemini.pdf` (in project root)
- Prioritised plan: `Feb 28 2026 CAll with Tasleem and Numaan.pdf` (in project root)
- Phase I vignettes: `Tasleem Tests March 1/TestVignettesMarch1Tasleem.pdf`

## EVAH RFP — Evidence for AI in Health (active, deadline April 1, 2026)

**Funders**: Wellcome Trust + Gates Foundation + Novo Nordisk Foundation (via J-PAL / APHRC)
**RFP**: `RFP/RFP Overview - Evidence for AI in Health_0.pdf`
**What**: Funding to evaluate AI-enabled CDSTs for frontline PHC workers in LMICs (Sub-Saharan Africa, South Asia, Southeast Asia)
**Pathway A** (our fit): Up to $1M, 3-12 months — real-world deployment & systems integration evidence for early-deployment tools
**Pathway B**: Up to $3M, 12-24 months — rigorous impact evaluation for at-scale tools

### Why CareMate Fits
- Exact use case match: AI CDST for SA primary care nurses, STG-based, SATS triage
- Validation data exists: 92/92 deep test, Phase I 24/30 vignettes (96% adj.)
- High-burden LMIC conditions, reducing inequities in under-resourced settings

### Key Eligibility Requirements
- Lead applicant must be **legally registered & operational in SSA** with PI based in region
- Must have **clinical implementation partner** (MoH, public sector facility, or NGO)
- Team must demonstrate **health systems research / impact evaluation expertise**
- **80% of funds** must flow to SSA-based entities
- Funding does **NOT** support software development — only evaluation + necessary implementation/scaling
- Must be **beyond proof of concept** with demonstrable accuracy in validation studies

### Action Items
- [x] RFP analysed (2026-03-02)
- [x] 5 clarifying questions drafted and submitted (2026-03-06)
- [x] Proposal draft ~50% complete
- [ ] Identify evaluation/research partner (UCT, Stellenbosch, Wits — health sciences)
- [ ] Formalize clinical implementation partner (letter of support from clinic/district)
- [ ] Build landing page at caremate.co.za
- [ ] Write Pathway A proposal (due April 1)

### Application Timeline
| Date | Milestone |
|------|-----------|
| Feb 20 | RFP released |
| **Mar 6** | **Questions deadline** (evah@povertyactionlab.org) |
| Mar 13 | FAQ published on RFP page |
| **Apr 1** | **Application deadline** (10am EDT / 4pm CAT) |
| Jun 2026 | Notification (estimated) |

## Key Architectural Decisions
- Standard Anthropic SDK + tool_use (NOT Agent SDK, NOT LangGraph/CrewAI)
- Haiku for all LLM calls (was Sonnet for synthesis, switched for speed)
- Symptom matching through knowledge graph — never condition name lookup
- Deterministic synthesis: only assessment_questions + re-ranking use LLM
- FastAPI backend on Railway, React frontend on Lovable
- **The Moat = Clinical Knowledge Engine** — ingestion pipeline + knowledge store + runtime services (synonym expansion, scoring, SATS, safety). Deterministic, validated, shared by all 3 agents
- **3 Agents with distinct problem shapes**: Triage (search-and-rank), Encounter (reasoning), Clinical Assistant (retrieval)
- **LLM calls are scaffolding** — will be replaced at scale by fine-tuned NER (extraction), trained ranker (re-ranking), rule-based checks (safety). Validation data = training data.
- **Claude-native NOT right for the moat** — moat needs deterministic scoring, queryable graph, validated weights (92/92 deep test). Claude-native is right for Encounter Agent + Clinical Assistant (reasoning tasks, lower volume)
- **Knowledge graph is ~3 MB** — fits in process memory. At scale: in-memory graph, no DB for triage, edge deployment on clinic tablets

## Scaling Path (decided 2026-03-05)
- **Phase 1 (now)**: Single instance, 100–1K/day, 9–10s latency, ~$0.01/triage
- **Phase 2 (post-validation)**: In-memory graph, N stateless instances, 10K–100K/day, 3–4s
- **Phase 3 (multi-country)**: Fully deterministic (no LLM for triage), country sharding, edge/offline, 1M+/day, <100ms, ~$0.0005/triage
- Cost at 1M/day: $3.6M/yr with LLMs vs $180K/yr deterministic (20x)
- Architecture diagram: `caremateai.health/architecture` (Scaling Path tab)
