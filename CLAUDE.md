# CareMate Backend — Project Context

## What This Project Is
Clinical decision support tool for South African primary healthcare nurses. Matches patient symptoms to conditions in the SA Standard Treatment Guidelines (STG), returning ranked differentials, first-line medicines, danger signs, and referral criteria. Built by Raheel Retiwalla with clinical guidance from Dr Tasleem Ras and Numaan.

## Team
- **Raheel Retiwalla** — Product & engineering (builds everything with Claude Code)
- **Dr Tasleem Ras** — Clinical lead, quality improvement expert, 20 years SA public health
- **Numaan** — Business development, clinic access, stakeholder management

## Project Location
`~/Downloads/caremate-backend-v2`

## Current System Status (as of 2026-02-28)

### What's Built and Live
- **Backend**: FastAPI on Railway (`https://caremate-api-production.up.railway.app`), healthy
- **Frontend**: Lovable React app (GitHub: `raheelr/caremateaihealth`), connected to backend
- **Database**: Supabase + pgvector, 442 conditions, 12,732 clinical edges, 1,528 knowledge chunks
- **Triage Agent**: Full pipeline — extract → expand → search → score → safety → synthesise
- **Deep Test**: 65/65 conditions found (100% top-5 hit rate), EXCELLENT grade
- **Performance**: 11-17s end-to-end (was 35-48s, optimised 2026-02-28)
- **Deploy command**: `cd ~/Downloads/caremate-backend-v2 && railway up`

### Folder Structure
```
caremate-backend-v2/
  agents/          ← triage agent + tools (BUILT)
  api/             ← FastAPI server (BUILT, 6 endpoints)
  safety/          ← safety checker (BUILT)
  db/              ← database queries + schema
  ingestion/       ← STG PDF → knowledge graph pipeline (COMPLETE)
  venv/            ← Python virtual environment
  stg.pdf          ← SA Standard Treatment Guidelines source
  SATS-Manual-A5-LR-spreads.pdf  ← SA Triage Scale manual (TO INGEST)
```

### Key Files
- `api/main.py` — FastAPI app, CORS, connection pool
- `api/models.py` — Pydantic models matching frontend contracts
- `agents/triage_agent.py` — TriageAgent class (analyze + refine), deterministic synthesis
- `agents/tools.py` — 6 tool handlers, batch DB queries, prevalence boost
- `agents/embeddings.py` — Voyage AI wrapper (voyage-3-lite, 512 dims)
- `safety/checker.py` — SafetyChecker (defense-in-depth review)
- `db/database.py` — all query functions

### API Endpoints (Live)
```
POST /api/triage/analyze     ← main triage (complaint + vitals + patient → differential)
POST /api/triage/refine      ← follow-up with assessment answers → re-ranked conditions
POST /api/triage/enrich      ← condition treatment details
POST /api/rag/query          ← RAG-based clinical question answering
POST /api/prescribing/suggest-dosing  ← medication dosing
GET  /api/health             ← health check (public, no API key)
```
- API Key header: `X-API-Key`

### Triage Pipeline (3 LLM calls + DB queries)
1. **Haiku**: Extract symptoms from complaint (temperature=0)
2. **DB only**: expand_synonyms → search_conditions (batch CTE queries) → score → safety → details
3. **Haiku**: Slim re-rank + assessment questions (~200 output tokens, max_tokens=1024)
   - Condition_symptoms generated from DB clinical_entities (no LLM)
   - Acuity computed deterministically from vitals thresholds
   - Safety review runs in parallel with assessment generation

### Current Acuity System (3-tier)
- `routine` — normal vitals, no red flags (DEFAULT)
- `priority` — mild vital abnormality (temp >= 39C, HR > 120/<50) OR single red flag
- `urgent` — severe vitals (BP >= 180, SpO2 < 92, RR >= 30) OR multiple red flags
- Computed in `_compute_vitals_acuity()` + safety review escalation guard

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

---

## Roadmap — Feb 28, 2026 Team Call Decisions

### Validation Framework (3 phases, agreed with Tasleem)
- **Phase I**: Tasleem writes clinical vignettes across 7 domains, run blind against CareMate, results on live dashboard. Confirmed.
- **Phase II**: Tasleem creates standardised vignettes given to BOTH 5-10 clinicians AND CareMate. Clinician responses establish the "reasonable doctor norm" — CareMate must meet or exceed that norm. Same simulated cases for everyone.
- **Phase III**: Feasibility testing in real nurse clinics (focus on nurse clinics, not GPs). Numaan managing clinic access via Tutuk.

### Tier 1 — Immediate / Required for Validation
1. **SATS integration** — Ingest `SATS-Manual-A5-LR-spreads.pdf` (SA Triage Scale training manual, shared by Tasleem). Extract colour-coded acuity rules (Red/Orange/Yellow/Green) and discriminator lists. Replaces/augments current hard-coded vitals thresholds with nationally standardised system. Every SA emergency department uses this. **Status: NOT STARTED** — PDF in project root, no code written.
2. **Phase II clinician survey form** — Structured form where clinicians receive a vignette and record: differential diagnosis, triage level, recommended investigations, treatment plan, referral decision. Needs to be built before Phase II begins. **Status: FRONTEND SCAFFOLD ONLY** — `ValidationDashboard.tsx` exists with Phase I/II/III tabs. `clinical_vignettes` + `vignette_results` tables exist in Supabase. No backend endpoints for survey submission.
3. **Regional/local prevalence tuning** — Probability ranking must account for local epidemiology (Western Cape vs KZN vs Limpopo — TB rates, HIV prevalence, malaria zones differ dramatically). Currently have SA-wide prevalence tiers (23 conditions with 1.15x-1.25x boost). **Status: ~80% DONE IN WORKTREE** — `wizardly-solomon` worktree has `agents/scoring_config.py` with centralized prevalence config. NOT merged to main.

### Tier 2 — High Value, Near-Term
4. **Investigation recommendations as structured output** — Labs/imaging from STG as discrete fields, not buried in text chunks
5. **Care setting context switch** — `care_setting` parameter (primary/hospital/emergency) filtering knowledge corpus
6. **Non-pharma interventions as structured output** — Lifestyle advice, counselling, physiotherapy referrals

### Tier 3 — Strategic / Medium-Term
7. **Longitudinal patient record (EHR layer)** — Patient history across encounters, continuity of care
8. **Drug interaction / prescription clash checking** — Requires EHR layer
9. **API-first / embeddable mode** — CareMate as embeddable API inside other EHRs

### Tier 4 — Vision / Long-Term
10. Real-time clinical oversight dashboard
11. Multi-country / Africa expansion
12. Paramedic / pre-hospital triage
13. Multimodal input (glasses/audio)

### Key Architectural Decisions from Call
- **Stay focused on primary care first** — complete end-to-end before hospital/tertiary STGs
- **Knowledge corpus will become multi-source** — needs `source_tag` column (stg_primary, sats_triage, stg_hospital) before ingesting SATS
- **Scoring weights need clinician validation** — Phase II will reveal if 0.18/0.12/0.08 weights match how clinicians think
- **Output format needs to expand** — add: structured investigations, non-pharma interventions, clearer refer yes/no, follow-up plan

### Reference Documents
- Meeting transcript: `Meeting started 2026_02_28 09_50 EST - Notes by Gemini.pdf` (in project root)
- Prioritised plan: `Feb 28 2026 CAll with Tasleem and Numaan.pdf` (in project root)

## Key Architectural Decisions
- Standard Anthropic SDK + tool_use (NOT Agent SDK, NOT LangGraph/CrewAI)
- Haiku for all LLM calls (was Sonnet for synthesis, switched for speed)
- Symptom matching through knowledge graph — never condition name lookup
- Deterministic synthesis: only assessment_questions + re-ranking use LLM
- FastAPI backend on Railway, React frontend on Lovable
