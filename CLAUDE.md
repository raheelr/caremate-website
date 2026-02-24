# CareMate Backend — Project Memory

## What This Project Is
Clinical decision support tool for South African primary healthcare nurses. Helps with triage by matching patient symptoms to conditions in the SA Standard Treatment Guidelines (STG), returning ranked differentials, first-line medicines, danger signs, and referral criteria.

## Project Location
`~/Downloads/caremate-backend-v2`

## Folder Structure
```
caremate-backend-v2/
  db/              ← database connection, schema, migrations
  ingestion/       ← STG PDF → knowledge graph pipeline (COMPLETE)
  agents/          ← triage agent (IN PROGRESS)
  api/             ← FastAPI server (TO BUILD)
  safety/          ← safety checker (TO BUILD)
  venv/            ← Python virtual environment
  stg.pdf          ← SA Standard Treatment Guidelines source document
  requirements.txt
  SETUP_GUIDE.md
```

## Database — Supabase + pgvector
- Host: Supabase (URL in .env as DATABASE_URL)
- Extension: pgvector enabled
- Key tables:
  - `conditions` — 410 conditions extracted from STG
  - `clinical_relationships` — 4,470 rows linking entities to conditions
  - `clinical_entities` — symptoms, signs, risk factors, red flags
  - `medicines` — 335 medicines with doses, routes, treatment lines
  - `condition_medicines` — links medicines to conditions with treatment_line, dose, special_notes
  - `referral_criteria` — when to refer for each condition
  - `knowledge_chunks` — 975 text chunks for semantic search

## Ingestion Pipeline — COMPLETE
- Processed 356/444 sections (88 filtered: heading-only, redirects, non-clinical)
- 410 conditions ingested (some resumed/retried)
- QA validator: 9/10 vignettes passing
- Failing vignette: V01 (PLHIV dysphagia typed as INDICATES not RED_FLAG — low priority)
- Key files: `ingestion/pipeline.py`, `ingestion/extractor.py`, `ingestion/segmenter.py`
- Resume command: `python3 ingestion/pipeline.py --pdf stg.pdf --resume`

## QA Validator
- File: `ingestion/qa_validator.py`
- Run: `python3 ingestion/qa_validator.py`
- 10 clinical vignettes covering: Oral Thrush, Depression, Hypertension, Diarrhoea

## Triage Agent — IN PROGRESS
Using standard Anthropic Python SDK with tool_use (NOT the Agent SDK — wrong tool for this use case).
Model: claude-haiku-4-5-20251001 for tool calls (cheap, fast), claude-sonnet-4-6 for final synthesis.

### Six tools planned:
1. `extract_symptoms` — natural language → standardised clinical terms
2. `expand_synonyms` — clinical terms → patient language variants
3. `search_conditions` — symptoms → matching conditions from DB
4. `score_differential` — rank conditions by symptom match + context
5. `get_condition_detail` — full STG entry for one condition
6. `check_safety_flags` — RED_FLAG features present? → escalation

### Agent loop:
```
Nurse input → extract_symptoms → expand_synonyms → search_conditions
           → score_differential → check_safety_flags
           → enough info? → if no: ask follow-up question
                          → if yes: return ranked differential + STG guidance
```

### How the knowledge graph lookup works:
- Never do name search — always symptom match through clinical_relationships
- Weights: diagnostic_feature=0.18, presenting_feature=0.12, associated_feature=0.08
- RED_FLAG match adds 0.10 bonus
- Context filtering: some conditions have prerequisites (hiv_positive, child_under_5)
  → flag for confirmation rather than silently suppress

## Frontend — Lovable React App
- GitHub: https://github.com/raheelr/caremateaihealth
- Currently uses its own AI logic (to be replaced with our backend)
- The swap is one line: change API base URL to point to our Railway backend
- Need to audit existing API contracts before building backend endpoints

## Deployment Target
- Backend: Railway (account exists)
- Backend URL pattern: `https://caremate-api.railway.app`
- Frontend: stays on Lovable/Vercel

## API Endpoints to Build
```
POST /api/triage/encounter    ← start new encounter, returns encounter_id
POST /api/triage/message      ← send nurse message, returns agent response
GET  /api/conditions/:id      ← full condition detail
GET  /api/health              ← health check for Railway
```
(Confirm these match Lovable frontend before finalising)

## Environment Variables (.env)
```
DATABASE_URL=postgresql://...  ← Supabase connection string
ANTHROPIC_API_KEY=...
```

## Python Environment
Always activate venv first:
```bash
cd ~/Downloads/caremate-backend-v2
source venv/bin/activate
```

## Key Architectural Decisions
- Standard Anthropic SDK + tool_use, NOT the Claude Agent SDK (that's for code/file agents)
- No agent frameworks (LangGraph, CrewAI) — unnecessary complexity for sequential workflow
- One triage agent + one safety checker (separate Haiku call reviewing output before it reaches nurse)
- Symptom matching through knowledge graph — never condition name lookup
- FastAPI backend on Railway, React frontend stays on Lovable

## Next Immediate Steps
1. Read Lovable GitHub repo to understand existing API contracts
2. Build `agents/triage_agent.py` with six tools
3. Build `api/main.py` FastAPI wrapper
4. Test localhost chat experience
5. Deploy to Railway
6. Point Lovable frontend at Railway URL

## Session Notes
- Previous work done in Claude.ai chat (context limit hit — switching to Claude Code)
- Full transcript saved at: /mnt/transcripts/ (on Claude.ai infrastructure, not local)
- ingestion_progress.json tracks all 356 conditions — do not delete
