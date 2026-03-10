"""
CareMate Clinical Triage API
-----------------------------
FastAPI backend replacing Supabase Edge Functions.

Endpoints:
  POST /api/triage/analyze       — initial triage analysis
  POST /api/triage/refine        — iterative refinement with answers
  POST /api/triage/enrich        — protocol enrichment for a condition
  POST /api/rag/query            — RAG-powered clinical Q&A (single-shot, legacy)
  POST /api/assistant/chat       — multi-turn conversational clinical assistant
  POST /api/prescribing/suggest-dosing — medicine dosing suggestions
  GET  /api/health               — health check

  Phase II Clinician Survey:
  GET  /api/vignettes                       — list all vignettes
  POST /api/vignettes                       — create a vignette
  GET  /api/vignettes/:id                   — get single vignette (strips expected answers)
  POST /api/vignettes/:id/respond           — submit clinician/CareMate response
  GET  /api/vignettes/:id/results           — compare clinician vs CareMate
  POST /api/vignettes/:id/run-caremate      — auto-run CareMate on vignette

Run:
  cd ~/Downloads/caremate-backend-v2
  source venv/bin/activate
  uvicorn api.main:app --reload --port 8000
"""

import os
import sys
import json
import logging
from contextlib import asynccontextmanager

import anthropic
import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from api.models import (
    AnalyzeRequest,
    RefineRequest,
    EnrichRequest,
    RAGQueryRequest,
    DosingRequest,
    AssistantChatRequest,
    GenerateSOAPRequest,
    GenerateCarePlanRequest,
    GenerateDischargeRequest,
    GuidelinesLookupRequest,
    RecommendedDrugsRequest,
    CreateVignetteRequest,
    SubmitResponseRequest,
)
from agents.triage_agent import TriageAgent
from agents.clinical_assistant import ClinicalAssistant
from agents.encounter_agent import generate_soap_note, generate_care_plan, generate_discharge_summary
from safety.checker import SafetyChecker
from db.database import (
    get_condition_detail, get_condition_by_stg_code, search_knowledge_chunks,
    create_vignette, list_vignettes, get_vignette, save_vignette_response,
    get_vignette_responses, get_vignette_comparison,
    create_assistant_conversation, get_assistant_messages, save_assistant_message,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("caremate.api")


# ── Application lifespan (connection pool) ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB pool on startup, close on shutdown."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not set — add it to .env")

    logger.info("Creating database connection pool...")
    pool = await asyncpg.create_pool(database_url, min_size=2, max_size=10)
    app.state.pool = pool
    app.state.agent = TriageAgent(pool)
    app.state.assistant = ClinicalAssistant(pool)
    app.state.safety = SafetyChecker()

    # Verify DB connection
    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM conditions")
    logger.info(f"Database ready — {count} conditions loaded")

    yield

    await pool.close()
    logger.info("Database pool closed")


app = FastAPI(
    title="CareMate Clinical Triage API",
    version="1.0.0",
    lifespan=lifespan,
)


# ── CORS ─────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.lovableproject\.com|https://.*\.lovable\.app|https://caremateai\.health|https://www\.caremateai\.health|http://localhost:(5173|3000)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Key Authentication ────────────────────────────────────────────────────

API_KEY = os.getenv("API_KEY")

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    # Health check is public
    if request.url.path == "/api/health":
        return await call_next(request)

    # OPTIONS requests pass through (CORS preflight)
    if request.method == "OPTIONS":
        return await call_next(request)

    # If no API_KEY is configured, skip auth (local dev)
    if not API_KEY:
        return await call_next(request)

    key = request.headers.get("X-API-Key")
    if key != API_KEY:
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key"},
        )

    return await call_next(request)


# ── GET /api/health ──────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    try:
        async with app.state.pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM conditions")
        return {"status": "healthy", "conditions_loaded": count}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}



# ── POST /api/triage/analyze ────────────────────────────────────────────────

@app.post("/api/triage/analyze")
async def analyze_complaint(request: AnalyzeRequest):
    """Initial triage: complaint + vitals → symptoms, acuity, ranked conditions, follow-up questions."""
    try:
        patient = request.patient.model_dump(exclude_none=True) if request.patient else None
        vitals = request.vitals.model_dump(exclude_none=True) if request.vitals else None
        core_history = request.core_history.model_dump(exclude_none=True) if request.core_history else None

        # Run triage agent (includes safety review internally, parallelized)
        result = await app.state.agent.analyze(
            complaint=request.complaint,
            patient=patient,
            vitals=vitals,
            core_history=core_history,
        )

        return result

    except Exception as e:
        logger.error(f"Analyze failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /api/triage/refine ─────────────────────────────────────────────────

@app.post("/api/triage/refine")
async def refine_assessment(request: RefineRequest):
    """Iterative refinement: nurse answers → re-ranked conditions, next questions."""
    try:
        conditions = [c.model_dump() for c in request.conditions]
        patient = request.patient.model_dump(exclude_none=True) if request.patient else None

        result = await app.state.agent.refine(
            complaint=request.complaint,
            conditions=conditions,
            answers=request.answers,
            all_time_answers=request.all_time_answers,
            current_round=request.current_round,
            patient=patient,
            request_next_round=request.request_next_round,
            stg_feature_data=request.stg_feature_data,
        )

        return result

    except Exception as e:
        logger.error(f"Refine failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /api/triage/enrich ─────────────────────────────────────────────────

@app.post("/api/triage/enrich")
async def enrich_presentation(request: EnrichRequest):
    """Fetch enriched clinical content for a confirmed condition."""
    try:
        async with app.state.pool.acquire() as conn:
            condition = await get_condition_by_stg_code(conn, request.condition_code)
            if not condition:
                raise HTTPException(status_code=404, detail=f"Condition {request.condition_code} not found")

            detail = await get_condition_detail(conn, condition["id"])
            chunks = await search_knowledge_chunks(conn, "", condition_id=condition["id"], limit=5)

        # Use Haiku to generate clinical prompts from STG data
        import anthropic
        client = anthropic.AsyncAnthropic()

        stg_context = (
            f"Condition: {request.condition_name}\n"
            f"Description: {(detail.get('description_text') or '')[:500]}\n"
            f"General measures: {(detail.get('general_measures') or '')[:500]}\n"
            f"Danger signs: {(detail.get('danger_signs') or '')[:300]}\n"
            f"Medicines: {json.dumps(detail.get('medicines_json', []), default=str)[:500]}"
        )

        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            temperature=0,
            messages=[{"role": "user", "content": (
                f"Given this STG guideline data, generate clinical management prompts.\n\n"
                f"{stg_context}\n\n"
                f"Return JSON: {{\"prompts\": [{{\"id\": \"p1\", \"question\": \"...\", \"type\": \"yes_no\"}}], "
                f"\"sources\": [\"STG section ref\"]}}\n"
                f"Generate 3-5 decision-support prompts for the nurse. Return ONLY valid JSON."
            )}],
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enrich failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /api/rag/query ─────────────────────────────────────────────────────

@app.post("/api/rag/query")
async def query_rag(request: RAGQueryRequest):
    """RAG-powered clinical Q&A against knowledge chunks."""
    try:
        async with app.state.pool.acquire() as conn:
            if request.condition_code:
                # Condition-scoped search
                condition = await get_condition_by_stg_code(conn, request.condition_code)
                if condition:
                    chunks = await search_knowledge_chunks(
                        conn, request.query, condition_id=condition["id"], limit=request.max_chunks
                    )
                else:
                    chunks = await search_knowledge_chunks(conn, request.query, limit=request.max_chunks)
            else:
                chunks = await search_knowledge_chunks(conn, request.query, limit=request.max_chunks)

        if not chunks:
            return {
                "answer": "No relevant guidelines found for this query.",
                "sources": [],
                "graph": {"entities": [], "paths": []},
                "metadata": {"chunks_retrieved": 0, "framework": request.framework},
            }

        context = "\n\n".join([
            f"[{r['section_role']}] {r['condition_name']} (STG {r['stg_code']}):\n{r['chunk_text']}"
            for r in chunks
        ])

        import anthropic
        client = anthropic.AsyncAnthropic()

        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            temperature=0,
            messages=[{"role": "user", "content": (
                f"Answer this clinical query using ONLY the provided STG guideline context. "
                f"Reference sources with [Source N] notation.\n\n"
                f"Query: {request.query}\n\n"
                f"STG Context:\n{context}\n\n"
                f"If the context doesn't contain enough information, say so clearly."
            )}],
        )

        sources = [
            {
                "heading": r.get("condition_name", ""),
                "section_ref": r.get("section_role", ""),
                "similarity": round(r.get("similarity", 0.0), 2) if "similarity" in r else 0.85,
                "excerpt": (r.get("chunk_text", "")[:200] + "...") if r.get("chunk_text") else "",
            }
            for r in chunks
        ]

        return {
            "answer": response.content[0].text,
            "sources": sources,
            "graph": {"entities": [], "paths": []},
            "metadata": {
                "framework": request.framework,
                "chunks_retrieved": len(chunks),
                "graph_entities_found": 0,
                "graph_paths_found": 0,
            },
        }

    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /api/assistant/chat ──────────────────────────────────────────────────

@app.post("/api/assistant/chat")
async def assistant_chat(request: AssistantChatRequest):
    """Multi-turn conversational clinical assistant with tool use."""
    try:
        async with app.state.pool.acquire() as conn:
            # Resolve or create conversation
            if request.conversation_id:
                conversation_id = request.conversation_id
                # Load conversation history (last 10 turns = 20 messages)
                history = await get_assistant_messages(conn, conversation_id, limit=20)
            else:
                encounter_id = (request.encounter_context or {}).get("encounter_id")
                conversation_id = await create_assistant_conversation(
                    conn,
                    encounter_id=encounter_id,
                    patient_context=request.encounter_context,
                )
                history = []

            # Save user message
            user_msg_id = await save_assistant_message(
                conn, conversation_id, "user", request.message,
            )

        # Run the assistant agent
        result = await app.state.assistant.chat(
            message=request.message,
            conversation_history=history,
            encounter_context=request.encounter_context,
        )

        # Save assistant response
        async with app.state.pool.acquire() as conn:
            asst_msg_id = await save_assistant_message(
                conn,
                conversation_id,
                "assistant",
                result["response"],
                sources=result.get("sources"),
                tools_used=result.get("tools_used"),
                tool_calls=result.get("tool_calls_detail"),
            )

        return {
            "conversation_id": conversation_id,
            "response": result["response"],
            "sources": result.get("sources", []),
            "tools_used": result.get("tools_used", []),
            "message_id": asst_msg_id,
        }

    except anthropic.APIStatusError as e:
        logger.error(f"Assistant chat failed (Anthropic API): {e}", exc_info=True)
        if e.status_code == 529:
            raise HTTPException(
                status_code=503,
                detail="The AI service is temporarily overloaded. Please try again in a few seconds.",
            )
        raise HTTPException(
            status_code=502,
            detail="The AI service returned an error. Please try again.",
        )
    except Exception as e:
        logger.error(f"Assistant chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /api/prescribing/suggest-dosing ─────────────────────────────────────

@app.post("/api/prescribing/suggest-dosing")
async def suggest_dosing(request: DosingRequest):
    """Get structured dosing suggestion for a drug + condition pair."""
    import re as _re

    try:
        async with app.state.pool.acquire() as conn:
            # Look up medicine with condition-specific dosing
            dosing = await conn.fetchrow("""
                SELECT m.name, m.adult_dose, m.adult_frequency, m.adult_duration,
                       m.paediatric_dose_mg_per_kg, m.paediatric_frequency,
                       m.contraindications, m.pregnancy_safe, m.pregnancy_notes,
                       m.routes,
                       cm.dose_context, cm.treatment_line, cm.age_group, cm.special_notes
                FROM medicines m
                LEFT JOIN condition_medicines cm ON cm.medicine_id = m.id
                LEFT JOIN conditions c ON c.id = cm.condition_id
                WHERE m.name ILIKE $1
                AND (c.name ILIKE $2 OR c.stg_code = $2)
                LIMIT 1
            """, f"%{request.drugName}%", f"%{request.conditionName}%")

        if dosing:
            # Build structured suggestion from DB fields
            dose_text = dosing["dose_context"] or dosing["adult_dose"] or ""
            freq_raw = dosing["adult_frequency"] or ""
            dur_raw = dosing["adult_duration"] or ""

            # Extract numeric dose_mg from dose text (e.g. "500mg" → 500, "1g" → 1000)
            dose_mg = None
            mg_match = _re.search(r"(\d+(?:\.\d+)?)\s*mg", dose_text, _re.IGNORECASE)
            if mg_match:
                dose_mg = float(mg_match.group(1))
            else:
                g_match = _re.search(r"(\d+(?:\.\d+)?)\s*g(?:\b|[^a-zA-Z])", dose_text, _re.IGNORECASE)
                if g_match:
                    dose_mg = float(g_match.group(1)) * 1000

            # Map frequency text to standard abbreviations
            freq_map = {
                "once daily": "od", "daily": "od", "od": "od",
                "twice daily": "bd", "bd": "bd", "12-hourly": "bd", "12 hourly": "bd",
                "three times daily": "tds", "tds": "tds", "8-hourly": "tds", "8 hourly": "tds",
                "four times daily": "qds", "qds": "qds", "6-hourly": "qds", "6 hourly": "qds",
                "as needed": "prn", "prn": "prn", "stat": "stat",
            }
            frequency = freq_map.get(freq_raw.lower().strip(), freq_raw) if freq_raw else ""

            # Extract duration days
            duration_days = None
            if dur_raw:
                dur_match = _re.search(r"(\d+)", dur_raw)
                if dur_match:
                    duration_days = int(dur_match.group(1))

            # Route
            routes = dosing.get("routes") or []
            route = routes[0] if routes else "oral"

            # Clinical note
            clinical_notes = []
            if dosing["special_notes"]:
                clinical_notes.append(dosing["special_notes"])
            if request.patientAge and request.patientAge < 18 and dosing["paediatric_dose_mg_per_kg"]:
                clinical_notes.append(f"Paediatric: {dosing['paediatric_dose_mg_per_kg']} mg/kg")
            if dosing["pregnancy_safe"] is False:
                clinical_notes.append("WARNING: Not safe in pregnancy")
            clinical_note = ". ".join(clinical_notes) if clinical_notes else ""

            suggestion = {
                "dose": dose_text,
                "dose_mg": dose_mg,
                "frequency": frequency,
                "duration_days": duration_days,
                "formulation": "",
                "route": route,
                "confidence": "high",
                "clinical_note": clinical_note,
                "source_quote": dosing["dose_context"] or dosing["adult_dose"] or "",
                "treatment_line": (dosing["treatment_line"] or "").replace("_", " "),
            }
            return {"suggestion": suggestion}

        # Fallback: use Haiku with knowledge chunks
        async with app.state.pool.acquire() as conn:
            chunks = await conn.fetch("""
                SELECT kc.chunk_text FROM knowledge_chunks kc
                JOIN conditions c ON c.id = kc.condition_id
                WHERE c.name ILIKE $1 AND kc.section_role IN ('DOSING_TABLE', 'MANAGEMENT')
                LIMIT 3
            """, f"%{request.conditionName}%")

        context = "\n".join([r["chunk_text"] for r in chunks]) if chunks else "No STG data found."

        import anthropic
        client = anthropic.AsyncAnthropic()

        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            temperature=0,
            messages=[{"role": "user", "content": (
                f"From the STG text below, extract dosing for {request.drugName} "
                f"for {request.conditionName}.\n"
                f"Patient: age={request.patientAge}, sex={request.patientSex}\n\n"
                f"STG text:\n{context}\n\n"
                f"Return ONLY valid JSON (no markdown fences):\n"
                f'{{"dose": "e.g. 500mg", "dose_mg": 500, "frequency": "tds", '
                f'"duration_days": 5, "route": "oral", "clinical_note": "...", '
                f'"source_quote": "exact quote from text"}}\n'
                f"If the drug is not found, return: "
                f'{{"dose": "", "dose_mg": null, "frequency": "", "duration_days": null, '
                f'"route": "", "clinical_note": "Drug not found in STG for this condition", '
                f'"source_quote": ""}}'
            )}],
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(text)
            parsed["confidence"] = "medium"
            parsed.setdefault("formulation", "")
            parsed.setdefault("treatment_line", "")
            return {"suggestion": parsed}
        except json.JSONDecodeError:
            # Final fallback: return as text in clinical_note
            return {"suggestion": {
                "dose": "", "dose_mg": None, "frequency": "", "duration_days": None,
                "formulation": "", "route": "", "confidence": "low",
                "clinical_note": text[:500], "source_quote": "", "treatment_line": "",
            }}

    except Exception as e:
        logger.error(f"Dosing suggestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Encounter Agent — Clinical Documentation ──────────────────────────────

@app.post("/api/encounter/generate-soap")
async def encounter_generate_soap(request: GenerateSOAPRequest):
    """Generate an STG-grounded SOAP note from encounter data."""
    try:
        async with app.state.pool.acquire() as conn:
            result = await generate_soap_note(
                conn,
                condition_name=request.condition_name,
                condition_code=request.condition_code,
                patient=request.patient,
                chief_complaint=request.chief_complaint,
                collected_data=request.collected_data,
                prescriptions=request.prescriptions,
                triage_context=request.triage_context,
            )
        return result
    except Exception as e:
        logger.error(f"SOAP generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/encounter/generate-care-plan")
async def encounter_generate_care_plan(request: GenerateCarePlanRequest):
    """Generate a patient-facing care plan grounded in STG."""
    try:
        async with app.state.pool.acquire() as conn:
            result = await generate_care_plan(
                conn,
                condition_name=request.condition_name,
                condition_code=request.condition_code,
                patient=request.patient,
                prescriptions=request.prescriptions,
                language=request.language,
            )
        return result
    except Exception as e:
        logger.error(f"Care plan generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/encounter/generate-discharge-summary")
async def encounter_generate_discharge(request: GenerateDischargeRequest):
    """Generate a clinician-facing discharge summary."""
    try:
        async with app.state.pool.acquire() as conn:
            result = await generate_discharge_summary(
                conn,
                condition_name=request.condition_name,
                condition_code=request.condition_code,
                patient=request.patient,
                prescriptions=request.prescriptions,
                collected_data=request.collected_data,
                triage_context=request.triage_context,
                soap_note=request.soap_note,
            )
        return result
    except Exception as e:
        logger.error(f"Discharge summary generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /api/encounter/clinical-opportunities ────────────────────────────

@app.post("/api/encounter/clinical-opportunities")
async def get_clinical_opportunities(request: dict):
    """Evaluate proactive clinical opportunities for the current encounter.

    Deterministic rules engine — no DB calls, no LLM calls.
    Surfaces: screening reminders, diagnosis-triggered workups,
    incidental vitals findings, SDOH programs, medication safety.
    """
    from agents.opportunities import ClinicalOpportunitiesEngine

    try:
        engine = ClinicalOpportunitiesEngine()
        opportunities = engine.evaluate(
            patient_age=request.get("patient_age"),
            patient_sex=request.get("patient_sex"),
            pregnancy_status=request.get("pregnancy_status"),
            confirmed_diagnosis=request.get("confirmed_diagnosis"),
            diagnosis_stg_code=request.get("diagnosis_stg_code"),
            vitals=request.get("vitals"),
            prescriptions=request.get("prescriptions", []),
            extracted_symptoms=request.get("extracted_symptoms", []),
        )
        return {"opportunities": opportunities}
    except Exception as e:
        logger.error(f"Clinical opportunities failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /api/guidelines/lookup ────────────────────────────────────────────

@app.post("/api/guidelines/lookup")
async def guidelines_lookup(request: GuidelinesLookupRequest):
    """Look up STG guideline sections for a condition — structured + raw fallback."""
    try:
        async with app.state.pool.acquire() as conn:
            # Find condition with all text fields
            condition = await conn.fetchrow("""
                SELECT id, name, stg_code, description_text, general_measures,
                       medicine_treatment, danger_signs, referral_criteria
                FROM conditions WHERE name ILIKE $1 LIMIT 1
            """, f"%{request.condition_name}%")
            if not condition:
                return {"condition_name": request.condition_name, "sections": [], "structured": None}

            cond_id = condition["id"]

            # ── Build structured data ──

            # 1. Parse description
            desc_raw = condition["description_text"] or ""
            desc_summary = desc_raw[:300].rsplit(".", 1)[0] + "." if len(desc_raw) > 300 and "." in desc_raw[:300] else desc_raw

            # 2. Parse general measures
            gm_raw = condition["general_measures"] or ""
            gm_summary = gm_raw[:300].rsplit(".", 1)[0] + "." if len(gm_raw) > 300 and "." in gm_raw[:300] else gm_raw

            # 3. Parse danger signs → array of strings
            ds_raw = condition["danger_signs"] or ""
            danger_signs = _parse_bullet_list(ds_raw)

            # 4. Parse referral criteria → array of strings
            ref_raw = condition["referral_criteria"]
            referral_criteria = []
            if ref_raw:
                if isinstance(ref_raw, str):
                    try:
                        referral_criteria = json.loads(ref_raw)
                    except (json.JSONDecodeError, TypeError):
                        referral_criteria = _parse_bullet_list(ref_raw)
                elif isinstance(ref_raw, list):
                    referral_criteria = ref_raw

            # 5. Structured medicines from condition_medicines
            med_rows = await conn.fetch("""
                SELECT m.name, m.routes, m.adult_dose, m.adult_frequency, m.adult_duration,
                       m.paediatric_dose_mg_per_kg, m.paediatric_frequency,
                       m.pregnancy_safe, m.schedule,
                       cm.treatment_line, cm.dose_context, cm.special_notes
                FROM condition_medicines cm
                JOIN medicines m ON m.id = cm.medicine_id
                WHERE cm.condition_id = $1
                ORDER BY
                    CASE cm.treatment_line
                        WHEN 'first_line' THEN 1
                        WHEN 'second_line' THEN 2
                        WHEN 'alternative' THEN 3
                        WHEN 'adjunct' THEN 4
                        ELSE 5
                    END, m.name
            """, cond_id)

            medicines = []
            for m in med_rows:
                med = {
                    "name": m["name"],
                    "treatment_line": m["treatment_line"] or "first_line",
                    "dose": m["dose_context"] or m["adult_dose"] or "",
                    "special_notes": m["special_notes"] or "",
                }
                if request.patient_age and request.patient_age < 18 and m["paediatric_dose_mg_per_kg"]:
                    med["dose"] = f"{m['paediatric_dose_mg_per_kg']} mg/kg {m['paediatric_frequency'] or ''}"
                medicines.append(med)

            # 6. Clinical tables & algorithms
            table_rows = await conn.fetch("""
                SELECT section_role, chunk_text, is_table, is_algorithm
                FROM knowledge_chunks
                WHERE condition_id = $1
                  AND (is_table = true OR is_algorithm = true)
                ORDER BY is_algorithm DESC, length(chunk_text) ASC
                LIMIT 4
            """, cond_id)

            clinical_tables = []
            for t in table_rows:
                text = t["chunk_text"] or ""
                if len(text) < 30:
                    continue
                ttype = "algorithm" if t["is_algorithm"] else "table"
                # Extract first line as title
                first_line = text.split("\n", 1)[0].strip()
                title = first_line if len(first_line) < 80 else ""
                clinical_tables.append({
                    "title": title,
                    "content": text,
                    "type": ttype,
                })

            structured = {
                "stg_code": condition["stg_code"],
                "description": desc_summary,
                "description_full": desc_raw if len(desc_raw) > len(desc_summary) + 20 else None,
                "general_measures": gm_summary,
                "general_measures_full": gm_raw if len(gm_raw) > len(gm_summary) + 20 else None,
                "medicines": medicines,
                "danger_signs": danger_signs,
                "referral_criteria": referral_criteria,
                "clinical_tables": clinical_tables,
            }

            # ── Raw sections fallback (kept for compatibility) ──
            rows = await conn.fetch("""
                SELECT section_role, chunk_text
                FROM knowledge_chunks
                WHERE condition_id = $1
                  AND NOT (section_role = 'DOSING_TABLE' AND is_table = true)
                  AND section_role != 'CLINICAL_PRESENTATION'
                ORDER BY
                    CASE section_role
                        WHEN 'MANAGEMENT' THEN 1
                        WHEN 'DOSING_TABLE' THEN 2
                        WHEN 'DANGER_SIGNS' THEN 3
                        WHEN 'REFERRAL' THEN 4
                    END
            """, cond_id)

            heading_map = {
                "MANAGEMENT": "General Measures",
                "DOSING_TABLE": "Medicine Treatment",
                "DANGER_SIGNS": "Danger Signs",
                "REFERRAL": "Referral Criteria",
            }
            sections = []
            for r in rows:
                text = r["chunk_text"]
                if not text or len(text) < 30:
                    continue
                sections.append({
                    "heading": heading_map.get(r["section_role"], r["section_role"]),
                    "section_ref": condition["stg_code"],
                    "body": text,
                    "source_name": "SA PHC STG 2024 8th Edition",
                })

        return {
            "condition_name": condition["name"],
            "condition_code": condition["stg_code"],
            "structured": structured,
            "sections": sections,
        }

    except Exception as e:
        logger.error(f"Guidelines lookup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _parse_bullet_list(raw: str) -> list[str]:
    """Parse raw STG text into a list of individual items."""
    if not raw or not raw.strip():
        return []
    import re
    text = re.sub(r"\s*LoE:\s*[A-Za-z0-9]+", "", raw.strip())
    text = re.sub(r"^[»►•●]\s*", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"^[–—]\s+", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+[.)]\s+", "- ", text, flags=re.MULTILINE)
    items = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        if line and len(line) > 3:
            items.append(line)
    return items


# ── POST /api/prescribing/recommended-drugs ────────────────────────────────

@app.post("/api/prescribing/recommended-drugs")
async def recommended_drugs(request: RecommendedDrugsRequest):
    """Get STG-recommended medicines for a condition, ordered by treatment line."""
    try:
        async with app.state.pool.acquire() as conn:
            condition = await conn.fetchrow(
                "SELECT id, name, stg_code FROM conditions WHERE name ILIKE $1 LIMIT 1",
                f"%{request.condition_name}%",
            )
            if not condition:
                return {"condition_name": request.condition_name, "drugs": []}

            rows = await conn.fetch("""
                SELECT m.name, m.routes, m.adult_dose, m.adult_frequency, m.adult_duration,
                       m.paediatric_dose_mg_per_kg, m.paediatric_frequency,
                       m.pregnancy_safe, m.schedule,
                       cm.treatment_line, cm.dose_context, cm.age_group, cm.special_notes
                FROM condition_medicines cm
                JOIN medicines m ON m.id = cm.medicine_id
                WHERE cm.condition_id = $1
                ORDER BY
                    CASE cm.treatment_line
                        WHEN 'first_line' THEN 1
                        WHEN 'second_line' THEN 2
                        WHEN 'alternative' THEN 3
                        WHEN 'adjunct' THEN 4
                        ELSE 5
                    END,
                    m.name
            """, condition["id"])

            drugs = []
            for r in rows:
                routes = r["routes"] or []
                drug = {
                    "name": r["name"],
                    "route": routes[0] if routes else "oral",
                    "routes": routes,
                    "treatment_line": r["treatment_line"] or "first_line",
                    "dose_context": r["dose_context"] or "",
                    "adult_dose": r["adult_dose"] or "",
                    "adult_frequency": r["adult_frequency"] or "",
                    "adult_duration": r["adult_duration"] or "",
                    "age_group": r["age_group"] or "all",
                    "special_notes": r["special_notes"] or "",
                    "pregnancy_safe": r["pregnancy_safe"],
                    "schedule": r["schedule"],
                }
                # Add paediatric dosing if patient is a child
                if request.patient_age and request.patient_age < 18:
                    drug["paediatric_dose_mg_per_kg"] = r["paediatric_dose_mg_per_kg"]
                    drug["paediatric_frequency"] = r["paediatric_frequency"] or ""
                drugs.append(drug)

        return {
            "condition_name": condition["name"],
            "condition_code": condition["stg_code"],
            "drugs": drugs,
        }

    except Exception as e:
        logger.error(f"Recommended drugs failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Phase II Clinician Survey Endpoints ────────────────────────────────────

@app.get("/api/vignettes")
async def list_all_vignettes(active_only: bool = True):
    """List all clinical vignettes (for survey selection)."""
    try:
        async with app.state.pool.acquire() as conn:
            vignettes = await list_vignettes(conn, active_only=active_only)
        # Parse JSONB strings back to dicts for response
        for v in vignettes:
            if isinstance(v.get("vitals"), str):
                v["vitals"] = json.loads(v["vitals"])
            if isinstance(v.get("core_history"), str):
                v["core_history"] = json.loads(v["core_history"])
        return {"vignettes": vignettes, "count": len(vignettes)}
    except Exception as e:
        logger.error(f"List vignettes failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/vignettes")
async def create_new_vignette(request: CreateVignetteRequest):
    """Create a new clinical vignette (admin/Tasleem)."""
    try:
        async with app.state.pool.acquire() as conn:
            vignette = await create_vignette(conn, request.model_dump())
        return {"vignette": vignette, "message": "Vignette created"}
    except asyncpg.UniqueViolationError:
        raise HTTPException(status_code=409, detail=f"Vignette code '{request.vignette_code}' already exists")
    except Exception as e:
        logger.error(f"Create vignette failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/vignettes/{vignette_id}")
async def get_single_vignette(vignette_id: int):
    """Get a single vignette by ID (for clinician to fill in)."""
    try:
        async with app.state.pool.acquire() as conn:
            vignette = await get_vignette(conn, vignette_id)
        if not vignette:
            raise HTTPException(status_code=404, detail="Vignette not found")
        # Parse JSONB and strip expected answers (clinicians shouldn't see them)
        if isinstance(vignette.get("vitals"), str):
            vignette["vitals"] = json.loads(vignette["vitals"])
        if isinstance(vignette.get("core_history"), str):
            vignette["core_history"] = json.loads(vignette["core_history"])
        vignette.pop("expected_conditions", None)
        vignette.pop("expected_acuity", None)
        vignette.pop("expected_sats_colour", None)
        return vignette
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get vignette failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/vignettes/{vignette_id}/respond")
async def submit_vignette_response(vignette_id: int, request: SubmitResponseRequest):
    """Submit a clinician or CareMate response to a vignette."""
    try:
        async with app.state.pool.acquire() as conn:
            # Verify vignette exists
            vignette = await get_vignette(conn, vignette_id)
            if not vignette:
                raise HTTPException(status_code=404, detail="Vignette not found")

            # Serialize nested models to dicts for JSON storage
            data = request.model_dump()
            data["differential_diagnosis"] = [d.model_dump() if hasattr(d, 'model_dump') else d for d in request.differential_diagnosis]
            data["investigations"] = [i.model_dump() if hasattr(i, 'model_dump') else i for i in request.investigations]
            data["treatment_plan"] = [t.model_dump() if hasattr(t, 'model_dump') else t for t in request.treatment_plan]

            response = await save_vignette_response(conn, vignette_id, data)
        return {"response": response, "message": "Response recorded"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Submit response failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/vignettes/{vignette_id}/results")
async def get_vignette_results(vignette_id: int):
    """Get comparison of clinician vs CareMate responses for a vignette."""
    try:
        async with app.state.pool.acquire() as conn:
            comparison = await get_vignette_comparison(conn, vignette_id)
        if not comparison:
            raise HTTPException(status_code=404, detail="Vignette not found")

        # Parse JSONB strings in vignette
        v = comparison["vignette"]
        for field in ("vitals", "core_history", "expected_conditions"):
            if isinstance(v.get(field), str):
                v[field] = json.loads(v[field])

        # Parse JSONB strings in responses
        for resp in comparison["clinician_responses"] + comparison["caremate_responses"]:
            for field in ("differential_diagnosis", "investigations", "treatment_plan", "red_flags_identified"):
                if isinstance(resp.get(field), str):
                    resp[field] = json.loads(resp[field])

        return comparison
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get vignette results failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/vignettes/{vignette_id}/run-caremate")
async def run_caremate_on_vignette(vignette_id: int):
    """Run CareMate's triage agent on a vignette and save the result as a response."""
    try:
        async with app.state.pool.acquire() as conn:
            vignette = await get_vignette(conn, vignette_id)
        if not vignette:
            raise HTTPException(status_code=404, detail="Vignette not found")

        # Parse vignette fields
        vitals_raw = vignette.get("vitals")
        if isinstance(vitals_raw, str):
            vitals_raw = json.loads(vitals_raw)
        history_raw = vignette.get("core_history")
        if isinstance(history_raw, str):
            history_raw = json.loads(history_raw)

        patient = {}
        if vignette.get("patient_age"):
            patient["age"] = vignette["patient_age"]
        if vignette.get("patient_sex"):
            patient["sex"] = vignette["patient_sex"]
        if vignette.get("pregnancy_status"):
            patient["pregnancy_status"] = vignette["pregnancy_status"]

        # Run triage agent
        result = await app.state.agent.analyze(
            complaint=vignette["complaint"],
            patient=patient or None,
            vitals=vitals_raw or None,
            core_history=history_raw or None,
        )

        # Convert CareMate output to vignette response format
        differential = []
        for i, cond in enumerate(result.get("conditions", [])):
            differential.append({
                "rank": i + 1,
                "condition_name": cond.get("condition_name", ""),
                "condition_code": cond.get("condition_code", ""),
                "confidence": cond.get("confidence", 0),
                "reasoning": cond.get("reasoning", ""),
            })

        response_data = {
            "respondent_type": "caremate",
            "respondent_name": "caremate_v1.3",
            "differential_diagnosis": differential,
            "triage_level": result.get("acuity"),
            "sats_colour": result.get("sats_colour"),
            "red_flags_identified": result.get("acuity_reasons", []),
            "notes": f"Extracted symptoms: {', '.join(result.get('extracted_symptoms', []))}",
        }

        async with app.state.pool.acquire() as conn:
            saved = await save_vignette_response(conn, vignette_id, response_data)

        return {
            "response": saved,
            "caremate_raw": result,
            "message": "CareMate assessment saved",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Run CareMate on vignette failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
