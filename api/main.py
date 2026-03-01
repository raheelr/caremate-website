"""
CareMate Clinical Triage API
-----------------------------
FastAPI backend replacing Supabase Edge Functions.

Endpoints:
  POST /api/triage/analyze       — initial triage analysis
  POST /api/triage/refine        — iterative refinement with answers
  POST /api/triage/enrich        — protocol enrichment for a condition
  POST /api/rag/query            — RAG-powered clinical Q&A
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
    CreateVignetteRequest,
    SubmitResponseRequest,
)
from agents.triage_agent import TriageAgent
from safety.checker import SafetyChecker
from db.database import (
    get_condition_detail, get_condition_by_stg_code, search_knowledge_chunks,
    create_vignette, list_vignettes, get_vignette, save_vignette_response,
    get_vignette_responses, get_vignette_comparison,
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
    allow_origins=[
        "https://caremateaihealth.lovable.app",  # Production Lovable
        "https://preview--caremateaihealth.lovable.app",  # Lovable preview
        "http://localhost:5173",                  # Local Vite dev
        "http://localhost:3000",                  # Local alt dev
        "https://caremate-backend-v2-production.up.railway.app",  # Railway
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Key Authentication ────────────────────────────────────────────────────

API_KEY = os.getenv("API_KEY")

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    # Health check is public (Railway healthcheck needs it)
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


# ── POST /api/prescribing/suggest-dosing ─────────────────────────────────────

@app.post("/api/prescribing/suggest-dosing")
async def suggest_dosing(request: DosingRequest):
    """Get dosing suggestion for a drug + condition pair."""
    try:
        async with app.state.pool.acquire() as conn:
            # Look up medicine with condition-specific dosing
            dosing = await conn.fetchrow("""
                SELECT m.name, m.adult_dose, m.adult_frequency, m.adult_duration,
                       m.paediatric_dose_mg_per_kg, m.paediatric_frequency,
                       m.contraindications, m.pregnancy_safe, m.pregnancy_notes,
                       cm.dose_context, cm.treatment_line, cm.age_group, cm.special_notes
                FROM medicines m
                LEFT JOIN condition_medicines cm ON cm.medicine_id = m.id
                LEFT JOIN conditions c ON c.id = cm.condition_id
                WHERE m.name ILIKE $1
                AND (c.name ILIKE $2 OR c.stg_code = $2)
                LIMIT 1
            """, f"%{request.drugName}%", f"%{request.conditionName}%")

        if dosing:
            # Build structured suggestion from DB
            parts = [f"**{dosing['name'].title()}**"]
            if dosing["dose_context"]:
                parts.append(f"Dose: {dosing['dose_context']}")
            elif dosing["adult_dose"]:
                freq = dosing["adult_frequency"] or ""
                dur = dosing["adult_duration"] or ""
                parts.append(f"Dose: {dosing['adult_dose']} {freq} {dur}".strip())
            if dosing["treatment_line"]:
                parts.append(f"({dosing['treatment_line'].replace('_', ' ')})")
            if dosing["special_notes"]:
                parts.append(f"Note: {dosing['special_notes']}")
            if request.patientAge and request.patientAge < 18 and dosing["paediatric_dose_mg_per_kg"]:
                parts.append(f"Paediatric: {dosing['paediatric_dose_mg_per_kg']} mg/kg")
            if dosing["pregnancy_safe"] is False:
                parts.append("**WARNING: Not safe in pregnancy**")
            return {"suggestion": " | ".join(parts)}

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
            max_tokens=1024,
            temperature=0,
            messages=[{"role": "user", "content": (
                f"From the STG text below, provide dosing for {request.drugName} "
                f"for {request.conditionName}.\n"
                f"Patient: age={request.patientAge}, sex={request.patientSex}\n\n"
                f"STG text:\n{context}\n\n"
                f"Provide a concise dosing recommendation. If the drug is not found in the text, say so."
            )}],
        )
        return {"suggestion": response.content[0].text}

    except Exception as e:
        logger.error(f"Dosing suggestion failed: {e}", exc_info=True)
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
