"""
Pydantic models matching the Lovable frontend API contracts.
Field names must match the frontend exactly (camelCase vitals, etc).
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Shared sub-models ────────────────────────────────────────────────────────

class PatientContext(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None                    # "female" | "male"
    pregnancy_status: Optional[str] = None       # "pregnant" | "not_pregnant" | "unknown"


class Vitals(BaseModel):
    systolic: Optional[float] = None
    diastolic: Optional[float] = None
    heartRate: Optional[float] = None
    temperature: Optional[float] = None
    respiratoryRate: Optional[float] = None
    oxygenSat: Optional[float] = None


class CoreHistory(BaseModel):
    onset: Optional[str] = None                  # "Sudden onset" | "Gradual onset" | "< 24 hours" etc.
    recurrence: Optional[str] = None             # "yes" | "no" | "unknown"
    medications: Optional[str] = None            # free text


class ConditionResult(BaseModel):
    condition_code: str                          # stg_code e.g. "1.2"
    condition_name: str
    confidence: float                            # 0.0-1.0
    matched_symptoms: list[str] = []
    reasoning: str = ""
    source_references: list[str] = []


class AssessmentQuestion(BaseModel):
    id: str
    question: str
    type: str = "yes_no"                         # "yes_no" | "text"
    required: Optional[bool] = None
    round: Optional[int] = None
    source_citation: Optional[str] = None        # e.g. "Standard: STG tonsillitis"
    grounding: Optional[str] = None              # "verified" | "prerequisite" | "unverified"


class ConditionSymptom(BaseModel):
    id: str
    question: str


# ── POST /api/triage/analyze ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    complaint: str
    patient: Optional[PatientContext] = None
    core_history: Optional[CoreHistory] = None
    vitals: Optional[Vitals] = None


class AnalyzeResponse(BaseModel):
    extracted_symptoms: list[str]
    acuity: str                                  # "routine" | "priority" | "urgent"
    acuity_reasons: list[str] = []
    acuity_sources: list[str] = []
    conditions: list[ConditionResult] = []
    condition_symptoms: dict = {}                # {condition_name: [{id, question}]}
    needs_assessment: bool = True
    assessment_questions: list[AssessmentQuestion] = []


# ── POST /api/triage/refine ──────────────────────────────────────────────────

class RefineRequest(BaseModel):
    complaint: str
    patient: Optional[PatientContext] = None
    conditions: list[ConditionResult] = []
    core_history: Optional[CoreHistory] = None
    questions: list[dict] = []
    answers: dict = {}
    current_round: int = 1
    request_next_round: bool = False
    stg_feature_data: Optional[dict] = None
    all_time_answers: dict = {}
    all_time_questions: list[dict] = []


class RefineResponse(BaseModel):
    refinement_source: Optional[str] = None      # "rules" if deterministic scoring
    conditions: Optional[list[ConditionResult]] = None
    next_round_questions: Optional[list[AssessmentQuestion]] = None
    red_flag_alert: Optional[str] = None


# ── POST /api/triage/enrich ──────────────────────────────────────────────────

class EnrichRequest(BaseModel):
    condition_code: str
    condition_name: str
    guideline: Optional[str] = None
    triage_context: Optional[dict] = None


class EnrichResponse(BaseModel):
    prompts: list[dict] = []
    sources: list[str] = []


# ── POST /api/rag/query ─────────────────────────────────────────────────────

class RAGQueryRequest(BaseModel):
    query: str
    framework: str = "STG"                       # "APC" | "NICE" | "STG"
    condition_code: Optional[str] = None
    patient_context: Optional[dict] = None
    max_chunks: int = 8
    stream: bool = False


class RAGQueryResponse(BaseModel):
    answer: str
    sources: list[dict] = []
    graph: dict = Field(default_factory=lambda: {"entities": [], "paths": []})
    metadata: dict = {}
    error: Optional[str] = None


# ── POST /api/prescribing/suggest-dosing ─────────────────────────────────────

class DosingRequest(BaseModel):
    drugName: str
    conditionName: str
    patientAge: Optional[int] = None
    patientSex: Optional[str] = None
    selectedTreatmentText: Optional[str] = None
    knowledgeSource: str = "STG"


class DosingResponse(BaseModel):
    suggestion: str
