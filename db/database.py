"""
Database Layer
--------------
Handles all PostgreSQL connections and data persistence.
Uses asyncpg for async database access.

Setup: You need a PostgreSQL database with pgvector extension.
Connection string goes in .env file as DATABASE_URL
"""

import os
import json
import asyncio
import asyncpg
from typing import Optional
from datetime import datetime


# ── Connection ───────────────────────────────────────────────────────────────

async def get_connection() -> asyncpg.Connection:
    """Get a database connection from the environment."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError(
            "DATABASE_URL environment variable not set.\n"
            "Add it to your .env file:\n"
            "DATABASE_URL=postgresql://user:password@host:5432/caremate"
        )
    return await asyncpg.connect(database_url)


async def get_pool() -> asyncpg.Pool:
    """Get a connection pool for concurrent access."""
    database_url = os.getenv("DATABASE_URL")
    return await asyncpg.create_pool(database_url, min_size=2, max_size=10)


# ── Schema setup ─────────────────────────────────────────────────────────────

async def create_schema(conn: asyncpg.Connection):
    """Create all tables from schema.sql. Safe to run if tables already exist."""
    schema_path = os.path.join(os.path.dirname(__file__), '..', 'db', 'schema.sql')
    with open(schema_path) as f:
        schema_sql = f.read()
    try:
        await conn.execute(schema_sql)
        print("✅ Database schema created")
    except Exception as e:
        if "already exists" in str(e):
            print("✅ Database schema ready (tables already exist)")
        else:
            raise


# ── Condition saving ─────────────────────────────────────────────────────────

async def save_condition(
    conn: asyncpg.Connection,
    extraction: dict
) -> int:
    """
    Save an extracted condition and all its related data.
    Returns the condition ID.
    """
    
    # 1. Save the condition itself
    condition_id = await conn.fetchval("""
        INSERT INTO conditions (
            stg_code,
            icd10_codes,
            name,
            chapter,
            chapter_name,
            description_text,
            general_measures,
            medicine_treatment,
            danger_signs,
            referral_criteria,
            source_pages,
            extraction_confidence,
            ambiguity_flags,
            needs_review,
            applies_to_children,
            applies_to_adults,
            applies_to_pregnant
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            $11, $12, $13, $14, $15, $16, $17
        )
        ON CONFLICT (stg_code) DO UPDATE SET
            name = EXCLUDED.name,
            icd10_codes = EXCLUDED.icd10_codes,
            extraction_confidence = EXCLUDED.extraction_confidence,
            ambiguity_flags = EXCLUDED.ambiguity_flags,
            needs_review = EXCLUDED.needs_review,
            updated_at = NOW()
        RETURNING id
    """,
        extraction.get('stg_code'),
        extraction.get('icd10_codes', []),
        extraction.get('condition_name_normalised', ''),
        extraction.get('chapter_number'),
        extraction.get('chapter_name', ''),
        extraction.get('sections', {}).get('description', ''),
        extraction.get('sections', {}).get('general_measures', ''),
        extraction.get('sections', {}).get('medicine_treatment', ''),
        extraction.get('sections', {}).get('danger_signs', ''),
        json.dumps(extraction.get('referral_criteria', [])),
        extraction.get('source_pages', []),
        1.0 - extraction.get('ambiguity_flags', {}).get('ambiguity_score', 0),
        json.dumps(extraction.get('ambiguity_flags', {})),
        bool(extraction.get('needs_review', False)),
        bool(extraction.get('applies_to_children', True)),
        bool(extraction.get('applies_to_adults', True)),
        None if extraction.get('applies_to_pregnant') in (None, 'null', 'None') else bool(extraction.get('applies_to_pregnant')),
    )
    
    # 2. Save clinical entities and relationships
    for feature in extraction.get('clinical_features', []):
        await _save_clinical_feature(conn, condition_id, feature)
    
    for danger_sign in extraction.get('danger_signs', []):
        await _save_danger_sign(conn, condition_id, danger_sign)
    
    # 3. Save medicines
    for med in extraction.get('medicines', []):
        await _save_medicine(conn, condition_id, med)
    
    # 4. Save synonym suggestions
    for synonym in extraction.get('patient_language_synonyms', []):
        # Find the canonical term it relates to (use condition name as fallback)
        canonical = extraction.get('condition_name_normalised', '').lower()
        await _save_synonym(conn, canonical, synonym)
    
    # 5. Save prerequisite context
    for prereq in extraction.get('prerequisite_context', []):
        await conn.execute("""
            INSERT INTO condition_prerequisites (condition_id, prerequisite)
            VALUES ($1, $2)
            ON CONFLICT DO NOTHING
        """, condition_id, prereq)
    
    # 6. Save raw text as knowledge chunks (for vector search later)
    await _save_knowledge_chunks(conn, condition_id, extraction)
    
    return condition_id


async def _save_clinical_feature(
    conn: asyncpg.Connection,
    condition_id: int,
    feature: dict
):
    """Save a clinical feature as an entity + relationship."""
    feature_name = feature.get('feature', '').lower().strip()
    if not feature_name:
        return
    
    # Upsert the entity
    entity_id = await conn.fetchval("""
        INSERT INTO clinical_entities (canonical_name, entity_type, aliases)
        VALUES ($1, 'SYMPTOM', $2)
        ON CONFLICT (canonical_name) DO UPDATE SET
            entity_type = EXCLUDED.entity_type
        RETURNING id
    """, feature_name, [])
    
    # Get condition entity (or create it)
    condition_entity_id = await _get_or_create_condition_entity(conn, condition_id)
    
    # Determine relationship type
    source_section = feature.get('source_section', 'DESCRIPTION')
    rel_type = 'RED_FLAG' if source_section == 'DANGER_SIGNS' else 'INDICATES'
    
    # Map feature type to weight
    feature_type_map = {
        'diagnostic_feature': 'diagnostic_feature',
        'presenting_feature': 'presenting_feature',
        'associated_feature': 'associated_feature',
    }
    feature_type = feature_type_map.get(feature.get('feature_type'), 'associated_feature')
    
    await conn.execute("""
        INSERT INTO clinical_relationships (
            source_entity_id, target_entity_id,
            relationship_type, feature_type,
            condition_id, source_section, confidence
        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT DO NOTHING
    """,
        entity_id, condition_entity_id,
        rel_type, feature_type,
        condition_id, source_section, 1.0
    )


async def _save_danger_sign(
    conn: asyncpg.Connection,
    condition_id: int,
    danger_sign: dict
):
    """Save a danger sign — these are RED_FLAG edges."""
    sign_name = danger_sign.get('sign', '').lower().strip()
    if not sign_name:
        return
    
    entity_id = await conn.fetchval("""
        INSERT INTO clinical_entities (canonical_name, entity_type, aliases)
        VALUES ($1, 'SYMPTOM', $2)
        ON CONFLICT (canonical_name) DO UPDATE SET entity_type = EXCLUDED.entity_type
        RETURNING id
    """, sign_name, [])
    
    condition_entity_id = await _get_or_create_condition_entity(conn, condition_id)
    
    await conn.execute("""
        INSERT INTO clinical_relationships (
            source_entity_id, target_entity_id,
            relationship_type, feature_type,
            condition_id, source_section, confidence
        ) VALUES ($1, $2, 'RED_FLAG', 'diagnostic_feature', $3, 'DANGER_SIGNS', 1.0)
        ON CONFLICT DO NOTHING
    """, entity_id, condition_entity_id, condition_id)


async def _save_medicine(
    conn: asyncpg.Connection,
    condition_id: int,
    med: dict
):
    """Save a medicine and its link to this condition."""
    med_name = med.get('name', '').strip()
    if not med_name:
        return
    
    # Upsert medicine
    med_id = await conn.fetchval("""
        INSERT INTO medicines (
            name, routes, adult_dose, paediatric_dose_mg_per_kg,
            adult_frequency, adult_duration, source_page
        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (name) DO UPDATE SET
            routes = EXCLUDED.routes
        RETURNING id
    """,
        med_name.lower(),
        [med.get('route', '')] if med.get('route') else [],
        med.get('dose_adults'),
        None,  # We'll extract mg/kg separately
        med.get('frequency'),
        med.get('duration'),
        None
    )
    
    # Link to condition
    await conn.execute("""
        INSERT INTO condition_medicines (
            condition_id, medicine_id, treatment_line,
            dose_context, age_group, special_notes
        ) VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT DO NOTHING
    """,
        condition_id, med_id,
        med.get('treatment_line', 'first_line'),
        f"{med.get('dose_adults', '')} {med.get('frequency', '')} {med.get('duration', '')}".strip(),
        med.get('age_group', 'all'),
        med.get('special_notes', '')
    )


async def _save_synonym(conn: asyncpg.Connection, canonical: str, synonym: str):
    """Save a patient language synonym."""
    if not canonical or not synonym:
        return
    await conn.execute("""
        INSERT INTO synonym_rings (canonical_term, synonym)
        VALUES ($1, $2)
        ON CONFLICT (canonical_term, synonym) DO NOTHING
    """, canonical.lower(), synonym.lower())


async def _save_knowledge_chunks(
    conn: asyncpg.Connection,
    condition_id: int,
    extraction: dict
):
    """Save text chunks for vector search (embedding added later)."""
    sections = extraction.get('sections', {})
    
    section_role_map = {
        'description': 'CLINICAL_PRESENTATION',
        'danger_signs': 'DANGER_SIGNS',
        'general_measures': 'MANAGEMENT',
        'medicine_treatment': 'DOSING_TABLE',
        'referral': 'REFERRAL',
    }
    
    for section_key, role in section_role_map.items():
        text = sections.get(section_key, '').strip()
        if not text or len(text) < 20:
            continue
        
        await conn.execute("""
            INSERT INTO knowledge_chunks (condition_id, chunk_text, section_role, source_page)
            VALUES ($1, $2, $3, $4)
        """,
            condition_id,
            text,
            role,
            extraction.get('source_pages', [None])[0]
        )


async def _get_or_create_condition_entity(
    conn: asyncpg.Connection,
    condition_id: int
) -> int:
    """Get or create a CONDITION entity for relationship targets."""
    # Look up condition name
    name = await conn.fetchval(
        "SELECT name FROM conditions WHERE id = $1", condition_id
    )
    if not name:
        return None
    
    return await conn.fetchval("""
        INSERT INTO clinical_entities (canonical_name, entity_type, aliases)
        VALUES ($1, 'CONDITION', $2)
        ON CONFLICT (canonical_name) DO UPDATE SET entity_type = EXCLUDED.entity_type
        RETURNING id
    """, name.lower(), [])


# ── Ingestion run tracking ───────────────────────────────────────────────────

async def start_ingestion_run(conn: asyncpg.Connection, source_file: str) -> int:
    """Record the start of an ingestion run."""
    return await conn.fetchval("""
        INSERT INTO ingestion_runs (source_file, status)
        VALUES ($1, 'running')
        RETURNING id
    """, source_file)


async def complete_ingestion_run(
    conn: asyncpg.Connection,
    run_id: int,
    conditions_extracted: int,
    conditions_needing_review: int
):
    """Mark an ingestion run as complete."""
    await conn.execute("""
        UPDATE ingestion_runs SET
            completed_at = NOW(),
            conditions_extracted = $2,
            conditions_needing_review = $3,
            status = 'complete'
        WHERE id = $1
    """, run_id, conditions_extracted, conditions_needing_review)


# ── Query helpers (used by agents later) ─────────────────────────────────────

async def get_conditions_for_symptoms(
    conn: asyncpg.Connection,
    symptom_names: list[str],
    patient_is_child: bool = False,
    patient_is_pregnant: bool = False,
    limit: int = 10
) -> list[dict]:
    """
    Find conditions that match a list of symptoms.
    Returns ranked list with match counts.
    """
    rows = await conn.fetch("""
        SELECT
            c.id,
            c.stg_code,
            c.name,
            c.chapter_name,
            c.extraction_confidence,
            COUNT(DISTINCT cr.id) as match_count,
            SUM(CASE
                WHEN cr.feature_type = 'diagnostic_feature' THEN 0.18
                WHEN cr.feature_type = 'presenting_feature' THEN 0.12
                ELSE 0.08
            END) as raw_score,
            ARRAY_AGG(DISTINCT e.canonical_name) as matched_features
        FROM conditions c
        JOIN clinical_relationships cr ON cr.condition_id = c.id
        JOIN clinical_entities e ON e.id = cr.source_entity_id
        WHERE e.canonical_name = ANY($1::text[])
        AND cr.relationship_type IN ('INDICATES', 'RED_FLAG')
        AND ($2 = FALSE OR c.applies_to_children = TRUE)
        AND ($3 = FALSE OR c.applies_to_pregnant IS NOT FALSE)
        GROUP BY c.id, c.stg_code, c.name, c.chapter_name, c.extraction_confidence
        ORDER BY raw_score DESC
        LIMIT $4
    """, symptom_names, patient_is_child, patient_is_pregnant, limit)
    
    return [dict(r) for r in rows]


async def get_condition_detail(
    conn: asyncpg.Connection,
    condition_id: int
) -> Optional[dict]:
    """Get full condition detail including treatment."""
    row = await conn.fetchrow("""
        SELECT c.*,
            COALESCE(
                (SELECT json_agg(json_build_object(
                    'name', m.name,
                    'route', m.routes[1],
                    'dose_context', cm.dose_context,
                    'treatment_line', cm.treatment_line,
                    'age_group', cm.age_group,
                    'special_notes', cm.special_notes
                ))
                FROM condition_medicines cm
                JOIN medicines m ON m.id = cm.medicine_id
                WHERE cm.condition_id = c.id
                ), '[]'::json
            ) as medicines_json
        FROM conditions c
        WHERE c.id = $1
    """, condition_id)
    
    return dict(row) if row else None


async def resolve_to_canonical(
    conn: asyncpg.Connection,
    terms: list[str],
    limit_per_term: int = 5,
) -> dict:
    """
    Resolve extracted symptom terms to canonical entity names in the DB.
    Uses three strategies:
      1. Exact match on clinical_entities
      2. Trigram similarity (pg_trgm) for close matches
      3. Word overlap for semantic bridges (e.g. "sore throat" → "painful red throat")

    Returns {original_term: [list of canonical matches]}.
    """
    stop_words = {
        'with', 'that', 'from', 'this', 'have', 'been', 'more', 'than',
        'very', 'and', 'the', 'for', 'not', 'but', 'are', 'was', 'has',
        'also', 'only', 'some', 'when', 'into', 'over', 'such',
    }

    results = {}
    for term in terms:
        term_lower = term.lower().strip()
        if not term_lower:
            continue

        matches = set()

        # 1. Exact match
        exact = await conn.fetchval("""
            SELECT canonical_name FROM clinical_entities
            WHERE canonical_name = $1 AND entity_type = 'SYMPTOM'
        """, term_lower)
        if exact:
            matches.add(exact)

        # 2. Trigram similarity (threshold 0.25 to catch partial matches)
        trig_rows = await conn.fetch("""
            SELECT canonical_name, similarity(canonical_name, $1) as sim
            FROM clinical_entities
            WHERE entity_type = 'SYMPTOM'
            AND similarity(canonical_name, $1) > 0.25
            ORDER BY sim DESC
            LIMIT $2
        """, term_lower, limit_per_term)
        for r in trig_rows:
            matches.add(r["canonical_name"])

        # 3. Word overlap — extract significant words, find entities containing them
        #    Only used when trigram found < 3 matches (avoids over-expanding well-matched terms)
        if len(matches) < 3:
            words = [w for w in term_lower.split() if len(w) > 3 and w not in stop_words]
            for word in words[:3]:
                word_rows = await conn.fetch("""
                    SELECT canonical_name
                    FROM clinical_entities
                    WHERE entity_type = 'SYMPTOM'
                    AND canonical_name ILIKE $1
                    AND LENGTH(canonical_name) < 80
                    LIMIT $2
                """, f"%{word}%", limit_per_term)
                for r in word_rows:
                    matches.add(r["canonical_name"])

        results[term_lower] = sorted(matches)

    return results


async def get_red_flag_matches(
    conn: asyncpg.Connection,
    symptom_names: list[str],
) -> list[dict]:
    """Find which symptoms match RED_FLAG relationships across all conditions."""
    rows = await conn.fetch("""
        SELECT DISTINCT e.canonical_name, c.id as condition_id, c.name as condition_name
        FROM clinical_relationships cr
        JOIN clinical_entities e ON e.id = cr.source_entity_id
        JOIN conditions c ON c.id = cr.condition_id
        WHERE e.canonical_name = ANY($1::text[])
        AND cr.relationship_type = 'RED_FLAG'
    """, [s.lower() for s in symptom_names])
    return [dict(r) for r in rows]


async def get_condition_red_flags(
    conn: asyncpg.Connection,
    condition_id: int,
) -> list[dict]:
    """Get all RED_FLAG features for a specific condition."""
    rows = await conn.fetch("""
        SELECT e.canonical_name, cr.feature_type
        FROM clinical_relationships cr
        JOIN clinical_entities e ON e.id = cr.source_entity_id
        WHERE cr.condition_id = $1
        AND cr.relationship_type = 'RED_FLAG'
    """, condition_id)
    return [dict(r) for r in rows]


async def get_condition_prerequisites(
    conn: asyncpg.Connection,
    condition_id: int,
) -> list[dict]:
    """Get prerequisites for a condition (e.g. 'hiv_positive')."""
    rows = await conn.fetch("""
        SELECT prerequisite, description
        FROM condition_prerequisites
        WHERE condition_id = $1
    """, condition_id)
    return [dict(r) for r in rows]


async def get_condition_by_stg_code(
    conn: asyncpg.Connection,
    stg_code: str,
) -> Optional[dict]:
    """Look up a condition by its STG code."""
    row = await conn.fetchrow(
        "SELECT * FROM conditions WHERE stg_code = $1", stg_code
    )
    return dict(row) if row else None


async def search_knowledge_chunks(
    conn: asyncpg.Connection,
    query: str,
    condition_id: Optional[int] = None,
    section_role: Optional[str] = None,
    limit: int = 5,
) -> list[dict]:
    """Search knowledge chunks by text similarity."""
    if condition_id:
        rows = await conn.fetch("""
            SELECT kc.*, c.stg_code, c.name as condition_name
            FROM knowledge_chunks kc
            JOIN conditions c ON c.id = kc.condition_id
            WHERE kc.condition_id = $1
            AND ($2::text IS NULL OR kc.section_role = $2)
            ORDER BY kc.section_role
            LIMIT $3
        """, condition_id, section_role, limit)
    else:
        rows = await conn.fetch("""
            SELECT kc.*, c.stg_code, c.name as condition_name
            FROM knowledge_chunks kc
            JOIN conditions c ON c.id = kc.condition_id
            WHERE kc.chunk_text ILIKE $1
            AND ($2::text IS NULL OR kc.section_role = $2)
            LIMIT $3
        """, f"%{query}%", section_role, limit)
    return [dict(r) for r in rows]
