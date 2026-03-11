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
            description_text = CASE WHEN LENGTH(EXCLUDED.description_text) > LENGTH(COALESCE(conditions.description_text, ''))
                                    THEN EXCLUDED.description_text ELSE conditions.description_text END,
            general_measures = CASE WHEN LENGTH(EXCLUDED.general_measures) > LENGTH(COALESCE(conditions.general_measures, ''))
                                    THEN EXCLUDED.general_measures ELSE conditions.general_measures END,
            medicine_treatment = CASE WHEN LENGTH(EXCLUDED.medicine_treatment) > LENGTH(COALESCE(conditions.medicine_treatment, ''))
                                      THEN EXCLUDED.medicine_treatment ELSE conditions.medicine_treatment END,
            danger_signs = CASE WHEN LENGTH(EXCLUDED.danger_signs) > LENGTH(COALESCE(conditions.danger_signs, ''))
                                THEN EXCLUDED.danger_signs ELSE conditions.danger_signs END,
            referral_criteria = CASE WHEN LENGTH(EXCLUDED.referral_criteria::text) > LENGTH(COALESCE(conditions.referral_criteria::text, '[]'))
                                     THEN EXCLUDED.referral_criteria ELSE conditions.referral_criteria END,
            source_pages = COALESCE(EXCLUDED.source_pages, conditions.source_pages),
            extraction_confidence = GREATEST(EXCLUDED.extraction_confidence, conditions.extraction_confidence),
            ambiguity_flags = EXCLUDED.ambiguity_flags,
            needs_review = EXCLUDED.needs_review,
            applies_to_children = EXCLUDED.applies_to_children,
            applies_to_adults = EXCLUDED.applies_to_adults,
            applies_to_pregnant = COALESCE(EXCLUDED.applies_to_pregnant, conditions.applies_to_pregnant),
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
    """Save text chunks for vector search (embedding added later).
    Sets is_table=TRUE for Docling-extracted table chunks and
    is_algorithm=TRUE for Vision-extracted flowchart chunks.
    """
    sections = extraction.get('sections', {})

    section_role_map = {
        'description': 'CLINICAL_PRESENTATION',
        'danger_signs': 'DANGER_SIGNS',
        'general_measures': 'MANAGEMENT',
        'medicine_treatment': 'DOSING_TABLE',
        'referral': 'REFERRAL',
    }

    source_page = extraction.get('source_pages', [None])[0]

    standard_chunks_saved = 0
    for section_key, role in section_role_map.items():
        text = sections.get(section_key, '').strip()
        if not text or len(text) < 20:
            continue

        await conn.execute("""
            INSERT INTO knowledge_chunks (condition_id, chunk_text, section_role, source_page,
                                          is_table, is_algorithm)
            VALUES ($1, $2, $3, $4, FALSE, FALSE)
        """,
            condition_id,
            text,
            role,
            source_page,
        )
        standard_chunks_saved += 1

    # Fallback: if no standard sections had content, chunk the raw_text
    # This catches conditions where Docling was primary and pdfplumber
    # sections were empty (86 conditions in current extraction)
    if standard_chunks_saved == 0:
        raw_text = extraction.get('raw_text', '').strip()
        if raw_text and len(raw_text) >= 20:
            # Split into ~2000 char chunks to keep them manageable for vector search
            max_chunk = 2000
            if len(raw_text) <= max_chunk:
                await conn.execute("""
                    INSERT INTO knowledge_chunks (condition_id, chunk_text, section_role, source_page,
                                                  is_table, is_algorithm)
                    VALUES ($1, $2, 'CLINICAL_PRESENTATION', $3, FALSE, FALSE)
                """,
                    condition_id,
                    raw_text,
                    source_page,
                )
            else:
                # Split on paragraph boundaries
                paragraphs = raw_text.split('\n\n')
                current_chunk = ''
                chunk_num = 0
                for para in paragraphs:
                    if current_chunk and len(current_chunk) + len(para) + 2 > max_chunk:
                        role = 'CLINICAL_PRESENTATION' if chunk_num == 0 else 'MANAGEMENT'
                        await conn.execute("""
                            INSERT INTO knowledge_chunks (condition_id, chunk_text, section_role, source_page,
                                                          is_table, is_algorithm)
                            VALUES ($1, $2, $3, $4, FALSE, FALSE)
                        """,
                            condition_id,
                            current_chunk.strip(),
                            role,
                            source_page,
                        )
                        current_chunk = para
                        chunk_num += 1
                    else:
                        current_chunk = current_chunk + '\n\n' + para if current_chunk else para
                # Save final chunk
                if current_chunk.strip() and len(current_chunk.strip()) >= 20:
                    role = 'CLINICAL_PRESENTATION' if chunk_num == 0 else 'MANAGEMENT'
                    await conn.execute("""
                        INSERT INTO knowledge_chunks (condition_id, chunk_text, section_role, source_page,
                                                      is_table, is_algorithm)
                        VALUES ($1, $2, $3, $4, FALSE, FALSE)
                    """,
                        condition_id,
                        current_chunk.strip(),
                        role,
                        source_page,
                    )

    # Save Docling table content as a separate chunk with is_table=TRUE
    table_text = sections.get('_tables', '').strip()
    if table_text and len(table_text) >= 20:
        await conn.execute("""
            INSERT INTO knowledge_chunks (condition_id, chunk_text, section_role, source_page,
                                          is_table, is_algorithm)
            VALUES ($1, $2, 'DOSING_TABLE', $3, TRUE, FALSE)
        """,
            condition_id,
            table_text,
            source_page,
        )

    # Save Vision flowchart content as a separate chunk with is_algorithm=TRUE
    vision_text = sections.get('_vision', '').strip()
    if vision_text and len(vision_text) >= 20:
        await conn.execute("""
            INSERT INTO knowledge_chunks (condition_id, chunk_text, section_role, source_page,
                                          is_table, is_algorithm)
            VALUES ($1, $2, 'CLINICAL_PRESENTATION', $3, FALSE, TRUE)
        """,
            condition_id,
            vision_text,
            source_page,
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
    patient_sex: Optional[str] = None,
    patient_age: Optional[int] = None,
    limit: int = 10
) -> list[dict]:
    """
    Find conditions that match a list of symptoms.
    Returns ranked list with match counts.
    Filters by patient demographics (child, pregnant, sex, age).
    """
    rows = await conn.fetch("""
        SELECT
            c.id,
            c.stg_code,
            c.name,
            c.chapter_name,
            c.extraction_confidence,
            c.referral_required,
            c.care_setting,
            c.source_tag,
            c.duration_profile,
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
        AND ($5::text IS NULL
             OR ($5 = 'male' AND c.applies_to_male IS NOT FALSE)
             OR ($5 = 'female' AND c.applies_to_female IS NOT FALSE))
        AND ($6::int IS NULL OR (c.min_age_years <= $6 AND c.max_age_years >= $6))
        GROUP BY c.id, c.stg_code, c.name, c.chapter_name, c.extraction_confidence,
                 c.referral_required, c.care_setting, c.source_tag, c.duration_profile
        ORDER BY raw_score DESC
        LIMIT $4
    """, symptom_names, patient_is_child, patient_is_pregnant, limit,
         patient_sex, patient_age)

    return [dict(r) for r in rows]


async def get_conditions_for_medications(
    conn: asyncpg.Connection,
    medication_names: list[str],
) -> list[dict]:
    """
    Find conditions treated by the patient's current medications.
    Uses the condition_medicines + medicines tables.
    Returns conditions with the matched medication info.
    """
    if not medication_names:
        return []

    rows = await conn.fetch("""
        SELECT DISTINCT c.id, c.stg_code, c.name, c.chapter_name,
               m.name as medicine_name, cm.treatment_line
        FROM condition_medicines cm
        JOIN conditions c ON c.id = cm.condition_id
        JOIN medicines m ON m.id = cm.medicine_id
        WHERE m.name ILIKE ANY($1::text[])
        ORDER BY c.name
    """, [f"%{med.lower().strip()}%" for med in medication_names if med.strip()])

    return [dict(r) for r in rows]


async def get_condition_rich_content(
    conn: asyncpg.Connection,
    condition_id: int,
    max_chunks: int = 4,
    max_chars: int = 8000,
) -> list[dict]:
    """Get table and algorithm chunks for a condition.

    Returns the most relevant rich content chunks (tables, algorithms)
    sorted by priority: algorithms first, then smaller tables.
    Each chunk is capped at max_chars to avoid huge payloads.
    """
    rows = await conn.fetch("""
        SELECT section_role, chunk_text, is_table, is_algorithm,
               length(chunk_text) as chunk_len
        FROM knowledge_chunks
        WHERE condition_id = $1
          AND (is_table = true OR is_algorithm = true)
        ORDER BY
            is_algorithm DESC,              -- algorithms first
            length(chunk_text) ASC          -- shorter (more focused) chunks first
        LIMIT $2
    """, condition_id, max_chunks)

    results = []
    for r in rows:
        text = r["chunk_text"] or ""
        if len(text) > max_chars:
            # Truncate at a line boundary
            text = text[:max_chars]
            last_nl = text.rfind("\n")
            if last_nl > max_chars * 0.7:
                text = text[:last_nl]
            text += "\n..."

        chunk_type = "algorithm" if r["is_algorithm"] else "table"
        # Try to extract a meaningful title (skip noise lines)
        import re as _re
        lines = text.strip().split("\n")
        title = ""
        for line in lines:
            clean = line.strip().lstrip("#").strip()
            if not clean:
                continue
            # Skip noise: page refs, chapter headers, table separators, dashes
            if (clean.startswith("|") or clean.startswith("[Page")
                    or _re.match(r"^CHAPTER\s+\d+", clean)
                    or _re.match(r"^---+$", clean)
                    or clean == "..."):
                continue
            title = clean[:100]
            break

        results.append({
            "title": title,
            "content": text,
            "type": chunk_type,
        })

    return results


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
                    'special_notes', cm.special_notes,
                    'paediatric_dose_mg_per_kg', m.paediatric_dose_mg_per_kg,
                    'paediatric_frequency', m.paediatric_frequency,
                    'paediatric_note', m.paediatric_note,
                    'pregnancy_safe', m.pregnancy_safe,
                    'pregnancy_notes', m.pregnancy_notes
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


async def get_condition_details_batch(
    conn: asyncpg.Connection,
    condition_ids: list[int],
) -> dict[int, dict]:
    """Batch fetch full STG details for multiple conditions in one query."""
    if not condition_ids:
        return {}
    rows = await conn.fetch("""
        SELECT c.*,
            COALESCE(
                (SELECT json_agg(json_build_object(
                    'name', m.name,
                    'route', m.routes[1],
                    'dose_context', cm.dose_context,
                    'treatment_line', cm.treatment_line,
                    'age_group', cm.age_group,
                    'special_notes', cm.special_notes,
                    'paediatric_dose_mg_per_kg', m.paediatric_dose_mg_per_kg,
                    'paediatric_frequency', m.paediatric_frequency,
                    'paediatric_note', m.paediatric_note,
                    'pregnancy_safe', m.pregnancy_safe,
                    'pregnancy_notes', m.pregnancy_notes
                ))
                FROM condition_medicines cm
                JOIN medicines m ON m.id = cm.medicine_id
                WHERE cm.condition_id = c.id
                ), '[]'::json
            ) as medicines_json
        FROM conditions c
        WHERE c.id = ANY($1::int[])
    """, condition_ids)
    return {row["id"]: dict(row) for row in rows}


async def resolve_to_canonical(
    conn: asyncpg.Connection,
    terms: list[str],
    limit_per_term: int = 5,
) -> dict:
    """
    Resolve extracted symptom terms to canonical entity names in the DB.
    Uses three strategies in batch (2 queries max instead of N-per-term):
      1. Exact match + trigram similarity via LATERAL join (single query)
      2. Word overlap for terms with < 3 matches (single query)

    Returns {original_term: [list of canonical matches]}.
    """
    terms_lower = [t.lower().strip() for t in terms if t.strip()]
    if not terms_lower:
        return {}

    results = {t: set() for t in terms_lower}

    # Query 1: Batch exact + trigram for ALL terms via LATERAL join
    rows = await conn.fetch("""
        SELECT t.term, ce.canonical_name
        FROM unnest($1::text[]) AS t(term)
        JOIN LATERAL (
            SELECT canonical_name
            FROM clinical_entities
            WHERE entity_type = 'SYMPTOM'
            AND (canonical_name = t.term OR similarity(canonical_name, t.term) > 0.25)
            ORDER BY similarity(canonical_name, t.term) DESC
            LIMIT $2
        ) ce ON TRUE
    """, terms_lower, limit_per_term)

    for r in rows:
        term = r["term"]
        if term in results:
            results[term].add(r["canonical_name"])

    # Query 2: Word overlap for terms with < 3 matches
    stop_words = {
        'with', 'that', 'from', 'this', 'have', 'been', 'more', 'than',
        'very', 'and', 'the', 'for', 'not', 'but', 'are', 'was', 'has',
        'also', 'only', 'some', 'when', 'into', 'over', 'such',
    }
    overlap_words = []
    word_to_terms: dict[str, list[str]] = {}
    for term in terms_lower:
        if len(results.get(term, set())) < 3:
            words = [w for w in term.split() if len(w) > 3 and w not in stop_words]
            for word in words[:3]:
                overlap_words.append(word)
                word_to_terms.setdefault(word, []).append(term)

    if overlap_words:
        # Build ILIKE patterns for all overlap words in one query
        patterns = [f"%{w}%" for w in set(overlap_words)]
        overlap_rows = await conn.fetch("""
            SELECT canonical_name
            FROM clinical_entities
            WHERE entity_type = 'SYMPTOM'
            AND canonical_name ILIKE ANY($1::text[])
            AND LENGTH(canonical_name) < 80
        """, patterns)

        for r in overlap_rows:
            name = r["canonical_name"]
            name_lower = name.lower()
            for word, parent_terms in word_to_terms.items():
                if word in name_lower:
                    for pt in parent_terms:
                        if len(results[pt]) < limit_per_term + 3:
                            results[pt].add(name)

    return {t: sorted(v) for t, v in results.items()}


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


async def get_condition_prerequisites_batch(
    conn: asyncpg.Connection,
    condition_ids: list[int],
) -> dict[int, list[dict]]:
    """Get prerequisites for multiple conditions in one query.

    Returns {condition_id: [{"prerequisite": ..., "description": ...}, ...]}.
    """
    if not condition_ids:
        return {}
    rows = await conn.fetch("""
        SELECT condition_id, prerequisite, description
        FROM condition_prerequisites
        WHERE condition_id = ANY($1)
    """, condition_ids)
    result: dict[int, list[dict]] = {}
    for r in rows:
        cid = r["condition_id"]
        result.setdefault(cid, []).append(
            {"prerequisite": r["prerequisite"], "description": r["description"]}
        )
    return result


async def get_condition_features_batch(
    conn: asyncpg.Connection,
    condition_ids: list[int],
) -> dict[int, list[dict]]:
    """Get clinical features for multiple conditions in one query.

    Returns {condition_id: [{name, feature_type, relationship_type}, ...]},
    sorted by priority: RED_FLAG first, then diagnostic, presenting, associated.
    """
    if not condition_ids:
        return {}
    rows = await conn.fetch("""
        SELECT cr.condition_id, e.canonical_name,
               cr.feature_type, cr.relationship_type
        FROM clinical_relationships cr
        JOIN clinical_entities e ON e.id = cr.source_entity_id
        WHERE cr.condition_id = ANY($1)
        AND e.entity_type = 'SYMPTOM'
        ORDER BY cr.condition_id,
                 CASE cr.relationship_type WHEN 'RED_FLAG' THEN 0 ELSE 1 END,
                 CASE cr.feature_type
                     WHEN 'diagnostic_feature' THEN 0
                     WHEN 'presenting_feature' THEN 1
                     ELSE 2 END
    """, condition_ids)
    result: dict[int, list[dict]] = {}
    for r in rows:
        cid = r["condition_id"]
        result.setdefault(cid, []).append({
            "name": r["canonical_name"],
            "feature_type": r["feature_type"],
            "relationship_type": r["relationship_type"],
        })
    return result


async def get_condition_by_stg_code(
    conn: asyncpg.Connection,
    stg_code: str,
) -> Optional[dict]:
    """Look up a condition by its STG code."""
    row = await conn.fetchrow(
        "SELECT * FROM conditions WHERE stg_code = $1", stg_code
    )
    return dict(row) if row else None


async def get_vitals_mappings(
    conn: asyncpg.Connection,
    vitals: dict,
) -> list[dict]:
    """
    Query vitals_condition_mapping for all thresholds matching the given vitals.
    Returns matched mappings sorted by score_boost descending.
    For each vital, picks the HIGHEST matching threshold (most specific).
    """
    if not vitals:
        return []

    results = []
    for vital_name, value in vitals.items():
        if value is None:
            continue
        # Query all matching thresholds for this vital
        rows = await conn.fetch("""
            SELECT vm.*, c.stg_code, c.name as condition_name,
                   c.chapter_name, c.extraction_confidence, c.duration_profile
            FROM vitals_condition_mapping vm
            JOIN conditions c ON c.id = vm.condition_id
            WHERE vm.vital_name = $1
            AND (
                (vm.operator = 'gte' AND $2::float >= vm.threshold)
                OR (vm.operator = 'gt' AND $2::float > vm.threshold)
                OR (vm.operator = 'lte' AND $2::float <= vm.threshold)
                OR (vm.operator = 'lt' AND $2::float < vm.threshold)
            )
            ORDER BY vm.score_boost DESC
        """, vital_name, float(value))

        # Group by condition_id, keep highest score_boost per condition
        seen = {}
        for r in rows:
            cid = r["condition_id"]
            if cid not in seen or r["score_boost"] > seen[cid]["score_boost"]:
                seen[cid] = dict(r)
        results.extend(seen.values())

    return sorted(results, key=lambda r: r["score_boost"], reverse=True)


async def vector_search_conditions(
    conn: asyncpg.Connection,
    query_embedding: list[float],
    patient_sex: Optional[str] = None,
    patient_age: Optional[int] = None,
    limit: int = 15,
    min_similarity: float = 0.65,
) -> list[dict]:
    """
    Vector search: find conditions whose knowledge chunks are semantically
    similar to the query embedding. Returns conditions with similarity scores.
    Respects gender and age filters.
    """
    vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    rows = await conn.fetch("""
        SELECT c.id, c.stg_code, c.name, c.chapter_name, c.extraction_confidence,
               c.duration_profile, kc.section_role,
               1 - (kc.embedding <=> $1::vector) as similarity
        FROM knowledge_chunks kc
        JOIN conditions c ON c.id = kc.condition_id
        WHERE kc.embedding IS NOT NULL
        AND ($2::text IS NULL
             OR ($2 = 'male' AND c.applies_to_male IS NOT FALSE)
             OR ($2 = 'female' AND c.applies_to_female IS NOT FALSE))
        AND ($3::int IS NULL OR (c.min_age_years <= $3 AND c.max_age_years >= $3))
        ORDER BY kc.embedding <=> $1::vector
        LIMIT $4
    """, vec_str, patient_sex, patient_age, limit)

    # Group by condition, keep best similarity per condition
    seen = {}
    for r in rows:
        sim = float(r["similarity"])
        if sim < min_similarity:
            continue
        cid = r["id"]
        if cid not in seen or sim > seen[cid]["similarity"]:
            seen[cid] = {
                "id": r["id"],
                "stg_code": r["stg_code"],
                "name": r["name"],
                "chapter_name": r["chapter_name"],
                "extraction_confidence": float(r["extraction_confidence"] or 1.0),
                "duration_profile": r["duration_profile"],
                "similarity": sim,
                "best_section": r["section_role"],
            }

    return sorted(seen.values(), key=lambda x: x["similarity"], reverse=True)


# ── Vignette CRUD (Phase II Survey) ─────────────────────────────────────────

async def create_vignette(conn: asyncpg.Connection, data: dict) -> dict:
    """Create a new clinical vignette. Returns the created vignette."""
    row = await conn.fetchrow("""
        INSERT INTO clinical_vignettes (
            vignette_code, title, domain, complaint,
            patient_age, patient_sex, pregnancy_status,
            vitals, core_history, additional_info,
            expected_conditions, expected_acuity, expected_sats_colour,
            difficulty, created_by
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
        RETURNING *
    """,
        data["vignette_code"], data["title"], data.get("domain"),
        data["complaint"], data.get("patient_age"), data.get("patient_sex"),
        data.get("pregnancy_status"),
        json.dumps(data.get("vitals") or {}),
        json.dumps(data.get("core_history") or {}),
        data.get("additional_info"),
        json.dumps(data.get("expected_conditions") or []),
        data.get("expected_acuity"), data.get("expected_sats_colour"),
        data.get("difficulty", "medium"), data.get("created_by"),
    )
    return dict(row)


async def list_vignettes(conn: asyncpg.Connection, active_only: bool = True) -> list[dict]:
    """List all vignettes with response counts."""
    where = "WHERE v.active = TRUE" if active_only else ""
    rows = await conn.fetch(f"""
        SELECT v.id, v.vignette_code, v.title, v.domain, v.complaint,
               v.patient_age, v.patient_sex, v.pregnancy_status,
               v.vitals, v.core_history, v.additional_info, v.difficulty,
               COUNT(vr.id) as response_count
        FROM clinical_vignettes v
        LEFT JOIN vignette_responses vr ON vr.vignette_id = v.id
        {where}
        GROUP BY v.id
        ORDER BY v.vignette_code
    """)
    return [dict(r) for r in rows]


async def get_vignette(conn: asyncpg.Connection, vignette_id: int) -> Optional[dict]:
    """Get a single vignette by ID."""
    row = await conn.fetchrow(
        "SELECT * FROM clinical_vignettes WHERE id = $1", vignette_id
    )
    return dict(row) if row else None


async def get_vignette_by_code(conn: asyncpg.Connection, code: str) -> Optional[dict]:
    """Get a single vignette by its code (e.g. 'PII-001')."""
    row = await conn.fetchrow(
        "SELECT * FROM clinical_vignettes WHERE vignette_code = $1", code
    )
    return dict(row) if row else None


async def save_vignette_response(conn: asyncpg.Connection, vignette_id: int, data: dict) -> dict:
    """Save a clinician or CareMate response to a vignette."""
    row = await conn.fetchrow("""
        INSERT INTO vignette_responses (
            vignette_id, respondent_type, respondent_name, respondent_credentials,
            differential_diagnosis, triage_level, sats_colour,
            investigations, treatment_plan,
            referral_decision, referral_reason,
            red_flags_identified, notes, time_taken_seconds
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
        RETURNING *
    """,
        vignette_id,
        data["respondent_type"], data.get("respondent_name"),
        data.get("respondent_credentials"),
        json.dumps(data.get("differential_diagnosis", [])),
        data.get("triage_level"), data.get("sats_colour"),
        json.dumps(data.get("investigations", [])),
        json.dumps(data.get("treatment_plan", [])),
        data.get("referral_decision"), data.get("referral_reason"),
        json.dumps(data.get("red_flags_identified", [])),
        data.get("notes"), data.get("time_taken_seconds"),
    )
    return dict(row)


async def get_vignette_responses(conn: asyncpg.Connection, vignette_id: int) -> list[dict]:
    """Get all responses for a vignette."""
    rows = await conn.fetch("""
        SELECT * FROM vignette_responses
        WHERE vignette_id = $1
        ORDER BY respondent_type, created_at
    """, vignette_id)
    return [dict(r) for r in rows]


async def get_vignette_comparison(conn: asyncpg.Connection, vignette_id: int) -> dict:
    """Get vignette with all responses grouped by type, plus expected answers."""
    vignette = await get_vignette(conn, vignette_id)
    if not vignette:
        return None
    responses = await get_vignette_responses(conn, vignette_id)

    clinician_responses = [r for r in responses if r["respondent_type"] == "clinician"]
    caremate_responses = [r for r in responses if r["respondent_type"] == "caremate"]

    return {
        "vignette": vignette,
        "clinician_responses": clinician_responses,
        "caremate_responses": caremate_responses,
        "total_clinicians": len(clinician_responses),
        "total_caremate": len(caremate_responses),
    }


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


# ── Assistant conversation persistence ──────────────────────


async def create_assistant_conversation(
    conn: asyncpg.Connection,
    encounter_id: str | None = None,
    patient_context: dict | None = None,
) -> str:
    """Create a new assistant conversation, return its UUID."""
    import json
    row = await conn.fetchrow(
        """
        INSERT INTO assistant_conversations (encounter_id, patient_context)
        VALUES ($1, $2::jsonb)
        RETURNING id
        """,
        encounter_id,
        json.dumps(patient_context or {}),
    )
    return str(row["id"])


async def get_assistant_messages(
    conn: asyncpg.Connection,
    conversation_id: str,
    limit: int = 20,
) -> list[dict]:
    """Get messages for a conversation, ordered oldest-first."""
    rows = await conn.fetch(
        """
        SELECT id, role, content, sources, tools_used, tool_calls, created_at
        FROM assistant_messages
        WHERE conversation_id = $1
        ORDER BY created_at ASC
        LIMIT $2
        """,
        conversation_id,
        limit,
    )
    results = []
    for r in rows:
        d = dict(r)
        d["created_at"] = d["created_at"].isoformat() if d["created_at"] else None
        import json
        for col in ("sources", "tools_used", "tool_calls"):
            if isinstance(d[col], str):
                d[col] = json.loads(d[col])
        results.append(d)
    return results


async def save_assistant_message(
    conn: asyncpg.Connection,
    conversation_id: str,
    role: str,
    content: str,
    sources: list | None = None,
    tools_used: list | None = None,
    tool_calls: list | None = None,
) -> int:
    """Save a message and return its id."""
    import json
    row = await conn.fetchrow(
        """
        INSERT INTO assistant_messages
            (conversation_id, role, content, sources, tools_used, tool_calls)
        VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb)
        RETURNING id
        """,
        conversation_id,
        role,
        content,
        json.dumps(sources or []),
        json.dumps(tools_used or []),
        json.dumps(tool_calls or []),
    )
    await conn.execute(
        "UPDATE assistant_conversations SET updated_at = NOW() WHERE id = $1",
        conversation_id,
    )
    return row["id"]
