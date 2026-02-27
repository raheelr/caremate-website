"""
Post-Hoc Enrichment Pass
-------------------------
Reads existing conditions + knowledge_chunks and enriches:

1. Patient-language symptoms → new presenting_feature graph edges
2. Missing common associated symptoms (fever, headache, etc.) → new graph edges
3. Synonym rings: patient terms → symptom canonical names (not condition names)
4. Feature type audit: fix mis-classified feature types

Usage:
  python3 ingestion/enrich.py --dry-run          # preview, no DB changes
  python3 ingestion/enrich.py --apply            # save to DB
  python3 ingestion/enrich.py --apply --limit 5  # test on 5 conditions
  python3 ingestion/enrich.py --apply --resume   # skip already-enriched conditions
"""

import os
import sys
import json
import time
import asyncio
import argparse
import logging
from typing import Optional

import asyncpg
import anthropic
from dotenv import load_dotenv

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("enrich")


# ── Enrichment tool schema (structured output via tool_use) ──────────────────

ENRICHMENT_TOOL = {
    "name": "enrich_condition",
    "description": "Add missing patient-language symptoms and synonym mappings for a condition",
    "input_schema": {
        "type": "object",
        "properties": {
            "new_patient_symptoms": {
                "type": "array",
                "description": (
                    "Patient-language symptoms that should be added as graph edges. "
                    "These are ways patients describe the condition's symptoms, "
                    "NOT formal clinical terms (those already exist). "
                    "E.g., 'sore throat' for a condition that has 'painful red throat' as a feature."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "symptom": {
                            "type": "string",
                            "description": "Patient-language symptom term (lowercase)"
                        },
                        "feature_type": {
                            "type": "string",
                            "enum": ["presenting_feature", "associated_feature"],
                            "description": "presenting=commonly reported, associated=sometimes present"
                        },
                        "maps_to_existing": {
                            "type": "string",
                            "description": (
                                "The existing canonical feature this is a synonym of, "
                                "or empty string if it's a genuinely new symptom not yet in the graph"
                            )
                        }
                    },
                    "required": ["symptom", "feature_type", "maps_to_existing"]
                }
            },
            "missing_common_symptoms": {
                "type": "array",
                "description": (
                    "Common clinical symptoms mentioned in the condition's text "
                    "that were NOT extracted as features. E.g., 'fever' if the text "
                    "mentions fever but it's not in the existing features list. "
                    "Use proper clinical terms here."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "symptom": {
                            "type": "string",
                            "description": "Clinical symptom term (lowercase)"
                        },
                        "feature_type": {
                            "type": "string",
                            "enum": ["diagnostic_feature", "presenting_feature", "associated_feature"],
                        },
                        "evidence": {
                            "type": "string",
                            "description": "Quote or paraphrase from the text that supports this"
                        }
                    },
                    "required": ["symptom", "feature_type", "evidence"]
                }
            },
            "synonym_mappings": {
                "type": "array",
                "description": (
                    "Mappings from patient language to SYMPTOM canonical names. "
                    "The canonical_term MUST be a symptom entity that exists (or will exist) "
                    "in the graph — NOT a condition name."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "patient_term": {
                            "type": "string",
                            "description": "How a patient would describe this (lowercase)"
                        },
                        "canonical_symptom": {
                            "type": "string",
                            "description": (
                                "The canonical symptom entity name this maps to (lowercase). "
                                "Must be a symptom, NOT a condition name."
                            )
                        }
                    },
                    "required": ["patient_term", "canonical_symptom"]
                }
            },
            "feature_type_corrections": {
                "type": "array",
                "description": (
                    "Corrections to existing feature types. Only include if clearly wrong. "
                    "E.g., a presenting_feature that should be diagnostic_feature."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "feature": {"type": "string"},
                        "current_type": {"type": "string"},
                        "corrected_type": {
                            "type": "string",
                            "enum": ["diagnostic_feature", "presenting_feature", "associated_feature"]
                        },
                        "reason": {"type": "string"}
                    },
                    "required": ["feature", "current_type", "corrected_type", "reason"]
                }
            }
        },
        "required": [
            "new_patient_symptoms",
            "missing_common_symptoms",
            "synonym_mappings",
            "feature_type_corrections"
        ]
    }
}


ENRICHMENT_SYSTEM = """You are a clinical NLP enrichment tool for a South African primary healthcare triage system.

You are given a condition from the Standard Treatment Guidelines (STG) along with:
- Its existing clinical feature graph edges (how it's currently findable)
- Its raw text from the STG (knowledge chunks)

Your job is to identify GAPS — things that would prevent a nurse's description from matching this condition.

RULES:
1. Only add symptoms that are EXPLICITLY mentioned or strongly implied in the STG text
2. Patient-language symptoms should be how a South African patient or nurse would describe it
3. Do NOT duplicate existing features — check the existing list carefully
4. For synonym_mappings, the canonical_symptom MUST be a symptom entity name, NEVER a condition name
5. Be thorough but precise — false positives are worse than missing a few terms
6. Include common symptom terms: if the text mentions "fever", "pain", "swelling", etc., these should be features
7. Think about what a nurse would type: "sore throat", "can't swallow", "high temperature", "runny nose"
8. Feature types:
   - diagnostic_feature: would strongly suggest THIS condition specifically
   - presenting_feature: commonly reported by patients with this condition
   - associated_feature: sometimes present but not specific to this condition
"""


class ConditionEnricher:
    """Enriches all conditions with patient-language symptoms and synonym mappings."""

    def __init__(self, pool: asyncpg.Pool, dry_run: bool = True):
        self.pool = pool
        self.dry_run = dry_run
        self.client = anthropic.Anthropic()
        self.model = "claude-haiku-4-5-20251001"

        # Stats
        self.processed = 0
        self.skipped = 0
        self.new_edges = 0
        self.new_synonyms = 0
        self.corrections = 0
        self.errors = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Detailed report
        self.report = []

    async def _get_already_enriched(self) -> set:
        """Get condition IDs that already have ENRICHMENT edges."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT DISTINCT condition_id FROM clinical_relationships
                WHERE source_section = 'ENRICHMENT'
            """)
        return {r["condition_id"] for r in rows}

    async def _reconnect_pool(self):
        """Recreate the connection pool if connections were lost."""
        try:
            await self.pool.close()
        except Exception:
            pass
        database_url = os.getenv("DATABASE_URL")
        self.pool = await asyncpg.create_pool(database_url, min_size=2, max_size=5)
        logger.info("Reconnected to database")

    async def enrich_all(self, limit: Optional[int] = None, resume: bool = False):
        """Run enrichment on all (or limited) conditions."""
        already_enriched = set()
        if resume:
            already_enriched = await self._get_already_enriched()
            logger.info(f"Resume mode: {len(already_enriched)} conditions already enriched, will skip")

        async with self.pool.acquire() as conn:
            if limit:
                conditions = await conn.fetch(
                    "SELECT id, stg_code, name FROM conditions ORDER BY stg_code LIMIT $1",
                    limit
                )
            else:
                conditions = await conn.fetch(
                    "SELECT id, stg_code, name FROM conditions ORDER BY stg_code"
                )

        total = len(conditions)
        logger.info(f"Enriching {total} conditions ({'DRY RUN' if self.dry_run else 'APPLYING'})")

        consecutive_errors = 0
        for i, cond in enumerate(conditions):
            if resume and cond["id"] in already_enriched:
                self.skipped += 1
                continue

            try:
                await self._enrich_one(cond["id"], cond["stg_code"], cond["name"], i + 1, total)
                self.processed += 1
                consecutive_errors = 0
            except anthropic.RateLimitError:
                logger.warning("Rate limited — waiting 30s...")
                time.sleep(30)
                try:
                    await self._enrich_one(cond["id"], cond["stg_code"], cond["name"], i + 1, total)
                    self.processed += 1
                    consecutive_errors = 0
                except Exception as e:
                    logger.error(f"Failed {cond['stg_code']} after retry: {e}")
                    self.errors += 1
                    consecutive_errors += 1
            except (OSError, asyncpg.PostgresConnectionError, ConnectionError) as e:
                logger.error(f"Connection error at {cond['stg_code']}: {e}")
                self.errors += 1
                consecutive_errors += 1
                # Try to reconnect
                logger.info("Attempting to reconnect...")
                time.sleep(5)
                try:
                    await self._reconnect_pool()
                    consecutive_errors = 0
                except Exception as re:
                    logger.error(f"Reconnect failed: {re}")
                    if consecutive_errors >= 5:
                        logger.error("Too many consecutive errors — aborting")
                        break
                    time.sleep(30)
            except Exception as e:
                logger.error(f"Failed {cond['stg_code']}: {e}")
                self.errors += 1
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    logger.error("Too many consecutive errors — aborting")
                    break

            # Small delay to avoid rate limits
            if (i + 1) % 20 == 0:
                logger.info(f"  Progress: {i+1}/{total} — pausing 2s...")
                time.sleep(2)

        self._print_summary()
        return self.report

    async def _enrich_one(self, condition_id: int, stg_code: str, name: str, idx: int, total: int):
        """Enrich a single condition."""
        async with self.pool.acquire() as conn:
            # Get existing features
            features = await conn.fetch("""
                SELECT e.canonical_name, cr.feature_type, cr.relationship_type
                FROM clinical_relationships cr
                JOIN clinical_entities e ON e.id = cr.source_entity_id
                WHERE cr.condition_id = $1
                ORDER BY cr.feature_type
            """, condition_id)

            # Get knowledge chunks (the raw STG text)
            chunks = await conn.fetch("""
                SELECT section_role, chunk_text
                FROM knowledge_chunks
                WHERE condition_id = $1
                ORDER BY section_role
            """, condition_id)

        # Build context for the LLM
        existing_features = []
        for f in features:
            existing_features.append(
                f"{f['feature_type']} | {f['relationship_type']} | {f['canonical_name']}"
            )

        chunks_text = ""
        for c in chunks:
            chunks_text += f"\n[{c['section_role']}]\n{c['chunk_text']}\n"

        # Get description_text as additional context
        async with self.pool.acquire() as conn:
            desc = await conn.fetchval(
                "SELECT description_text FROM conditions WHERE id = $1", condition_id
            )

        if not chunks and not desc and not features:
            # No STG text at all — use Claude's medical knowledge for this condition.
            # These are well-known conditions (Cholera, Meningitis, VDS, etc.) where
            # the ingestion pipeline failed to extract content from the PDF.
            prompt = (
                f"CONDITION: {name} (STG {stg_code})\n\n"
                f"This condition has NO extracted text from the STG guidelines and NO existing features.\n"
                f"Based on your medical knowledge of this condition in a South African primary healthcare context:\n\n"
                f"1. What symptoms would a patient present with? (patient-language terms)\n"
                f"2. What are the key clinical features a nurse should look for?\n"
                f"3. What synonym mappings would help match patient descriptions?\n\n"
                f"Be thorough — this condition currently has ZERO searchable features.\n"
                f"Focus on the most common presentation at a primary care clinic.\n"
                f"Include both patient-language terms ('burning pee') and clinical terms ('dysuria')."
            )
        elif not chunks and desc:
            prompt = (
                f"CONDITION: {name} (STG {stg_code})\n\n"
                f"EXISTING FEATURES ({len(features)} edges):\n"
                + ("\n".join(existing_features) if existing_features else "(none)")
                + f"\n\nDESCRIPTION:\n{desc}\n\n"
                f"Identify gaps: patient-language symptoms missing from the graph, "
                f"common symptoms mentioned in the text but not extracted, "
                f"and proper synonym mappings (patient term → symptom canonical name)."
            )
        else:
            prompt = (
                f"CONDITION: {name} (STG {stg_code})\n\n"
                f"EXISTING FEATURES ({len(features)} edges):\n"
                + ("\n".join(existing_features) if existing_features else "(none)")
                + f"\n\nSTG TEXT:\n{chunks_text}\n\n"
                f"Identify gaps: patient-language symptoms missing from the graph, "
                f"common symptoms mentioned in the text but not extracted, "
                f"and proper synonym mappings (patient term → symptom canonical name)."
            )

        # Call Haiku with tool_use
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=ENRICHMENT_SYSTEM,
            tools=[ENRICHMENT_TOOL],
            tool_choice={"type": "tool", "name": "enrich_condition"},
            messages=[{"role": "user", "content": prompt}],
        )

        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        # Extract tool result
        enrichment = None
        for block in response.content:
            if block.type == "tool_use" and block.name == "enrich_condition":
                enrichment = block.input
                break

        if not enrichment:
            logger.warning(f"  [{idx}/{total}] {stg_code} — no enrichment returned")
            return

        # Validate: reject clearly malformed responses
        # (e.g. Haiku returning 822 "missing" symptoms is garbage)
        MAX_ITEMS = 30  # no condition should need more than 30 of any category
        for key in ["new_patient_symptoms", "missing_common_symptoms", "synonym_mappings"]:
            items = enrichment.get(key, [])
            if len(items) > MAX_ITEMS:
                logger.warning(f"  [{idx}/{total}] {stg_code} — {key} has {len(items)} items (capped at {MAX_ITEMS})")
                enrichment[key] = items[:MAX_ITEMS]
            # Validate each item is a dict (not a string)
            enrichment[key] = [item for item in enrichment.get(key, []) if isinstance(item, dict)]

        # Count additions
        n_patient = len(enrichment.get("new_patient_symptoms", []))
        n_missing = len(enrichment.get("missing_common_symptoms", []))
        n_synonyms = len(enrichment.get("synonym_mappings", []))
        n_corrections = len(enrichment.get("feature_type_corrections", []))
        n_total = n_patient + n_missing + n_synonyms + n_corrections

        if n_total == 0:
            logger.info(f"  [{idx}/{total}] {stg_code} {name} — already complete")
            return

        logger.info(
            f"  [{idx}/{total}] {stg_code} {name} — "
            f"+{n_patient} patient symptoms, +{n_missing} missing, "
            f"+{n_synonyms} synonyms, {n_corrections} corrections"
        )

        # Build report entry
        entry = {
            "stg_code": stg_code,
            "name": name,
            "new_patient_symptoms": enrichment.get("new_patient_symptoms", []),
            "missing_common_symptoms": enrichment.get("missing_common_symptoms", []),
            "synonym_mappings": enrichment.get("synonym_mappings", []),
            "feature_type_corrections": enrichment.get("feature_type_corrections", []),
        }
        self.report.append(entry)

        # Apply to DB if not dry run
        if not self.dry_run:
            await self._apply_enrichment(condition_id, enrichment)

        self.new_edges += n_patient + n_missing
        self.new_synonyms += n_synonyms
        self.corrections += n_corrections

    async def _apply_enrichment(self, condition_id: int, enrichment: dict):
        """Write enrichment data to the database."""
        async with self.pool.acquire() as conn:
            # Get condition entity ID for relationship target
            condition_entity_id = await conn.fetchval("""
                SELECT e.id FROM clinical_entities e
                JOIN conditions c ON LOWER(c.name) = e.canonical_name
                WHERE c.id = $1 AND e.entity_type = 'CONDITION'
                LIMIT 1
            """, condition_id)

            if not condition_entity_id:
                # Create condition entity if missing
                cond_name = await conn.fetchval(
                    "SELECT name FROM conditions WHERE id = $1", condition_id
                )
                if cond_name:
                    condition_entity_id = await conn.fetchval("""
                        INSERT INTO clinical_entities (canonical_name, entity_type, aliases)
                        VALUES ($1, 'CONDITION', '{}')
                        ON CONFLICT (canonical_name) DO UPDATE SET entity_type = 'CONDITION'
                        RETURNING id
                    """, cond_name.lower())

            # 1. Add new patient-language symptoms as graph edges
            for item in enrichment.get("new_patient_symptoms", []):
                symptom = item["symptom"].lower().strip()
                if not symptom:
                    continue
                entity_id = await conn.fetchval("""
                    INSERT INTO clinical_entities (canonical_name, entity_type, aliases)
                    VALUES ($1, 'SYMPTOM', '{}')
                    ON CONFLICT (canonical_name) DO UPDATE SET entity_type = 'SYMPTOM'
                    RETURNING id
                """, symptom)

                if condition_entity_id:
                    await conn.execute("""
                        INSERT INTO clinical_relationships (
                            source_entity_id, target_entity_id,
                            relationship_type, feature_type,
                            condition_id, source_section, confidence
                        ) VALUES ($1, $2, 'INDICATES', $3, $4, 'ENRICHMENT', 0.9)
                        ON CONFLICT (source_entity_id, condition_id, relationship_type, feature_type)
                        DO NOTHING
                    """, entity_id, condition_entity_id, item["feature_type"], condition_id)

            # 2. Add missing common symptoms
            for item in enrichment.get("missing_common_symptoms", []):
                symptom = item["symptom"].lower().strip()
                if not symptom:
                    continue
                entity_id = await conn.fetchval("""
                    INSERT INTO clinical_entities (canonical_name, entity_type, aliases)
                    VALUES ($1, 'SYMPTOM', '{}')
                    ON CONFLICT (canonical_name) DO UPDATE SET entity_type = 'SYMPTOM'
                    RETURNING id
                """, symptom)

                if condition_entity_id:
                    await conn.execute("""
                        INSERT INTO clinical_relationships (
                            source_entity_id, target_entity_id,
                            relationship_type, feature_type,
                            condition_id, source_section, confidence
                        ) VALUES ($1, $2, 'INDICATES', $3, $4, 'ENRICHMENT', 0.85)
                        ON CONFLICT (source_entity_id, condition_id, relationship_type, feature_type)
                        DO NOTHING
                    """, entity_id, condition_entity_id, item["feature_type"], condition_id)

            # 3. Add synonym mappings (patient term → symptom canonical name)
            for item in enrichment.get("synonym_mappings", []):
                patient_term = item["patient_term"].lower().strip()
                canonical_symptom = item["canonical_symptom"].lower().strip()
                if not patient_term or not canonical_symptom:
                    continue

                # Verify canonical_symptom is actually a symptom entity (or will be after enrichment)
                is_symptom = await conn.fetchval("""
                    SELECT EXISTS(
                        SELECT 1 FROM clinical_entities
                        WHERE canonical_name = $1 AND entity_type = 'SYMPTOM'
                    )
                """, canonical_symptom)

                # Also check it's not a condition name
                is_condition = await conn.fetchval("""
                    SELECT EXISTS(
                        SELECT 1 FROM conditions WHERE LOWER(name) = $1
                    )
                """, canonical_symptom)

                if is_condition:
                    continue  # Skip — don't map patient language to condition names

                # Save synonym (even if canonical doesn't exist yet — it may be added by new_patient_symptoms)
                await conn.execute("""
                    INSERT INTO synonym_rings (canonical_term, synonym)
                    VALUES ($1, $2)
                    ON CONFLICT (canonical_term, synonym) DO NOTHING
                """, canonical_symptom, patient_term)

            # 4. Apply feature type corrections
            for item in enrichment.get("feature_type_corrections", []):
                feature = item["feature"].lower().strip()
                corrected_type = item["corrected_type"]

                entity_id = await conn.fetchval("""
                    SELECT id FROM clinical_entities
                    WHERE canonical_name = $1 AND entity_type = 'SYMPTOM'
                """, feature)

                if entity_id:
                    await conn.execute("""
                        UPDATE clinical_relationships
                        SET feature_type = $1
                        WHERE source_entity_id = $2 AND condition_id = $3
                    """, corrected_type, entity_id, condition_id)

    def _print_summary(self):
        """Print enrichment summary."""
        cost_input = self.total_input_tokens * 0.00000080   # Haiku input
        cost_output = self.total_output_tokens * 0.00000400  # Haiku output
        total_cost = cost_input + cost_output

        print(f"\n{'='*60}")
        print(f"ENRICHMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Conditions processed:     {self.processed}")
        print(f"Conditions skipped:       {self.skipped}")
        print(f"Errors:                   {self.errors}")
        print(f"New graph edges:          {self.new_edges}")
        print(f"New synonym mappings:     {self.new_synonyms}")
        print(f"Feature type corrections: {self.corrections}")
        print(f"{'='*60}")
        print(f"Input tokens:             {self.total_input_tokens:,}")
        print(f"Output tokens:            {self.total_output_tokens:,}")
        print(f"Estimated cost:           ${total_cost:.2f}")
        print(f"{'='*60}")

        if self.dry_run:
            print("\nDRY RUN — no changes saved. Use --apply to save to database.")

    def save_report(self, path: str):
        """Save detailed report to JSON."""
        with open(path, "w") as f:
            json.dump({
                "summary": {
                    "conditions_processed": self.processed,
                    "errors": self.errors,
                    "new_edges": self.new_edges,
                    "new_synonyms": self.new_synonyms,
                    "corrections": self.corrections,
                    "dry_run": self.dry_run,
                },
                "conditions": self.report,
            }, f, indent=2)
        logger.info(f"Report saved to {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Enrich conditions with patient-language symptoms")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no DB writes")
    parser.add_argument("--apply", action="store_true", help="Apply changes to database")
    parser.add_argument("--limit", type=int, default=None, help="Process only N conditions")
    parser.add_argument("--resume", action="store_true", help="Skip already-enriched conditions")
    parser.add_argument("--report", type=str, default="enrichment_report.json", help="Report output path")
    args = parser.parse_args()

    if not args.apply and not args.dry_run:
        print("Specify --dry-run or --apply")
        sys.exit(1)

    dry_run = not args.apply

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL not set — add it to .env")
        sys.exit(1)

    pool = await asyncpg.create_pool(database_url, min_size=2, max_size=5)

    try:
        enricher = ConditionEnricher(pool, dry_run=dry_run)
        await enricher.enrich_all(limit=args.limit, resume=args.resume)
        enricher.save_report(args.report)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
