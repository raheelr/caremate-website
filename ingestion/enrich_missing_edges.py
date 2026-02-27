"""
Enrich Missing Edges
--------------------
Adds patient-language edges and synonyms for conditions that fail the deep test
due to missing graph connectivity. Targeted enrichment — no LLM needed.
"""

import os
import sys
import asyncio
import asyncpg
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass


# Each entry: (stg_code, entity_canonical_name, entity_type, feature_type, synonyms[])
ENRICHMENTS = [
    # ── Peptic Ulcer Disease (2.3) ──
    # Missing: epigastric pain, hunger pain, burning stomach
    ("2.3", "epigastric pain", "symptom", "presenting_feature", [
        "burning tummy pain", "pain in upper belly", "stomach burn",
        "burning stomach pain", "upper stomach pain",
    ]),
    ("2.3", "pain worse when hungry", "symptom", "diagnostic_feature", [
        "worse when hungry", "pain before eating", "hunger pain",
        "tummy sore when empty", "pain relieved by eating",
    ]),
    ("2.3", "heartburn", "symptom", "presenting_feature", [
        "chest burn", "acid reflux", "sour stomach",
    ]),

    # ── Injectable Contraception (7.2.3) ──
    # Non-symptom counseling condition — needs request-based edges
    ("7.2.3", "contraception request", "symptom", "presenting_feature", [
        "birth control injection", "contraceptive injection", "family planning injection",
        "depo injection", "depo shot", "birth control shot",
        "how to use birth control", "want to prevent pregnancy",
    ]),
    ("7.2.3", "injectable contraception counselling", "symptom", "presenting_feature", [
        "injection for family planning", "3-month injection",
        "contraception advice", "prevent pregnancy injection",
    ]),

    # ── Tick Bite Fever (10.14) ──
    # Missing: patient language for eschar, bush exposure
    ("10.14", "tick bite", "symptom", "presenting_feature", [
        "bitten by tick", "tick on skin", "tick mark",
        "insect bite in bush", "bite from tick",
    ]),
    ("10.14", "eschar", "sign", "diagnostic_feature", [
        "bite mark with black scab", "black spot from bite",
        "dark scab at bite site", "round black sore",
    ]),
    ("10.14", "bush exposure", "risk_factor", "associated_feature", [
        "been in the bush", "walking in long grass",
        "camping in veld", "exposure to ticks",
    ]),
    ("10.14", "myalgia", "symptom", "presenting_feature", [
        "muscle pain", "body aches", "sore muscles",
        "body pain", "aching all over",
    ]),

    # ── Osteoarthritis (14.5) ──
    # Missing: patient language for joint symptoms
    ("14.5", "joint pain", "symptom", "presenting_feature", [
        "sore joints", "painful joints", "joint ache",
        "sore knees", "knee pain", "hip pain",
    ]),
    ("14.5", "morning stiffness", "symptom", "presenting_feature", [
        "stiff joints in morning", "stiff when waking up",
        "joints stiff after rest", "hard to move in morning",
    ]),
    ("14.5", "progressive joint deterioration", "symptom", "diagnostic_feature", [
        "joint pain getting worse", "joints getting worse over time",
        "pain worse over months", "gradual joint damage",
    ]),

    # ── Burns (21.3.2) ──
    # Missing: patient language for thermal injury
    ("21.3.2", "thermal burn injury", "symptom", "presenting_feature", [
        "burnt by hot water", "burn from fire", "burnt skin",
        "boiling water burn", "flame burn", "hot oil burn",
        "scalded", "scald burn",
    ]),
    ("21.3.2", "burn pain", "symptom", "presenting_feature", [
        "very painful burn", "burn is sore", "burning pain on skin",
        "skin on fire", "painful burn wound",
    ]),
    ("21.3.2", "burn on extremity", "symptom", "presenting_feature", [
        "burn on arm", "burn on hand", "burn on leg",
        "burn on foot", "burn on face",
    ]),

    # ── Paracetamol Poisoning (21.3.4) ──
    # Missing: patient language for overdose
    ("21.3.4", "paracetamol overdose", "symptom", "presenting_feature", [
        "took too many paracetamol", "too many panado",
        "swallowed too many pills", "overdose on paracetamol",
        "took whole bottle of panado", "panado overdose",
    ]),
    ("21.3.4", "medication ingestion", "symptom", "presenting_feature", [
        "swallowed too many tablets", "drug overdose",
        "took too many pills", "ate too many tablets",
    ]),
    ("21.3.4", "abdominal pain after ingestion", "symptom", "presenting_feature", [
        "tummy pain after pills", "stomach pain after tablets",
        "nausea after taking pills", "vomiting after overdose",
    ]),
]


async def main():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    conn = await asyncpg.connect(database_url)

    try:
        entities_added = 0
        edges_added = 0
        synonyms_added = 0

        for stg_code, entity_name, entity_type, feature_type, patient_synonyms in ENRICHMENTS:
            # Find condition
            cond = await conn.fetchrow(
                "SELECT id FROM conditions WHERE stg_code = $1", stg_code
            )
            if not cond:
                print(f"  WARNING: condition {stg_code} not found, skipping")
                continue

            # Find or create entity
            entity = await conn.fetchrow(
                "SELECT id FROM clinical_entities WHERE canonical_name = $1",
                entity_name
            )
            if not entity:
                entity = await conn.fetchrow(
                    "INSERT INTO clinical_entities (canonical_name, entity_type) VALUES ($1, $2) RETURNING id",
                    entity_name, entity_type
                )
                entities_added += 1

            # Check if edge already exists
            existing = await conn.fetchrow(
                """SELECT id FROM clinical_relationships
                   WHERE condition_id = $1 AND source_entity_id = $2 AND feature_type = $3""",
                cond["id"], entity["id"], feature_type
            )
            if not existing:
                await conn.execute(
                    """INSERT INTO clinical_relationships
                       (source_entity_id, target_entity_id, relationship_type, feature_type, condition_id, confidence)
                       VALUES ($1, $1, 'INDICATES', $2, $3, 0.9)""",
                    entity["id"], feature_type, cond["id"]
                )
                edges_added += 1

            # Add patient-language synonyms
            for syn in patient_synonyms:
                existing_syn = await conn.fetchrow(
                    "SELECT id FROM synonym_rings WHERE canonical_term = $1 AND synonym = $2",
                    entity_name, syn
                )
                if not existing_syn:
                    await conn.execute(
                        """INSERT INTO synonym_rings (canonical_term, synonym, relationship_type)
                           VALUES ($1, $2, 'patient_language')""",
                        entity_name, syn
                    )
                    synonyms_added += 1

            print(f"  {stg_code}: {entity_name} — enriched")

        print(f"\n{'=' * 60}")
        print(f"ENRICHMENT COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Entities added: {entities_added}")
        print(f"  Edges added: {edges_added}")
        print(f"  Synonyms added: {synonyms_added}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
