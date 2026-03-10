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

    # ── Measles (10.8) ──
    # Measles was missing from DB entirely — extracted via targeted pipeline run
    ("10.8", "maculopapular rash", "symptom", "diagnostic_feature", [
        "red blotchy rash", "spotty rash on face", "rash spreading from face to body",
        "rash starting behind ears", "rash on face and body",
    ]),
    ("10.8", "conjunctivitis", "symptom", "presenting_feature", [
        "red eyes", "watery eyes", "sore eyes", "pink eye with rash",
    ]),
    ("10.8", "coryza", "symptom", "presenting_feature", [
        "runny nose", "blocked nose", "nasal congestion with rash",
    ]),
    ("10.8", "koplik spots", "sign", "diagnostic_feature", [
        "white spots in mouth", "white spots inside cheek",
    ]),
    ("10.8", "incomplete immunisation", "risk_factor", "associated_feature", [
        "not vaccinated", "missed vaccinations", "no measles vaccine",
        "incomplete vaccination card",
    ]),

    # ── Chronic Kidney Disease (8.1) ──
    # CKD has 25 edges but all lab-based. Missing clinical presentation symptoms.
    ("8.1", "fatigue", "symptom", "presenting_feature", [
        "always tired", "no energy", "feeling weak", "exhausted all the time",
    ]),
    ("8.1", "peripheral oedema", "symptom", "presenting_feature", [
        "swollen ankles", "swollen feet", "puffy legs", "legs swelling in evenings",
    ]),
    ("8.1", "nocturia", "symptom", "presenting_feature", [
        "peeing at night", "waking up to pee", "pass urine often at night",
        "getting up to wee at night",
    ]),
    ("8.1", "pruritus", "symptom", "associated_feature", [
        "itching all over", "skin itching", "itchy skin without rash",
        "scratching all the time", "itchy at night",
    ]),
    ("8.1", "nausea in renal disease", "symptom", "associated_feature", [
        "feeling sick", "nauseous", "queasy",
    ]),
    ("8.1", "decreased appetite", "symptom", "associated_feature", [
        "not hungry", "lost appetite", "off food", "not eating well",
    ]),

    # ── Growth Faltering / Not Growing Well (3.2.3) ──
    # Missing: patient language for stunting and growth failure
    ("3.2.3", "growth faltering", "symptom", "presenting_feature", [
        "child much smaller than peers", "not growing well",
        "shorter than other children", "small for age",
    ]),
    ("3.2.3", "failure to thrive", "symptom", "presenting_feature", [
        "not gaining weight", "weight below expected",
        "thin limbs", "not growing as fast as other kids",
    ]),
    ("3.2.3", "poor dietary intake", "risk_factor", "associated_feature", [
        "poor diet", "poor diet little protein", "skips meals",
        "limited food variety", "mainly porridge diet",
    ]),

    # ── Severe Acute Malnutrition (3.2.1) ──
    # Missing: patient language for oedematous malnutrition (kwashiorkor)
    ("3.2.1.1", "oedematous malnutrition", "symptom", "diagnostic_feature", [
        "swollen feet and face in child", "child swelling all over",
        "puffy face child", "swollen legs malnourished child",
    ]),
    ("3.2.1.1", "skin changes malnutrition", "symptom", "presenting_feature", [
        "skin peeling and patchy", "skin peeling child",
        "hair falling out child", "sparse hair",
    ]),

    # ── Prostate Cancer (8.8) ──
    # Missing: patient language for lower urinary tract symptoms
    ("8.8", "lower urinary tract symptoms", "symptom", "presenting_feature", [
        "trouble passing urine", "difficulty starting to urinate",
        "weak urine stream", "dribbling after peeing",
    ]),
    ("8.8", "nocturia prostate", "symptom", "presenting_feature", [
        "getting up many times at night to pee", "peeing often at night old man",
        "up 4-5 times at night to urinate",
    ]),
    ("8.8", "prostate abnormality on exam", "sign", "diagnostic_feature", [
        "hard lump in prostate", "nodule in prostate",
        "irregular prostate on exam", "firm prostate",
    ]),

    # ── Pneumonia (17.3.4.1) ──
    # Missing: patient language for productive cough + pleuritic chest pain
    ("17.3.4.1", "productive cough", "symptom", "presenting_feature", [
        "coughing up green stuff", "coughing up phlegm", "coughing green mucus",
        "thick sputum", "green sputum", "yellow sputum",
    ]),
    ("17.3.4.1", "pleuritic chest pain", "symptom", "presenting_feature", [
        "chest pain when breathing", "sharp chest pain on breathing in",
        "chest sore when I breathe", "pain in chest when coughing",
    ]),

    # ── Hypertension (4.7.1) ──
    # Missing: patient language for hypertension presentation + non-adherence
    ("4.7.1", "hypertensive headache", "symptom", "presenting_feature", [
        "pounding headache", "throbbing headache", "headache at back of head",
        "bad headache with high blood pressure", "pressure headache",
    ]),
    ("4.7.1", "medication non-adherence", "risk_factor", "associated_feature", [
        "ran out of blood pressure pills", "stopped taking BP meds",
        "not taking my tablets", "ran out of medication",
    ]),
    ("4.7.1", "postural dizziness", "symptom", "presenting_feature", [
        "dizziness on standing", "dizzy when getting up", "lightheaded on standing",
        "head spinning when I stand up",
    ]),

    # ── Cardiac Failure, Congestive (4.5) ──
    # Missing: patient language for dyspnoea on exertion + orthopnoea
    ("4.5", "exertional dyspnoea", "symptom", "presenting_feature", [
        "short of breath walking", "breathless when walking",
        "can't walk far without stopping", "out of breath on exertion",
    ]),
    ("4.5", "orthopnoea", "symptom", "diagnostic_feature", [
        "cannot lie flat to sleep", "need extra pillows to breathe",
        "have to sit up to breathe", "can't sleep flat",
    ]),
    ("4.5", "paroxysmal nocturnal dyspnoea", "symptom", "diagnostic_feature", [
        "waking up breathless at night", "wake up gasping for air",
        "sudden breathlessness at night", "can't breathe lying down at night",
    ]),
    ("4.5", "ankle oedema", "symptom", "presenting_feature", [
        "swollen ankles evening", "feet swelling in the evening",
        "ankles puffed up", "legs swollen by end of day",
    ]),

    # ── Angina Pectoris, Stable (4.2) ──
    # Missing: patient language for exertional chest pain
    ("4.2", "exertional chest pain", "symptom", "diagnostic_feature", [
        "chest pain when walking", "chest pain on exertion",
        "pain in chest when climbing stairs", "chest tightness when exercising",
    ]),
    ("4.2", "chest pain relieved by rest", "symptom", "diagnostic_feature", [
        "pain goes away with rest", "chest pain stops when I sit down",
        "pain eases when resting", "chest better after resting",
    ]),
    ("4.2", "angina", "symptom", "presenting_feature", [
        "angina pain", "angina attack",
    ]),

    # ── Boil / Abscess (5.4.1) ──
    # Missing: patient language for skin abscess
    ("5.4.1", "skin abscess", "symptom", "presenting_feature", [
        "painful lump on skin", "red swelling with pus",
        "boil getting bigger", "big sore lump", "swollen sore on skin",
    ]),
    ("5.4.1", "boil", "symptom", "presenting_feature", [
        "boil on leg", "boil on face", "boil on back",
        "boil that burst", "boil with pus coming out",
    ]),

    # ── Scabies (5.8.1) ──
    # Missing: patient language for nocturnal itch, burrow pattern
    ("5.8.1", "nocturnal pruritus", "symptom", "diagnostic_feature", [
        "itching worse at night", "itch at night", "scratching all night",
        "itchy only at night", "can't sleep from itching",
    ]),
    ("5.8.1", "interdigital rash", "symptom", "diagnostic_feature", [
        "rash between fingers", "rash on hands between fingers",
        "bumps between fingers", "itchy rash on fingers",
    ]),
    ("5.8.1", "household contacts affected", "risk_factor", "diagnostic_feature", [
        "other family members also itching", "whole family itching",
        "partner also itching", "children also scratching",
    ]),

    # ── Measles (10.8) — additional edges ──
    # Rubella beats it because "red blotchy rash" is too generic
    ("10.8", "fever preceding rash", "symptom", "diagnostic_feature", [
        "fever before rash appeared", "high fever then rash came",
        "sick for days before rash", "fever for 3-4 days then rash",
    ]),

    # ── Herpes Zoster / Shingles (11.3.12) ──
    # Missing: patient language for dermatomal rash
    ("11.3.12", "dermatomal vesicular rash", "symptom", "diagnostic_feature", [
        "painful rash on one side of body", "blisters in a band",
        "rash on one side only", "rash in a line on my body",
    ]),
    ("11.3.12", "neuropathic pain", "symptom", "presenting_feature", [
        "burning pain on skin", "shooting pain along rash",
        "skin feels like burning", "nerve pain on chest",
    ]),

    # ── Croup (17.3) ──
    # Missing: patient language for stridor, barking cough
    ("17.3", "barking cough", "symptom", "diagnostic_feature", [
        "cough sounds like a dog", "seal-like cough",
        "harsh barking cough", "croupy cough",
    ]),
    ("17.3", "inspiratory stridor", "symptom", "diagnostic_feature", [
        "noisy breathing in", "squeaky breathing",
        "harsh sound when breathing in", "noisy breathing child",
    ]),
    ("17.3", "nocturnal onset", "symptom", "presenting_feature", [
        "child woke up at night struggling to breathe",
        "started in the middle of the night",
        "woke up with difficulty breathing", "worse at night",
    ]),

    # ── Meningitis, Acute (15.8) ──
    # Missing: patient language for meningeal signs
    ("15.8", "nuchal rigidity", "sign", "diagnostic_feature", [
        "stiff neck", "neck stiffness", "can't bend neck forward",
        "painful to move neck", "neck is stiff and sore",
    ]),
    ("15.8", "photophobia", "symptom", "diagnostic_feature", [
        "cannot look at light", "light hurts eyes",
        "eyes sensitive to light", "bright light makes headache worse",
    ]),
    ("15.8", "severe headache meningitis", "symptom", "presenting_feature", [
        "worst headache of life", "thunderclap headache",
        "worst headache ever", "sudden severe headache",
    ]),

    # ── Dementia (15.2) ──
    # Missing: patient language for cognitive decline
    ("15.2", "progressive memory loss", "symptom", "presenting_feature", [
        "becoming forgetful", "keeps forgetting things",
        "memory getting worse", "can't remember recent events",
    ]),
    ("15.2", "spatial disorientation", "symptom", "presenting_feature", [
        "gets lost easily", "gets lost in familiar places",
        "can't find way home", "confused about where they are",
    ]),
    ("15.2", "repetitive behaviour", "symptom", "presenting_feature", [
        "repeats questions", "asks same thing over and over",
        "tells same story repeatedly", "repeating himself",
    ]),
    ("15.2", "functional decline", "symptom", "presenting_feature", [
        "needs help with daily tasks", "can't manage alone",
        "needs help dressing", "can't cook for themselves anymore",
    ]),

    # ── Allergic Conjunctivitis (18.1) ──
    # Missing: patient language for itchy/watery eyes
    ("18.1", "ocular pruritus", "symptom", "diagnostic_feature", [
        "itchy watery eyes", "eyes itchy and watering",
        "rubbing eyes all the time", "itchy eyes",
    ]),
    ("18.1", "eyelid swelling allergic", "symptom", "presenting_feature", [
        "swollen eyelids", "puffy eyes", "eyelids swollen",
        "eyes swollen in morning",
    ]),

    # ── Diabetes Type 2 (9.2) ──
    # Missing: patient language for polydipsia/polyuria
    ("9.2", "polydipsia", "symptom", "presenting_feature", [
        "always thirsty", "drinking water all the time",
        "can't quench my thirst", "extremely thirsty",
    ]),
    ("9.2", "polyuria", "symptom", "presenting_feature", [
        "peeing a lot", "urinating very often",
        "passing urine many times", "always running to the toilet",
    ]),
    ("9.2", "unexplained weight loss diabetes", "symptom", "presenting_feature", [
        "losing weight without trying", "weight loss for no reason",
        "getting thinner without dieting", "losing weight unexpectedly",
    ]),

    # ── Acute Pain (20.1) ──
    # Missing: patient language for acute pain presentation
    ("20.1", "acute pain presentation", "symptom", "presenting_feature", [
        "bad pain", "just started hurting", "severe pain",
        "needs something for the pain", "unbearable pain",
    ]),

    # ── Male Urethritis Syndrome (12.3) ──
    # Missing: patient language for urethral symptoms
    ("12.3", "urethral discharge male", "symptom", "diagnostic_feature", [
        "yellow discharge from penis", "discharge from penis",
        "pus from penis", "drip from penis", "penile discharge",
    ]),
    ("12.3", "dysuria male", "symptom", "presenting_feature", [
        "burning passing urine", "pain when peeing man",
        "burns when I pee", "stinging when urinating",
    ]),

    # ── Emergency Contraception (7.4) ──
    # Missing: patient language for morning-after pill
    ("7.4", "emergency contraception request", "symptom", "presenting_feature", [
        "need morning after pill", "morning-after pill",
        "unprotected sex last night", "condom broke need help",
        "forgot to take pill", "missed pill had sex",
    ]),

    # ── COPD (17.1.5) ──
    # Missing: patient language for chronic productive cough
    ("17.1.5", "chronic productive cough", "symptom", "diagnostic_feature", [
        "morning cough every day with sputum", "coughing up phlegm every morning",
        "chronic cough with mucus", "daily cough for years",
    ]),
    ("17.1.5", "smoking history", "risk_factor", "diagnostic_feature", [
        "long smoking history", "smoked for many years",
        "heavy smoker", "pack a day for 20 years",
    ]),

    # ── Bronchiolitis in Children (17.1.4) ──
    # Missing: patient language for infant respiratory distress
    ("17.1.4", "tachypnoea infant", "symptom", "presenting_feature", [
        "baby coughing and breathing fast", "baby breathing very fast",
        "infant breathing quickly", "rapid breathing in baby",
    ]),
    ("17.1.4", "nasal flaring infant", "symptom", "diagnostic_feature", [
        "nasal flaring", "nostrils flaring when breathing",
        "nose flaring baby", "nostrils opening wide",
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
