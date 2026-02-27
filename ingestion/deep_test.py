#!/usr/bin/env python3
"""
CareMate Deep Test Suite
========================
Comprehensive evaluation of the triage search pipeline across all STG chapters.
Tests 60 conditions using realistic patient-language symptom descriptions,
with correct STG codes verified against the actual database.

Evaluates:
  1. Graph search (clinical_relationships + clinical_entities)
  2. Text search (knowledge_chunks ILIKE)
  3. Vector search (Voyage AI embeddings, if populated)

Scoring:
  - Graph: weight from clinical_relationships
      INDICATES / RED_FLAG with diagnostic_feature = 0.18
      INDICATES / RED_FLAG with presenting_feature = 0.12
      associated_feature / other = 0.08
      RED_FLAG bonus: +0.10
  - Text search: +0.05 per chunk match
  - Vector search: similarity * 0.15
  - Multi-method boost: condition found by 2+ methods -> score * 1.3

Matching:
  A test PASSES if any condition in the top-5 combined results has an
  stg_code that STARTS WITH the expected code prefix.
  E.g. expected "4.7" matches "4.7", "4.7.1", "4.7.2".

Reports:
  - Hit rates: top-1, top-3, top-5
  - Per-chapter breakdown
  - Missed conditions with diagnostic detail
  - Search method attribution (graph vs text vs vector)

Usage:
    python3 ingestion/deep_test.py
    python3 ingestion/deep_test.py --verbose           # show per-case detail
    python3 ingestion/deep_test.py --chapter Cardiovascular  # test single chapter
    python3 ingestion/deep_test.py --json              # output JSON report
    python3 ingestion/deep_test.py --list              # list all test cases
"""

import asyncio
import asyncpg
import json
import os
import sys
import time
import argparse
from collections import defaultdict
from typing import Optional

# -- Path setup ----------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()


# == Test Case Definitions =====================================================
# Each case uses patient language (not clinical jargon).
# stg_code values are verified against the real database.
# chapter_name is the human-readable chapter group (used for --chapter filter).

TEST_CASES = [
    # ==========================================================================
    # Dental and Oral Conditions (1.x)
    # ==========================================================================
    {
        "id": "ORAL-01",
        "chapter_name": "Dental and Oral Conditions",
        "condition_name": "Candidiasis, Oral (Thrush)",
        "stg_code": "1.2",
        "symptoms": ["white patches in mouth", "sore mouth", "difficulty eating"],
    },
    {
        "id": "ORAL-02",
        "chapter_name": "Dental and Oral Conditions",
        "condition_name": "Gingivitis, Uncomplicated",
        "stg_code": "1.3",
        "symptoms": ["bleeding gums", "swollen gums", "bad breath"],
    },
    {
        "id": "ORAL-03",
        "chapter_name": "Dental and Oral Conditions",
        "condition_name": "Herpes Simplex of Mouth and Lips",
        "stg_code": "1.4",
        "symptoms": ["cold sore on lip", "painful blisters on mouth", "tingling before sore appeared"],
    },
    # ==========================================================================
    # Gastro-Intestinal Conditions (2.x)
    # ==========================================================================
    {
        "id": "GI-01",
        "chapter_name": "Gastro-Intestinal Conditions",
        "condition_name": "Abdominal Pain",
        "stg_code": "2.1",
        "symptoms": ["tummy pain", "cramping in belly", "stomach sore"],
    },
    {
        "id": "GI-02",
        "chapter_name": "Gastro-Intestinal Conditions",
        "condition_name": "Peptic Ulcer Disease (Acute)",
        "stg_code": "2.3",
        "symptoms": ["burning tummy pain", "worse when hungry", "pain in upper belly"],
    },
    {
        "id": "GI-04",
        "chapter_name": "Gastro-Intestinal Conditions",
        "condition_name": "Diarrhoea, Acute in Children",
        "stg_code": "2.9",
        "symptoms": ["runny tummy", "watery stools", "tummy cramps", "child not drinking"],
    },
    # ==========================================================================
    # Nutrition and Anaemia (3.x)
    # ==========================================================================
    {
        "id": "NUTR-01",
        "chapter_name": "Nutrition and Anaemia",
        "condition_name": "Anaemia, Iron Deficiency",
        "stg_code": "3.1",
        "symptoms": ["feeling very tired", "pale skin", "short of breath on walking"],
    },
    {
        "id": "NUTR-02",
        "chapter_name": "Nutrition and Anaemia",
        "condition_name": "Anaemia, Macrocytic or Megaloblastic",
        "stg_code": "3.1.2",
        "symptoms": ["tiredness", "sore tongue", "tingling hands and feet", "pins and needles"],
    },
    {
        "id": "NUTR-03",
        "chapter_name": "Nutrition and Anaemia",
        "condition_name": "Worm Infestation",
        "stg_code": "3.3",
        "symptoms": ["tummy pain", "worms in stool", "itchy bottom"],
    },

    # ==========================================================================
    # Cardiovascular Conditions (4.x)
    # ==========================================================================
    {
        "id": "CVS-01",
        "chapter_name": "Cardiovascular Conditions",
        "condition_name": "Angina Pectoris, Stable",
        "stg_code": "4.2",
        "symptoms": ["chest pain when walking", "angina", "pain goes away with rest"],
    },
    {
        "id": "CVS-02",
        "chapter_name": "Cardiovascular Conditions",
        "condition_name": "Cardiac Failure, Congestive (CCF)",
        "stg_code": "4.5",
        "symptoms": ["swollen ankles", "short of breath lying down", "feeling tired all the time"],
    },
    {
        "id": "CVS-03",
        "chapter_name": "Cardiovascular Conditions",
        "condition_name": "Hypertension In Adults",
        "stg_code": "4.7",
        "symptoms": ["high blood pressure", "headache", "dizziness"],
    },
    # ==========================================================================
    # Dermatological Conditions (5.x)
    # ==========================================================================
    {
        "id": "DERM-01",
        "chapter_name": "Dermatological Conditions",
        "condition_name": "Eczema, Acute, Moist or Weeping",
        "stg_code": "5.8.2",
        "symptoms": ["itchy rash", "weeping skin", "rash in skin folds"],
    },
    {
        "id": "DERM-02",
        "chapter_name": "Dermatological Conditions",
        "condition_name": "Boil / Abscess",
        "stg_code": "5.4",
        "symptoms": ["painful lump on skin", "red swelling with pus", "boil getting bigger"],
    },
    {
        "id": "DERM-03",
        "chapter_name": "Dermatological Conditions",
        "condition_name": "Scabies",
        "stg_code": "5.7.2",
        "symptoms": ["itching worse at night", "rash between fingers", "other family members also itching"],
    },
    {
        "id": "DERM-04",
        "chapter_name": "Dermatological Conditions",
        "condition_name": "Dermatitis, Seborrhoeic",
        "stg_code": "5.8.3",
        "symptoms": ["flaky scalp", "oily scales on face", "red patches on skin"],
    },
    # ==========================================================================
    # Obstetrics and Gynaecology (6.x) -- labeled as Endocrine in DB
    # ==========================================================================
    {
        "id": "OBGYN-01",
        "chapter_name": "Obstetrics and Gynaecology",
        "condition_name": "Miscarriage",
        "stg_code": "6.1",
        "symptoms": ["bleeding in pregnancy", "cramps in lower tummy", "pregnant and bleeding"],
    },
    {
        "id": "OBGYN-02",
        "chapter_name": "Obstetrics and Gynaecology",
        "condition_name": "Pre-eclampsia",
        "stg_code": "6.4.2.4",
        "symptoms": ["pregnant", "high blood pressure", "swollen feet", "headache"],
    },
    {
        "id": "OBGYN-03",
        "chapter_name": "Obstetrics and Gynaecology",
        "condition_name": "Mastitis",
        "stg_code": "6.7.4",
        "symptoms": ["painful breast while breastfeeding", "red hot area on breast", "fever"],
    },
    {
        "id": "OBGYN-04",
        "chapter_name": "Obstetrics and Gynaecology",
        "condition_name": "Menorrhagia (Heavy Menstrual Bleeding)",
        "stg_code": "6.11.2",
        "symptoms": ["very heavy periods", "soaking through pads", "bleeding for more than seven days"],
    },

    # ==========================================================================
    # Contraception (7.x)
    # ==========================================================================
    {
        "id": "CONTRA-01",
        "chapter_name": "Contraception",
        "condition_name": "Contraception, Injectable (Progestin-only)",
        "stg_code": "7.2.3",
        "symptoms": ["wants injection for family planning", "prevent pregnancy with injection"],
    },
    {
        "id": "CONTRA-02",
        "chapter_name": "Contraception",
        "condition_name": "Contraception, Emergency",
        "stg_code": "7.4",
        "symptoms": ["unprotected sex", "need morning after pill", "forgot to take pill"],
    },

    # ==========================================================================
    # Urinary Conditions (8.x)
    # ==========================================================================
    {
        "id": "URIN-01",
        "chapter_name": "Urinary Conditions",
        "condition_name": "Urinary Tract Infection (UTI)",
        "stg_code": "8.4",
        "symptoms": ["burning when I pee", "need to pee all the time", "smelly urine"],
    },
    {
        "id": "URIN-02",
        "chapter_name": "Urinary Conditions",
        "condition_name": "Prostatitis, Acute Bacterial",
        "stg_code": "8.5",
        "symptoms": ["pain between legs", "trouble peeing", "fever", "pain in lower back"],
    },

    # ==========================================================================
    # Endocrine / Diabetes (9.x) -- labeled as Obstetrics in DB
    # ==========================================================================
    {
        "id": "ENDO-01",
        "chapter_name": "Endocrine Conditions",
        "condition_name": "Type 2 Diabetes Mellitus, Adults",
        "stg_code": "9.2.1",
        "symptoms": ["always thirsty", "peeing a lot", "losing weight without trying"],
    },
    {
        "id": "ENDO-02",
        "chapter_name": "Endocrine Conditions",
        "condition_name": "Hypoglycaemia in Diabetics",
        "stg_code": "9.3",
        "symptoms": ["shaking and sweating", "feeling dizzy", "diabetic feeling confused"],
    },
    {
        "id": "ENDO-03",
        "chapter_name": "Endocrine Conditions",
        "condition_name": "Hypothyroidism in Adults",
        "stg_code": "9.6.3",
        "symptoms": ["gaining weight", "always cold", "very tired", "constipation"],
    },

    # ==========================================================================
    # Infectious Conditions (10.x)
    # ==========================================================================
    {
        "id": "INF-01",
        "chapter_name": "Infectious Conditions",
        "condition_name": "Chickenpox (Varicella)",
        "stg_code": "10.3",
        "symptoms": ["itchy blisters all over body", "fever", "rash started on trunk"],
    },
    {
        "id": "INF-02",
        "chapter_name": "Infectious Conditions",
        "condition_name": "Malaria",
        "stg_code": "10.6",
        "symptoms": ["fever and chills", "body aches", "travelled to malaria area"],
    },
    {
        "id": "INF-03",
        "chapter_name": "Infectious Conditions",
        "condition_name": "Tick Bite Fever",
        "stg_code": "10.14",
        "symptoms": ["bite mark with black scab", "fever after being in the bush", "headache and muscle pain"],
    },

    # ==========================================================================
    # HIV and AIDS (11.x)
    # ==========================================================================
    {
        "id": "HIV-01",
        "chapter_name": "HIV and AIDS",
        "condition_name": "Antiretroviral Therapy (ART)",
        "stg_code": "11.1",
        "symptoms": ["hiv positive", "weight loss", "recurrent infections"],
    },
    {
        "id": "HIV-02",
        "chapter_name": "HIV and AIDS",
        "condition_name": "Candidiasis, Oesophageal",
        "stg_code": "11.3.2",
        "symptoms": ["pain when swallowing", "difficulty swallowing", "hiv positive", "white patches in throat"],
    },
    {
        "id": "HIV-03",
        "chapter_name": "HIV and AIDS",
        "condition_name": "Herpes Zoster (Shingles)",
        "stg_code": "11.3.12",
        "symptoms": ["painful rash on one side of body", "blisters in a band", "burning pain on skin"],
    },

    # ==========================================================================
    # Sexually Transmitted Infections (12.x)
    # ==========================================================================
    {
        "id": "STI-01",
        "chapter_name": "Sexually Transmitted Infections",
        "condition_name": "Vaginal Discharge Syndrome (VDS)",
        "stg_code": "12.1",
        "symptoms": ["vaginal discharge", "bad smell down there", "itching"],
    },
    {
        "id": "STI-02",
        "chapter_name": "Sexually Transmitted Infections",
        "condition_name": "Lower Abdominal Pain (LAP) - PID",
        "stg_code": "12.2",
        "symptoms": ["lower tummy pain", "pain during sex", "abnormal discharge", "fever"],
    },
    {
        "id": "STI-03",
        "chapter_name": "Sexually Transmitted Infections",
        "condition_name": "Male Urethritis Syndrome (MUS)",
        "stg_code": "12.3",
        "symptoms": ["burning when peeing", "discharge from penis", "pain passing urine"],
    },
    {
        "id": "STI-04",
        "chapter_name": "Sexually Transmitted Infections",
        "condition_name": "Genital Ulcer Syndrome (GUS)",
        "stg_code": "12.5",
        "symptoms": ["sore on private parts", "painful ulcer down below"],
    },

    # ==========================================================================
    # Immunisation (13.x)
    # ==========================================================================
    {
        "id": "IMMUN-01",
        "chapter_name": "Immunisation",
        "condition_name": "Childhood Immunisation Schedule (EPI)",
        "stg_code": "13.1",
        "symptoms": ["baby needs vaccinations", "child immunisation schedule"],
    },
    {
        "id": "IMMUN-02",
        "chapter_name": "Immunisation",
        "condition_name": "Tetanus Prophylaxis, Post-Trauma",
        "stg_code": "13.6",
        "symptoms": ["dirty wound", "stepped on nail", "cut with rusty metal"],
    },

    # ==========================================================================
    # Musculoskeletal Conditions (14.x)
    # ==========================================================================
    {
        "id": "MSK-01",
        "chapter_name": "Musculoskeletal Conditions",
        "condition_name": "Gout, Acute",
        "stg_code": "14.3",
        "symptoms": ["very painful big toe", "swollen red joint", "cannot walk on it"],
    },
    {
        "id": "MSK-02",
        "chapter_name": "Musculoskeletal Conditions",
        "condition_name": "Osteoarthrosis (Osteoarthritis)",
        "stg_code": "14.5",
        "symptoms": ["sore knees", "stiff joints in morning", "joint pain getting worse"],
    },

    # ==========================================================================
    # Central Nervous System Conditions (15.x)
    # ==========================================================================
    {
        "id": "CNS-01",
        "chapter_name": "Central Nervous System Conditions",
        "condition_name": "Epilepsy",
        "stg_code": "15.7",
        "symptoms": ["fits", "falling down and shaking", "blacking out"],
    },
    {
        "id": "CNS-02",
        "chapter_name": "Central Nervous System Conditions",
        "condition_name": "Meningitis, Acute",
        "stg_code": "15.8",
        "symptoms": ["stiff neck", "high fever", "worst headache of life", "cannot look at light"],
    },
    {
        "id": "CNS-03",
        "chapter_name": "Central Nervous System Conditions",
        "condition_name": "Headache, Mild, Non-Specific",
        "stg_code": "15.9",
        "symptoms": ["headache", "head is sore", "tension in my head"],
    },
    # ==========================================================================
    # Mental Health Conditions (16.x)
    # ==========================================================================
    {
        "id": "MH-01",
        "chapter_name": "Mental Health Conditions",
        "condition_name": "Depressive Disorders",
        "stg_code": "16.4.1",
        "symptoms": ["feeling sad all the time", "not enjoying anything", "trouble sleeping", "no energy"],
    },
    {
        "id": "MH-02",
        "chapter_name": "Mental Health Conditions",
        "condition_name": "Anxiety Disorders",
        "stg_code": "16.3",
        "symptoms": ["worrying all the time", "cannot relax", "heart racing", "trouble sleeping"],
    },
    {
        "id": "MH-03",
        "chapter_name": "Mental Health Conditions",
        "condition_name": "Schizophrenia Spectrum Disorders",
        "stg_code": "16.5.2",
        "symptoms": ["hearing voices", "suspicious of everyone", "talking to self"],
    },
    {
        "id": "MH-04",
        "chapter_name": "Mental Health Conditions",
        "condition_name": "Substance Use Disorders",
        "stg_code": "16.9",
        "symptoms": ["drinking too much alcohol", "cannot stop using drugs", "shaking hands in morning"],
    },

    # ==========================================================================
    # Respiratory Conditions (17.x)
    # ==========================================================================
    {
        "id": "RESP-01",
        "chapter_name": "Respiratory Conditions",
        "condition_name": "Acute Asthma / Acute Exacerbation of COPD, Adults",
        "stg_code": "17.1",
        "symptoms": ["wheezing", "tight chest", "short of breath", "cough at night"],
    },
    {
        "id": "RESP-02",
        "chapter_name": "Respiratory Conditions",
        "condition_name": "Pneumonia, Uncomplicated (Community Acquired, Adults)",
        "stg_code": "17.3.1",
        "symptoms": ["coughing up green stuff", "high fever", "chest pain when breathing"],
    },
    {
        "id": "RESP-03",
        "chapter_name": "Respiratory Conditions",
        "condition_name": "Pulmonary Tuberculosis (TB), Adults",
        "stg_code": "17.4.1",
        "symptoms": ["cough for more than two weeks", "night sweats", "losing weight"],
    },
    {
        "id": "RESP-04",
        "chapter_name": "Respiratory Conditions",
        "condition_name": "Croup",
        "stg_code": "17.3",
        "symptoms": ["barking cough", "noisy breathing", "child woke up at night struggling to breathe"],
    },

    # ==========================================================================
    # Eye Conditions (18.x)
    # ==========================================================================
    {
        "id": "EYE-01",
        "chapter_name": "Eye Conditions",
        "condition_name": "Conjunctivitis, Allergic",
        "stg_code": "18.1",
        "symptoms": ["itchy watery eyes", "swollen eyelids", "sneezing"],
    },
    {
        "id": "EYE-02",
        "chapter_name": "Eye Conditions",
        "condition_name": "Painful Red Eye",
        "stg_code": "18.5",
        "symptoms": ["red eye", "eye is sore", "blurred vision", "light hurts my eye"],
    },

    # ==========================================================================
    # Ear, Nose and Throat Conditions (19.x)
    # ==========================================================================
    {
        "id": "ENT-01",
        "chapter_name": "Ear, Nose and Throat Conditions",
        "condition_name": "Allergic Rhinitis",
        "stg_code": "19.1",
        "symptoms": ["runny nose", "sneezing a lot", "itchy nose"],
    },
    {
        "id": "ENT-02",
        "chapter_name": "Ear, Nose and Throat Conditions",
        "condition_name": "Common Cold (Viral Rhinitis)",
        "stg_code": "19.2",
        "symptoms": ["blocked nose", "sneezing", "mild sore throat", "cough"],
    },
    {
        "id": "ENT-03",
        "chapter_name": "Ear, Nose and Throat Conditions",
        "condition_name": "Otitis Media, Acute",
        "stg_code": "19.4.2",
        "symptoms": ["ear pain", "fever", "child pulling at ear"],
    },
    {
        "id": "ENT-04",
        "chapter_name": "Ear, Nose and Throat Conditions",
        "condition_name": "Tonsillitis and Pharyngitis",
        "stg_code": "19.6",
        "symptoms": ["sore throat", "painful swallowing", "fever", "tonsils swollen"],
    },

    # ==========================================================================
    # Pain (20.x)
    # ==========================================================================
    {
        "id": "PAIN-01",
        "chapter_name": "Pain",
        "condition_name": "Acute Pain",
        "stg_code": "20.1",
        "symptoms": ["bad pain", "just started hurting", "needs something for the pain"],
    },
    {
        "id": "PAIN-02",
        "chapter_name": "Pain",
        "condition_name": "Chronic Non-Cancer Pain",
        "stg_code": "20.3",
        "symptoms": ["pain for months", "nothing helps the pain", "cannot sleep from pain"],
    },

    # ==========================================================================
    # Emergencies and Injuries (21.x)
    # ==========================================================================
    {
        "id": "EMERG-01",
        "chapter_name": "Emergencies and Injuries",
        "condition_name": "Anaphylaxis",
        "stg_code": "21.2.10",
        "symptoms": ["swollen face and lips", "cannot breathe", "rash all over body", "collapsed"],
    },
    {
        "id": "EMERG-02",
        "chapter_name": "Emergencies and Injuries",
        "condition_name": "Burns",
        "stg_code": "21.3.2",
        "symptoms": ["burnt by hot water", "burn on arm", "very painful burn"],
    },
    {
        "id": "EMERG-03",
        "chapter_name": "Emergencies and Injuries",
        "condition_name": "Paracetamol Poisoning",
        "stg_code": "21.3.4",
        "symptoms": ["swallowed too many pills", "took whole bottle of panado", "tummy pain after pills"],
    },

    # ==========================================================================
    # Palliative Care (22.x)
    # ==========================================================================
    {
        "id": "PALL-01",
        "chapter_name": "Palliative Care",
        "condition_name": "Constipation, Palliative Care",
        "stg_code": "22.1",
        "symptoms": ["has not had a bowel movement in days", "tummy bloated", "cancer patient constipated"],
    },
    {
        "id": "PALL-02",
        "chapter_name": "Palliative Care",
        "condition_name": "Depression (Palliative Care)",
        "stg_code": "22.2.3",
        "symptoms": ["terminal patient feeling hopeless", "no will to live", "crying all day"],
    },
]


# == Search Layer ==============================================================

async def graph_search(conn: asyncpg.Connection, symptoms: list[str]) -> dict:
    """
    Search via clinical_relationships + clinical_entities.
    Returns ranked list of {id, name, stg_code, score, matched_symptoms, method}.

    Uses IDF-like weighting: symptoms matching many conditions get penalized,
    while specific symptoms get boosted. Also rewards conditions matching
    multiple distinct symptoms.
    """
    results = {}

    for symptom in symptoms:
        term = symptom.lower().strip()
        if not term:
            continue

        rows = await conn.fetch("""
            SELECT cr.condition_id, c.name, c.stg_code,
                   cr.relationship_type, cr.feature_type,
                   ce.canonical_name
            FROM clinical_relationships cr
            JOIN conditions c ON c.id = cr.condition_id
            JOIN clinical_entities ce ON ce.id = cr.source_entity_id
            WHERE (ce.canonical_name ILIKE $1 OR ce.aliases::text ILIKE $1)
              AND cr.relationship_type IN ('INDICATES', 'RED_FLAG', 'ASSOCIATED_WITH', 'RISK_FACTOR')
        """, f"%{term}%")

        # IDF-like penalty: if a symptom matches too many conditions,
        # each match is worth less (prevents "fever" from swamping results)
        n_conditions = len(set(r["condition_id"] for r in rows))
        if n_conditions > 50:
            idf_factor = 0.15  # very generic (e.g. "fever", "pain")
        elif n_conditions > 20:
            idf_factor = 0.4   # somewhat generic
        elif n_conditions > 10:
            idf_factor = 0.7   # moderate specificity
        else:
            idf_factor = 1.0   # specific symptom — full weight

        for row in rows:
            cid = row["condition_id"]
            if cid not in results:
                results[cid] = {
                    "id": cid,
                    "name": row["name"],
                    "stg_code": row["stg_code"],
                    "score": 0.0,
                    "matched_symptoms": set(),
                    "method": "graph",
                }

            # Weight by feature_type, scaled by IDF
            ft = row["feature_type"] or ""
            if ft == "diagnostic_feature":
                results[cid]["score"] += 0.18 * idf_factor
            elif ft == "presenting_feature":
                results[cid]["score"] += 0.12 * idf_factor
            else:
                results[cid]["score"] += 0.08 * idf_factor

            # RED_FLAG bonus (not scaled by IDF — red flags always matter)
            if row["relationship_type"] == "RED_FLAG":
                results[cid]["score"] += 0.10

            results[cid]["matched_symptoms"].add(term)

    # Multi-symptom match bonus: reward conditions matching more distinct symptoms
    n_query_symptoms = len(symptoms)
    for cid in results:
        n_matched = len(results[cid]["matched_symptoms"])
        if n_matched >= 2:
            # Proportional bonus: matching 3/3 symptoms >> matching 1/3
            coverage = n_matched / max(n_query_symptoms, 1)
            results[cid]["score"] *= (1.0 + coverage)  # e.g. 3/3 = 2x, 2/3 = 1.67x

    # Convert sets, sort descending
    ranked = sorted(results.values(), key=lambda x: x["score"], reverse=True)
    for r in ranked:
        r["matched_symptoms"] = sorted(r["matched_symptoms"])
    return ranked


async def text_search(conn: asyncpg.Connection, symptoms: list[str]) -> list[dict]:
    """
    Text search across knowledge_chunks with IDF-like weighting.
    Generic terms matching many conditions get penalized.
    Conditions matching multiple distinct symptoms get boosted.
    """
    results = {}

    n_query_symptoms = len(symptoms)

    for symptom in symptoms:
        term = symptom.lower().strip()
        if len(term) < 3:
            continue

        rows = await conn.fetch("""
            SELECT kc.condition_id, c.name, c.stg_code
            FROM knowledge_chunks kc
            JOIN conditions c ON c.id = kc.condition_id
            WHERE kc.chunk_text ILIKE $1
        """, f"%{term}%")

        # IDF-like penalty for generic terms
        n_conditions = len(set(r["condition_id"] for r in rows))
        if n_conditions > 100:
            idf_factor = 0.10   # extremely generic
        elif n_conditions > 50:
            idf_factor = 0.20   # very generic
        elif n_conditions > 20:
            idf_factor = 0.45
        elif n_conditions > 10:
            idf_factor = 0.70
        else:
            idf_factor = 1.0    # specific — full weight

        # Also cap per-condition hits (multiple chunks shouldn't over-boost)
        condition_hits = defaultdict(int)
        for row in rows:
            condition_hits[row["condition_id"]] += 1

        for row in rows:
            cid = row["condition_id"]
            if cid not in results:
                results[cid] = {
                    "id": cid,
                    "name": row["name"],
                    "stg_code": row["stg_code"],
                    "score": 0.0,
                    "matched_symptoms": set(),
                    "method": "text",
                }
            # Diminishing returns for multiple chunk hits on same term
            hits = condition_hits[cid]
            per_hit = 0.05 * idf_factor / max(hits, 1)  # split score across hits
            results[cid]["score"] += per_hit
            results[cid]["matched_symptoms"].add(term)

    # Multi-symptom match bonus (same as graph search)
    for cid in results:
        n_matched = len(results[cid]["matched_symptoms"])
        if n_matched >= 2:
            coverage = n_matched / max(n_query_symptoms, 1)
            results[cid]["score"] *= (1.0 + coverage)

    ranked = sorted(results.values(), key=lambda x: x["score"], reverse=True)
    for r in ranked:
        r["matched_symptoms"] = sorted(r["matched_symptoms"])
    return ranked


async def vector_search(conn: asyncpg.Connection, symptoms: list[str]) -> list[dict]:
    """
    Vector (semantic) search using Voyage AI embeddings.
    Returns empty list if embeddings not available.
    Score = similarity * 0.15.
    """
    try:
        from agents.embeddings import get_embedding
    except ImportError:
        return []

    query_text = " ".join(symptoms)
    query_embedding = await get_embedding(query_text)
    if not query_embedding:
        return []

    # Check if embeddings exist
    has_embeddings = await conn.fetchval(
        "SELECT EXISTS(SELECT 1 FROM knowledge_chunks WHERE embedding IS NOT NULL LIMIT 1)"
    )
    if not has_embeddings:
        return []

    vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    rows = await conn.fetch("""
        SELECT kc.condition_id, c.name, c.stg_code,
               1 - (kc.embedding <=> $1::vector) as similarity
        FROM knowledge_chunks kc
        JOIN conditions c ON c.id = kc.condition_id
        WHERE kc.embedding IS NOT NULL
        ORDER BY kc.embedding <=> $1::vector
        LIMIT 20
    """, vec_str)

    # Group by condition, keep best similarity
    seen = {}
    for r in rows:
        sim = float(r["similarity"])
        if sim < 0.35:
            continue
        cid = r["condition_id"]
        candidate_score = sim * 0.25
        if cid not in seen or candidate_score > seen[cid]["score"]:
            seen[cid] = {
                "id": cid,
                "name": r["name"],
                "stg_code": r["stg_code"],
                "score": candidate_score,
                "method": "vector",
            }

    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)


# == Combined Search (merge all methods) =======================================

async def synonym_search(conn: asyncpg.Connection, symptoms: list[str]) -> list[dict]:
    """
    Match patient-language symptoms directly to conditions via synonym_rings.
    synonym_rings maps patient language → canonical_term (condition/concept name).
    We then find the actual condition by matching canonical_term to conditions.name.
    Score: 0.20 per match (strong signal — patient language maps directly to condition).
    """
    results = {}
    n_query = len(symptoms)

    for symptom in symptoms:
        term = symptom.lower().strip()
        if len(term) < 3:
            continue

        # Find canonical terms matching this patient-language symptom
        # Search both directions:
        #   1. synonym contains patient term (e.g., "painful blisters on lips" contains "blisters")
        #   2. patient term contains synonym (e.g., "boil getting bigger" contains "boil")
        # For direction 2, require minimum 4 chars to avoid matching generic short words
        rows = await conn.fetch("""
            SELECT DISTINCT sr.canonical_term
            FROM synonym_rings sr
            WHERE sr.synonym ILIKE $1
               OR ($2 ILIKE '%' || sr.synonym || '%' AND length(sr.synonym) >= 4)
        """, f"%{term}%", term)

        if not rows:
            continue

        # IDF-like: if too many canonical terms match, each is worth less
        n_canonical = len(rows)
        if n_canonical > 30:
            idf = 0.05   # extremely generic (e.g. "pain", "fever")
        elif n_canonical > 20:
            idf = 0.12
        elif n_canonical > 10:
            idf = 0.30
        elif n_canonical > 5:
            idf = 0.60
        else:
            idf = 1.0    # specific — full weight

        for r in rows:
            canonical = r["canonical_term"]

            # Find conditions matching this canonical term
            # Step 1: Direct condition name match
            conds = await conn.fetch("""
                SELECT id, name, stg_code FROM conditions
                WHERE name ILIKE $1
                LIMIT 5
            """, f"%{canonical}%")
            score_per_hit = 0.20  # direct name match

            # Step 2: Two-hop entity resolution if name match fails
            # canonical → clinical_entity → clinical_relationships → condition
            if not conds:
                conds = await conn.fetch("""
                    SELECT DISTINCT c.id, c.name, c.stg_code
                    FROM clinical_entities ce
                    JOIN clinical_relationships cr
                        ON (cr.source_entity_id = ce.id OR cr.target_entity_id = ce.id)
                    JOIN conditions c ON cr.condition_id = c.id
                    WHERE ce.canonical_name ILIKE $1
                    AND cr.feature_type IN ('presenting_feature', 'diagnostic_feature')
                    LIMIT 10
                """, f"%{canonical}%")
                score_per_hit = 0.15  # indirect two-hop match

            for c in conds:
                cid = c["id"]
                if cid not in results:
                    results[cid] = {
                        "id": cid,
                        "name": c["name"],
                        "stg_code": c["stg_code"],
                        "score": 0.0,
                        "matched_symptoms": set(),
                        "method": "synonym",
                    }
                results[cid]["score"] += score_per_hit * idf
                results[cid]["matched_symptoms"].add(term)

    # Multi-symptom match bonus
    for cid in results:
        n_matched = len(results[cid]["matched_symptoms"])
        if n_matched >= 2:
            coverage = n_matched / max(n_query, 1)
            results[cid]["score"] *= (1.0 + coverage)

    ranked = sorted(results.values(), key=lambda x: x["score"], reverse=True)
    for r in ranked:
        r["matched_symptoms"] = sorted(r["matched_symptoms"])
    return ranked


async def combined_search(conn: asyncpg.Connection, symptoms: list[str]) -> dict:
    """
    Run graph + text + vector + synonym, merge results.
    Multi-method boost: if found by 2+ methods, score *= 1.3
    Returns {"merged": [...], "graph_count": N, "text_count": N, "vector_count": N, "synonym_count": N}
    """
    graph_results = await graph_search(conn, symptoms)
    text_results = await text_search(conn, symptoms)
    vector_results = await vector_search(conn, symptoms)
    synonym_results = await synonym_search(conn, symptoms)

    merged = {}
    method_map = {}  # cid -> set of methods

    for source, results in [
        ("graph", graph_results),
        ("text", text_results),
        ("vector", vector_results),
        ("synonym", synonym_results),
    ]:
        for r in results:
            cid = r["id"]
            if cid not in method_map:
                method_map[cid] = set()
            method_map[cid].add(source)

            if cid not in merged:
                merged[cid] = {
                    "id": cid,
                    "name": r["name"],
                    "stg_code": r["stg_code"],
                    "score": 0.0,
                    "primary_method": source,
                }
            # Sum scores from all methods
            merged[cid]["score"] += r["score"]

    # Multi-method boost
    for cid in merged:
        methods = method_map.get(cid, set())
        if len(methods) >= 2:
            merged[cid]["score"] *= 1.3
        merged[cid]["found_by"] = sorted(methods)

    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

    return {
        "merged": ranked,
        "graph_count": len(graph_results),
        "text_count": len(text_results),
        "vector_count": len(vector_results),
        "synonym_count": len(synonym_results),
    }


# == STG Code Matching =========================================================

def stg_code_matches(actual_code: str, expected_prefix: str) -> bool:
    """
    Flexible STG code matching.
    Returns True if actual_code starts with expected_prefix.
    E.g. expected "4.7" matches "4.7", "4.7.1", "4.7.2".
    """
    if not actual_code or not expected_prefix:
        return False
    # Exact match or prefix match (must be followed by '.' or end of string)
    if actual_code == expected_prefix:
        return True
    if actual_code.startswith(expected_prefix + "."):
        return True
    # Also check if expected is more specific and the actual is a parent
    if expected_prefix.startswith(actual_code + "."):
        return True
    return False


# == Test Runner ===============================================================

async def run_single_test(conn: asyncpg.Connection, test_case: dict, verbose: bool = False) -> dict:
    """
    Run a single test case.
    PASSES if any condition in top-5 combined results has stg_code starting
    with the expected prefix.
    """
    tc_id = test_case["id"]
    expected_code = test_case["stg_code"]
    symptoms = test_case["symptoms"]
    chapter_name = test_case["chapter_name"]

    # Verify the condition exists in DB (using prefix match)
    condition = await conn.fetchrow(
        "SELECT id, name, stg_code FROM conditions WHERE stg_code = $1",
        expected_code
    )
    if not condition:
        # Try prefix match
        condition = await conn.fetchrow(
            "SELECT id, name, stg_code FROM conditions WHERE stg_code LIKE $1 ORDER BY stg_code LIMIT 1",
            f"{expected_code}%"
        )
    if not condition:
        # Try matching by condition name
        name_key = test_case["condition_name"].split(",")[0].split("(")[0].strip()
        condition = await conn.fetchrow(
            "SELECT id, name, stg_code FROM conditions WHERE name ILIKE $1 LIMIT 1",
            f"%{name_key}%"
        )

    db_found = condition is not None
    actual_stg = condition["stg_code"] if condition else expected_code
    actual_name = condition["name"] if condition else test_case["condition_name"]

    # Run combined search
    search = await combined_search(conn, symptoms)
    top5 = search["merged"][:5]

    # Check if any top-5 result matches the expected code prefix
    rank_in_top5 = None
    matched_entry = None
    for i, entry in enumerate(top5):
        if stg_code_matches(entry["stg_code"], expected_code):
            rank_in_top5 = i + 1
            matched_entry = entry
            break

    # Also find the absolute rank in full results
    absolute_rank = None
    for i, entry in enumerate(search["merged"]):
        if stg_code_matches(entry["stg_code"], expected_code):
            absolute_rank = i + 1
            break

    # Determine status
    if not db_found:
        status = "NOT_IN_DB"
    elif rank_in_top5 == 1:
        status = "TOP_1"
    elif rank_in_top5 is not None and rank_in_top5 <= 3:
        status = "TOP_3"
    elif rank_in_top5 is not None and rank_in_top5 <= 5:
        status = "TOP_5"
    elif absolute_rank is not None and absolute_rank <= 10:
        status = "TOP_10"
    elif absolute_rank is not None:
        status = f"RANK_{absolute_rank}"
    else:
        status = "MISS"

    # Edge count for diagnostics
    edge_count = None
    if condition:
        edge_count = await conn.fetchval(
            "SELECT COUNT(*) FROM clinical_relationships WHERE condition_id = $1",
            condition["id"]
        )

    result = {
        "id": tc_id,
        "stg_code": actual_stg,
        "expected_code": expected_code,
        "condition_name": actual_name,
        "chapter_name": chapter_name,
        "symptoms_searched": symptoms,
        "status": status,
        "rank_in_top5": rank_in_top5,
        "absolute_rank": absolute_rank,
        "found_by": matched_entry["found_by"] if matched_entry else [],
        "edge_count": edge_count,
        "search_counts": {
            "graph": search["graph_count"],
            "text": search["text_count"],
            "vector": search["vector_count"],
            "synonym": search.get("synonym_count", 0),
        },
    }

    if verbose or status in ("MISS", "NOT_IN_DB"):
        result["top_5_actual"] = [
            {
                "name": r["name"],
                "stg_code": r["stg_code"],
                "score": round(r["score"], 4),
                "found_by": r.get("found_by", []),
            }
            for r in top5
        ]

    return result


async def run_all_tests(conn: asyncpg.Connection, verbose: bool = False,
                        chapter_filter: Optional[str] = None) -> list[dict]:
    """Run all test cases, optionally filtered by chapter_name substring."""
    cases = TEST_CASES
    if chapter_filter:
        filter_lower = chapter_filter.lower()
        cases = [tc for tc in cases if filter_lower in tc["chapter_name"].lower()]

    results = []
    total = len(cases)

    for i, tc in enumerate(cases, 1):
        prefix = f"[{i}/{total}]"
        sys.stdout.write(f"\r  {prefix} Testing {tc['id']}: {tc['condition_name'][:45]}...")
        sys.stdout.flush()
        result = await run_single_test(conn, tc, verbose)
        results.append(result)

    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()
    return results


# == Report Generators =========================================================

def print_report(results: list[dict], verbose: bool = False):
    """Print a comprehensive human-readable report."""

    testable = [r for r in results if r["status"] != "NOT_IN_DB"]
    not_in_db = [r for r in results if r["status"] == "NOT_IN_DB"]

    total = len(testable)
    if total == 0:
        print("\n  No testable conditions found in database.")
        if not_in_db:
            print(f"  {len(not_in_db)} conditions not in DB:")
            for r in not_in_db:
                print(f"    - {r['expected_code']}: {r['condition_name']}")
        return

    # -- Overall Hit Rates -----------------------------------------------------
    top1 = sum(1 for r in testable if r["status"] == "TOP_1")
    top3 = sum(1 for r in testable if r["status"] in ("TOP_1", "TOP_3"))
    top5 = sum(1 for r in testable if r["status"] in ("TOP_1", "TOP_3", "TOP_5"))
    top10_statuses = {"TOP_1", "TOP_3", "TOP_5", "TOP_10"}
    top10 = sum(1 for r in testable if r["status"] in top10_statuses)
    misses = [r for r in testable if r["status"] == "MISS"]

    print(f"\n{'='*72}")
    print("  CAREMATE DEEP TEST REPORT")
    print(f"{'='*72}")
    print(f"\n  Conditions tested:  {total}")
    print(f"  Not in database:   {len(not_in_db)}")
    print(f"  Total test cases:  {len(results)}")

    print(f"\n  {'_'*40}")
    print(f"  OVERALL HIT RATES (combined search, top-5 pass)")
    print(f"  {'_'*40}")
    print(f"  Top-1:   {top1:>3}/{total}  ({100*top1/total:5.1f}%)")
    print(f"  Top-3:   {top3:>3}/{total}  ({100*top3/total:5.1f}%)")
    print(f"  Top-5:   {top5:>3}/{total}  ({100*top5/total:5.1f}%)")
    print(f"  Top-10:  {top10:>3}/{total}  ({100*top10/total:5.1f}%)")
    print(f"  Miss:    {len(misses):>3}/{total}  ({100*len(misses)/total:5.1f}%)")

    # Average rank of found conditions
    found = [r for r in testable if r["absolute_rank"] is not None]
    if found:
        avg_rank = sum(r["absolute_rank"] for r in found) / len(found)
        print(f"\n  Average rank (found): {avg_rank:.1f}")

    # -- Search Method Attribution ---------------------------------------------
    print(f"\n  {'_'*40}")
    print(f"  SEARCH METHOD ATTRIBUTION")
    print(f"  {'_'*40}")

    method_counts = {"graph": 0, "text": 0, "vector": 0, "synonym": 0}
    for r in testable:
        for method in r.get("found_by", []):
            method_counts[method] = method_counts.get(method, 0) + 1

    for method in ["graph", "text", "vector", "synonym"]:
        cnt = method_counts[method]
        print(f"  {method:>8}: contributed to {cnt:>3}/{total} matched conditions")

    # -- Per-Chapter Breakdown -------------------------------------------------
    print(f"\n  {'_'*40}")
    print(f"  PER-CHAPTER BREAKDOWN")
    print(f"  {'_'*40}")

    chapters = defaultdict(list)
    for r in testable:
        chapters[r["chapter_name"]].append(r)

    print(f"  {'Chapter':<40}  {'N':>3}  {'T1':>3}  {'T3':>3}  {'T5':>3}  {'Miss':>4}")
    print(f"  {'_'*40}  {'_'*3}  {'_'*3}  {'_'*3}  {'_'*3}  {'_'*4}")

    for ch_name in sorted(chapters.keys()):
        ch_results = chapters[ch_name]
        ch_total = len(ch_results)
        ch_top1 = sum(1 for r in ch_results if r["status"] == "TOP_1")
        ch_top3 = sum(1 for r in ch_results if r["status"] in ("TOP_1", "TOP_3"))
        ch_top5 = sum(1 for r in ch_results if r["status"] in ("TOP_1", "TOP_3", "TOP_5"))
        ch_miss = sum(1 for r in ch_results if r["status"] == "MISS")
        label = ch_name[:40]
        print(f"  {label:<40}  {ch_total:>3}  {ch_top1:>3}  {ch_top3:>3}  {ch_top5:>3}  {ch_miss:>4}")

    # -- Missed Conditions -----------------------------------------------------
    if misses:
        print(f"\n  {'_'*40}")
        print(f"  MISSED CONDITIONS ({len(misses)})")
        print(f"  {'_'*40}")
        for r in misses:
            print(f"\n  MISS: {r['id']} - {r['condition_name']} (expected STG {r['expected_code']})")
            print(f"    Chapter:     {r['chapter_name']}")
            print(f"    Symptoms:    {', '.join(r['symptoms_searched'])}")
            print(f"    Graph edges: {r.get('edge_count', '?')}")
            if r.get("top_5_actual"):
                print(f"    Top 5 actual results:")
                for j, actual in enumerate(r["top_5_actual"], 1):
                    print(f"      {j}. {actual['name']} ({actual['stg_code']}) "
                          f"score={actual['score']} via={','.join(actual.get('found_by', []))}")

    # -- Not In Database -------------------------------------------------------
    if not_in_db:
        print(f"\n  {'_'*40}")
        print(f"  NOT IN DATABASE ({len(not_in_db)})")
        print(f"  {'_'*40}")
        for r in not_in_db:
            print(f"  - {r['expected_code']}: {r['condition_name']}")

    # -- Detailed Per-Case (verbose) -------------------------------------------
    if verbose:
        print(f"\n  {'_'*40}")
        print(f"  DETAILED RESULTS")
        print(f"  {'_'*40}")
        for r in testable:
            icon = {
                "TOP_1": "[1]", "TOP_3": "[3]", "TOP_5": "[5]",
                "TOP_10": "[10]", "MISS": "[X]",
            }.get(r["status"], f"[{r['status']}]")
            print(f"\n  {icon} {r['id']}: {r['condition_name']} (STG {r['stg_code']})")
            print(f"      Expected:   {r['expected_code']}")
            print(f"      Symptoms:   {', '.join(r['symptoms_searched'])}")
            print(f"      Top-5 rank: {r['rank_in_top5'] or '-'}")
            print(f"      Abs. rank:  {r['absolute_rank'] or '-'}")
            print(f"      Found by:   {', '.join(r['found_by']) or 'NONE'}")
            print(f"      Edges:      {r.get('edge_count', '?')}")
            if r.get("top_5_actual"):
                print(f"      Top 5:")
                for j, actual in enumerate(r["top_5_actual"], 1):
                    print(f"        {j}. {actual['name']} ({actual['stg_code']}) "
                          f"score={actual['score']}")

    # -- Summary ---------------------------------------------------------------
    print(f"\n{'='*72}")
    if total > 0:
        pct = 100 * top5 / total
        grade = ("EXCELLENT" if pct >= 85 else
                 "GOOD" if pct >= 70 else
                 "FAIR" if pct >= 50 else
                 "NEEDS WORK")
        print(f"  GRADE: {grade}  (Top-5 hit rate: {pct:.0f}%)")
    print(f"{'='*72}\n")


def output_json(results: list[dict]):
    """Output full results as JSON to stdout."""
    print(json.dumps(results, indent=2, default=str))


# == Main ======================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="CareMate Deep Test Suite -- comprehensive triage search evaluation"
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed per-case results")
    parser.add_argument("--chapter", "-c", type=str, default=None,
                        help="Test only cases whose chapter_name contains this string "
                             "(e.g. 'Cardio', 'Mental', 'Respiratory')")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output results as JSON instead of formatted report")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List all test cases without running them")
    args = parser.parse_args()

    # -- List mode -------------------------------------------------------------
    if args.list:
        print(f"\nCareMate Deep Test Suite -- {len(TEST_CASES)} test cases\n")
        current_ch = None
        for tc in sorted(TEST_CASES, key=lambda x: (x["chapter_name"], x["id"])):
            if tc["chapter_name"] != current_ch:
                current_ch = tc["chapter_name"]
                print(f"\n  {current_ch}")
                print(f"  {'_' * 55}")
            symptom_preview = ", ".join(tc["symptoms"][:2])
            if len(tc["symptoms"]) > 2:
                symptom_preview += ", ..."
            print(f"    {tc['id']:>10}  STG {tc['stg_code']:>8}  "
                  f"{tc['condition_name'][:35]:<35}  {symptom_preview}")
        chapters_set = set(tc["chapter_name"] for tc in TEST_CASES)
        print(f"\n  Total: {len(TEST_CASES)} test cases across "
              f"{len(chapters_set)} chapter groups\n")
        return

    # -- Connect to database ---------------------------------------------------
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set in .env")
        sys.exit(1)

    print("\n  CareMate Deep Test Suite")
    print("  " + "=" * 40)

    try:
        conn = await asyncpg.connect(database_url)
    except Exception as e:
        print(f"  ERROR: Cannot connect to database: {e}")
        sys.exit(1)

    # Quick DB summary
    total_conditions = await conn.fetchval("SELECT COUNT(*) FROM conditions")
    total_edges = await conn.fetchval("SELECT COUNT(*) FROM clinical_relationships")
    total_chunks = await conn.fetchval("SELECT COUNT(*) FROM knowledge_chunks")
    has_embeddings = await conn.fetchval(
        "SELECT EXISTS(SELECT 1 FROM knowledge_chunks WHERE embedding IS NOT NULL LIMIT 1)"
    )

    print(f"  Database: {total_conditions} conditions, {total_edges} edges, "
          f"{total_chunks} chunks")
    print(f"  Vector search: {'ENABLED' if has_embeddings else 'DISABLED (no embeddings)'}")

    # Apply chapter filter
    cases_to_run = TEST_CASES
    if args.chapter:
        filter_lower = args.chapter.lower()
        cases_to_run = [tc for tc in TEST_CASES if filter_lower in tc["chapter_name"].lower()]
        print(f"  Filter: chapter_name contains '{args.chapter}' ({len(cases_to_run)} cases)")

    if not cases_to_run:
        print(f"\n  No test cases match filter '{args.chapter}'.")
        print(f"  Available chapters:")
        for ch in sorted(set(tc["chapter_name"] for tc in TEST_CASES)):
            print(f"    - {ch}")
        await conn.close()
        return

    print(f"\n  Running {len(cases_to_run)} test cases...")
    start = time.time()

    results = await run_all_tests(conn, verbose=args.verbose, chapter_filter=args.chapter)

    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")

    await conn.close()

    # Output
    if args.json:
        output_json(results)
    else:
        print_report(results, verbose=args.verbose)

    # Save results to deep_test_results.json
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "deep_test_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Full results saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    asyncio.run(main())
