"""
South African Triage Scale (SATS) — Triage Early Warning Score (TEWS)
=====================================================================

Implements the nationally standardised SATS colour-coded triage system
used across all SA public and private emergency departments.

Source: SATS Training Manual 2012, South African Triage Group (SATG).

Colour codes:
  RED    — Emergency, immediate resuscitation
  ORANGE — Very urgent, < 10 minutes
  YELLOW — Urgent, < 60 minutes
  GREEN  — Routine, < 4 hours

The system has two components:
1. Clinical discriminators — specific signs that assign a colour directly
2. TEWS vital sign scoring — maps vitals to a score, score to a colour

Discriminators override TEWS (e.g., active seizure → RED regardless of vitals).
"""

from typing import Optional

# ── TEWS Score → Colour Mapping ──────────────────────────────────────────────

TEWS_COLOUR_MAP = [
    # (min_score, max_score, colour, priority_name, target_minutes)
    (0, 2, "green", "Routine", 240),
    (3, 4, "yellow", "Urgent", 60),
    (5, 6, "orange", "Very Urgent", 10),
    (7, 99, "red", "Emergency", 0),
]

# ── SATS → Legacy Acuity Mapping ─────────────────────────────────────────────
# Maps SATS colours to the existing 3-tier acuity system for backward
# compatibility with the frontend (until it supports SATS colours natively).
SATS_TO_ACUITY = {
    "green": "routine",
    "yellow": "routine",   # Yellow is "urgent" in SATS but maps to routine in our 3-tier
    "orange": "priority",
    "red": "urgent",
}


# ── Adult TEWS Thresholds (age > 12 or height > 150cm) ───────────────────────

def _score_rr_adult(rr: float) -> int:
    if rr < 9:
        return 3
    if rr <= 14:
        return 0
    if rr <= 20:
        return 1
    if rr <= 29:
        return 2
    return 3  # >= 30


def _score_hr_adult(hr: float) -> int:
    if hr < 41:
        return 3
    if hr <= 50:
        return 1
    if hr <= 100:
        return 0
    if hr <= 110:
        return 1
    if hr <= 129:
        return 2
    return 3  # >= 130


def _score_sbp_adult(sbp: float) -> int:
    if sbp < 71:
        return 3
    if sbp <= 100:
        return 1
    if sbp <= 110:
        return 0
    if sbp <= 129:
        return 1
    return 2  # > 129


def _score_temp(temp: float) -> int:
    """Temperature scoring — same for all age groups."""
    if temp < 35.0:
        return 2
    if temp <= 38.4:
        return 0
    return 2  # >= 38.4 (feels hot)


def _score_avpu(avpu: str) -> int:
    """AVPU scoring for adults. Values: alert, voice, pain, unresponsive, confused."""
    avpu = avpu.lower().strip()
    if avpu in ("a", "alert"):
        return 0
    if avpu in ("v", "voice"):
        return 1
    if avpu in ("p", "pain"):
        return 2
    if avpu in ("u", "unresponsive"):
        return 3
    if avpu in ("c", "confused"):
        return 1  # Adult confused = 1
    return 0


def _score_mobility(mobility: str) -> int:
    """Mobility scoring. Values: walking, with_help, stretcher, immobile."""
    mobility = mobility.lower().strip()
    if mobility in ("walking", "normal", "normal for age"):
        return 0
    if mobility in ("with_help", "with help"):
        return 1
    if mobility in ("stretcher", "immobile", "unable"):
        return 2
    return 0


# ── TEWS Calculator ──────────────────────────────────────────────────────────

def compute_tews(
    vitals: dict,
    patient_age: Optional[int] = None,
    mobility: str = "walking",
    avpu: str = "alert",
    trauma: bool = False,
) -> dict:
    """Compute TEWS score from vital signs.

    Args:
        vitals: Dict with keys matching the API contract:
            - heartRate, temperature, respiratoryRate, systolic/bp_systolic,
              oxygenSat, diastolic/bp_diastolic
        patient_age: Age in years (determines which TEWS table to use)
        mobility: "walking", "with_help", "stretcher", "immobile"
        avpu: "alert", "voice", "pain", "unresponsive", "confused"
        trauma: Whether patient has had any injury in last 48 hours

    Returns:
        Dict with: tews_score, colour, priority, target_minutes,
                   component_scores (breakdown), reasons (human-readable)
    """
    # Normalise vital sign keys (handle both API formats)
    hr = vitals.get("heartRate") or vitals.get("heart_rate")
    temp = vitals.get("temperature") or vitals.get("temp_celsius")
    rr = vitals.get("respiratoryRate") or vitals.get("respiratory_rate")
    sbp = vitals.get("systolic") or vitals.get("bp_systolic")
    spo2 = vitals.get("oxygenSat") or vitals.get("spo2")

    # Determine age group for TEWS table
    # Default to adult if age not provided
    age = patient_age or 30
    is_adult = age > 12

    components = {}
    reasons = []
    total = 0

    # Score each parameter
    if rr is not None:
        if is_adult:
            s = _score_rr_adult(rr)
        else:
            s = _score_rr_child(rr, age)
        components["respiratory_rate"] = s
        total += s
        if s >= 2:
            reasons.append(f"Respiratory rate {rr}/min (TEWS +{s})")

    if hr is not None:
        if is_adult:
            s = _score_hr_adult(hr)
        else:
            s = _score_hr_child(hr, age)
        components["heart_rate"] = s
        total += s
        if s >= 2:
            reasons.append(f"Heart rate {hr} bpm (TEWS +{s})")

    if sbp is not None and is_adult:
        s = _score_sbp_adult(sbp)
        components["systolic_bp"] = s
        total += s
        if s >= 2:
            reasons.append(f"Systolic BP {sbp} mmHg (TEWS +{s})")

    if temp is not None:
        s = _score_temp(temp)
        components["temperature"] = s
        total += s
        if s >= 2:
            if temp >= 38.4:
                reasons.append(f"Fever {temp}°C (TEWS +{s})")
            else:
                reasons.append(f"Hypothermia {temp}°C (TEWS +{s})")

    if spo2 is not None and spo2 < 92:
        reasons.append(f"SpO2 {spo2}% < 92% — move to resuscitation")

    # AVPU
    s = _score_avpu(avpu)
    components["avpu"] = s
    total += s
    if s >= 2:
        reasons.append(f"Reduced consciousness: {avpu} (TEWS +{s})")

    # Mobility
    s = _score_mobility(mobility)
    components["mobility"] = s
    total += s
    if s >= 1:
        reasons.append(f"Mobility: {mobility} (TEWS +{s})")

    # Trauma
    if trauma:
        components["trauma"] = 1
        total += 1
        reasons.append("Trauma within 48 hours (TEWS +1)")
    else:
        components["trauma"] = 0

    # Map total to colour
    colour = "green"
    priority = "Routine"
    target = 240
    for min_s, max_s, col, pri, tgt in TEWS_COLOUR_MAP:
        if min_s <= total <= max_s:
            colour = col
            priority = pri
            target = tgt
            break

    return {
        "tews_score": total,
        "colour": colour,
        "priority": priority,
        "target_minutes": target,
        "acuity": SATS_TO_ACUITY[colour],
        "component_scores": components,
        "reasons": reasons,
    }


# ── Child TEWS Scoring ───────────────────────────────────────────────────────

def _score_rr_child(rr: float, age: int) -> int:
    """RR scoring for children. Uses age to determine thresholds."""
    if age >= 3:
        # Older child (3-12 years)
        if rr < 15:
            return 0
        if rr <= 16:
            return 1
        if rr <= 21:
            return 2
        return 3  # >= 22
    else:
        # Younger child (< 3 years)
        if rr < 20:
            return 0
        if rr <= 25:
            return 1
        if rr <= 39:
            return 2
        return 3  # >= 40


def _score_hr_child(hr: float, age: int) -> int:
    """HR scoring for children. Uses age to determine thresholds."""
    if age >= 3:
        # Older child (3-12 years)
        if hr < 60:
            return 0
        if hr <= 79:
            return 1
        if hr <= 99:
            return 2
        return 3  # >= 100
    else:
        # Younger child (< 3 years)
        if hr < 70:
            return 0
        if hr <= 79:
            return 1
        if hr <= 130:
            return 2
        if hr <= 159:
            return 3
        return 3  # >= 160


# ── Clinical Discriminators ──────────────────────────────────────────────────
# These override TEWS — if any discriminator matches, the colour is set directly.

ADULT_EMERGENCY_SIGNS = [
    "obstructed airway",
    "not breathing",
    "seizure - current",
    "seizures - current",
    "active seizure",
    "burn - facial",
    "inhalation burn",
    "hypoglycaemia",
    "glucose less than 3",
    "glucose < 3",
    "cardiac arrest",
]

ADULT_VERY_URGENT_SIGNS = [
    "shortness of breath - acute",
    "acute shortness of breath",
    "reduced consciousness",
    "haemoptysis",
    "coughing blood",
    "chest pain",
    "stabbed neck",
    "uncontrolled haemorrhage",
    "arterial bleed",
    "post ictal",
    "focal neurology",
    "acute stroke",
    "aggression",
    "threatened limb",
    "compound fracture",
    "open fracture",
    "burn over 20%",
    "electrical burn",
    "circumferential burn",
    "chemical burn",
    "poisoning",
    "overdose",
    "vomiting fresh blood",
    "haematemesis",
    "severe pain",
    "pregnancy and abdominal trauma",
    "pregnancy and abdominal pain",
]

ADULT_URGENT_SIGNS = [
    "high energy transfer",
    "motor vehicle accident",
    "pedestrian vehicle accident",
    "fall from height",
    "high velocity gunshot",
    "eye injury",
    "dislocation of larger joint",
    "diabetic - glucose over 11",
    "ketonuria",
    "vomiting persistently",
    "moderate pain",
    "controlled haemorrhage",
    "closed fracture",
    "burn - other",
    "abdominal pain",
]

PAEDIATRIC_EMERGENCY_SIGNS = [
    "not breathing",
    "apnoea",
    "obstructed breathing",
    "central cyanosis",
    "spo2 < 92",
    "severe respiratory distress",
    "cold hands with weak pulse",
    "capillary refill >= 3",
    "uncontrolled bleeding",
    "responds only to pain",
    "unresponsive",
    "confusion",
    "convulsing",
    "immediately post-ictal",
    "severe dehydration",
    "facial burn",
    "inhalation burn",
    "hypoglycaemia",
    "glucose < 3",
    "purpuric rash",
    "non-blanching rash",
]


def check_discriminators(
    complaint: str,
    symptoms: list[str],
    patient_age: Optional[int] = None,
    vitals: Optional[dict] = None,
) -> dict:
    """Check for SATS clinical discriminators that override TEWS colour.

    Returns:
        Dict with: discriminator_colour (red/orange/yellow or None),
                   matched_discriminators (list of matched signs),
                   is_emergency (bool)
    """
    age = patient_age or 30
    is_child = age < 13

    # Combine complaint and symptoms for matching
    search_text = (complaint + " " + " ".join(symptoms)).lower()

    matched = []
    colour = None

    # Check for vitals-based discriminators
    if vitals:
        glucose = vitals.get("glucose") or vitals.get("blood_glucose")
        spo2 = vitals.get("oxygenSat") or vitals.get("spo2")

        if glucose is not None and glucose < 3:
            matched.append("Hypoglycaemia: glucose < 3 mmol/L")
            colour = "red"

        if spo2 is not None and spo2 < 92:
            matched.append(f"SpO2 {spo2}% < 92%")
            colour = "red"

    # Check emergency signs
    emergency_signs = PAEDIATRIC_EMERGENCY_SIGNS if is_child else ADULT_EMERGENCY_SIGNS
    for sign in emergency_signs:
        if sign in search_text:
            matched.append(f"Emergency: {sign}")
            colour = "red"

    if colour == "red":
        return {
            "discriminator_colour": "red",
            "matched_discriminators": matched,
            "is_emergency": True,
        }

    # Check very urgent signs
    very_urgent_signs = ADULT_VERY_URGENT_SIGNS
    for sign in very_urgent_signs:
        if sign in search_text:
            matched.append(f"Very urgent: {sign}")
            if colour != "orange":
                colour = "orange"

    if colour == "orange":
        return {
            "discriminator_colour": "orange",
            "matched_discriminators": matched,
            "is_emergency": False,
        }

    # Check urgent signs
    urgent_signs = ADULT_URGENT_SIGNS
    for sign in urgent_signs:
        if sign in search_text:
            matched.append(f"Urgent: {sign}")
            if colour != "yellow":
                colour = "yellow"

    return {
        "discriminator_colour": colour,
        "matched_discriminators": matched,
        "is_emergency": False,
    }


def compute_sats_acuity(
    vitals: dict,
    complaint: str = "",
    symptoms: list[str] | None = None,
    patient_age: int | None = None,
    mobility: str = "walking",
    avpu: str = "alert",
    trauma: bool = False,
) -> dict:
    """Full SATS triage: discriminators first, then TEWS.

    This is the main entry point — replaces _compute_vitals_acuity().

    Returns dict with:
        acuity: str ("routine"/"priority"/"urgent") — backward compatible
        sats_colour: str ("green"/"yellow"/"orange"/"red")
        sats_priority: str ("Routine"/"Urgent"/"Very Urgent"/"Emergency")
        tews_score: int
        reasons: list[str]
        discriminators_matched: list[str]
        target_minutes: int
    """
    symptoms = symptoms or []

    # Step 1: Check discriminators (override TEWS)
    disc = check_discriminators(complaint, symptoms, patient_age, vitals)

    # Step 2: Calculate TEWS from vitals
    tews = compute_tews(vitals, patient_age, mobility, avpu, trauma)

    # Step 3: Final colour = higher severity of (discriminator, TEWS)
    colour_rank = {"green": 0, "yellow": 1, "orange": 2, "red": 3}
    disc_colour = disc.get("discriminator_colour")
    tews_colour = tews["colour"]

    if disc_colour and colour_rank.get(disc_colour, 0) > colour_rank.get(tews_colour, 0):
        final_colour = disc_colour
    else:
        final_colour = tews_colour

    # Map to priority and target
    priority_map = {"green": "Routine", "yellow": "Urgent", "orange": "Very Urgent", "red": "Emergency"}
    target_map = {"green": 240, "yellow": 60, "orange": 10, "red": 0}

    # Combine reasons
    all_reasons = list(disc.get("matched_discriminators", []))
    all_reasons.extend(tews.get("reasons", []))

    return {
        "acuity": SATS_TO_ACUITY[final_colour],
        "sats_colour": final_colour,
        "sats_priority": priority_map[final_colour],
        "tews_score": tews["tews_score"],
        "reasons": all_reasons,
        "discriminators_matched": disc.get("matched_discriminators", []),
        "target_minutes": target_map[final_colour],
        "component_scores": tews.get("component_scores", {}),
    }
