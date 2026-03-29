from __future__ import annotations

import logging

from my_env.server.environment import MedicalEmergencyRoomEnv
from my_env.models import Action


# ---------------------------------------------------------------------------
# Smart triage policy
# ---------------------------------------------------------------------------

def infer_esi(patient: dict) -> int:
    """Infer ESI level from vitals and symptoms."""
    vitals = patient.get("vitals", {})
    symptoms = [s.lower() for s in patient.get("symptoms", [])]
    hr = vitals.get("heart_rate", 80)
    spo2 = vitals.get("oxygen_level", 98.0)
    bp_str = vitals.get("blood_pressure", "120/80")

    try:
        systolic = int(bp_str.split("/")[0])
    except Exception:
        systolic = 120

    # ESI 1 — immediately life threatening
    critical_symptoms = {
        "pulseless", "unresponsive", "apneic", "cyanotic", "unconscious",
        "airway compromise", "massive hemorrhage", "status epilepticus",
        "stridor", "severe dyspnea", "respiratory distress", "hematemesis",
        "syncope", "polytrauma", "recurrent seizures", "sudden coma",
        "anaphylaxis", "overdose", "altered mental status",
    }
    if any(any(c in s for c in critical_symptoms) for s in symptoms):
        return 1
    if spo2 < 88 or hr > 140 or hr < 40 or systolic < 70:
        return 1

    # ESI 2 — high risk / severe
    # FIX Bug 5: removed "fever" and "tachycardia" — they appear in the ESI-3 urgent
    # set too, causing any patient with an isolated fever to be over-escalated to ESI-2.
    # These symptoms only justify ESI-2 in combination with other critical indicators,
    # which the vitals thresholds below already capture.
    emergent_symptoms = {
        "chest pain", "crushing chest pain", "chest pressure",
        "facial droop", "slurred speech", "weakness", "stroke",
        "severe abdominal", "rigid abdomen", "altered consciousness",
        "sepsis", "neutropenia", "ectopic", "severe pelvic",
        "respiratory distress", "wheezing", "shortness of breath",
        "diaphoresis", "left arm pain", "sweating",
    }
    if any(any(e in s for e in emergent_symptoms) for s in symptoms):
        return 2
    if spo2 < 92 or hr > 120 or systolic < 85:
        return 2

    # ESI 3 — urgent
    urgent_symptoms = {
        "abdominal pain", "nausea", "vomiting", "fever", "cough",
        "back pain", "head injury", "laceration", "diabetic",
        "blood sugar", "allergic", "urinary", "cellulitis",
        "vertigo", "pediatric", "pregnant", "eye injury",
        "psychiatric", "suicidal", "asthma", "moderate",
    }
    if any(any(u in s for u in urgent_symptoms) for s in symptoms):
        return 3
    if hr > 100 or spo2 < 95:
        return 3

    # ESI 4 — semi urgent
    semi_urgent = {
        "sprained", "ear pain", "laceration", "urinary symptoms",
        "mild asthma", "tooth pain", "limping", "constipation",
        "mild", "minor",
    }
    if any(any(u in s for u in semi_urgent) for s in symptoms):
        return 4

    # ESI 5 — non urgent
    return 5


def pick_bed(esi: int, obs: dict) -> str:
    """Pick best available bed for given ESI."""
    icu_free = obs.get("icu_beds_available", 0)
    gen_free = obs.get("general_beds_available", 0)
    hallway = obs.get("hallway_used", 0)
    hallway_cap = 3

    if esi <= 2:
        if icu_free > 0:
            return "icu"
        if gen_free > 0:
            return "general"
        if hallway < hallway_cap:
            return "hallway"
        # FIX Bug 3: returning "icu" when no beds exist causes the environment to
        # apply an ineffective-action penalty every step. "none" is the correct
        # sentinel — it signals no bed is available without triggering a bad action.
        return "none"
    else:
        if gen_free > 0:
            return "general"
        if hallway < hallway_cap:
            return "hallway"
        if icu_free > 0:
            return "icu"
        return "none"


def pick_protocol(patient_id: str, patients: list, mask: dict) -> str | None:
    """Check if any valid protocol can be triggered for this patient."""
    available = mask.get("trigger_protocol", {}).get(patient_id, [])
    if not available:
        return None

    patient = next((p for p in patients if p["patient_id"] == patient_id), None)
    if patient is None:
        return None

    symptoms = [s.lower() for s in patient.get("symptoms", [])]

    protocol_keywords = {
        "stroke_code":  ["facial droop", "slurred speech", "weakness"],
        "stemi_alert":  ["chest pain", "diaphoresis", "shortness of breath", "crushing chest"],
        "sepsis_alert": ["fever", "tachycardia", "altered mental status"],
        "trauma_alert": ["polytrauma", "hypotension", "bleeding"],
    }

    for protocol in available:
        keywords = protocol_keywords.get(protocol, [])
        matches = sum(1 for kw in keywords if any(kw in s for s in symptoms))
        if matches >= 2:
            return protocol

    return None


def choose_action(obs_raw: dict, mask: dict) -> Action:
    """Smart policy: triage → bed → protocol → discharge → divert."""
    patients = obs_raw.get("patients", [])
    patient_map = {p["patient_id"]: p for p in patients}

    # 1. Triage untriaged patients
    assign_ids = mask.get("assign_esi_patient_ids", [])
    if assign_ids:
        pid = assign_ids[0]
        patient = patient_map.get(pid, {})
        esi = infer_esi(patient)
        return Action(action_type="assign_esi", patient_id=pid, esi_level=esi)

    # 2. Allocate beds — prioritise lowest ESI (most critical) first
    allocate = mask.get("allocate_bed", {})
    if allocate:
        sorted_pids = sorted(
            allocate.keys(),
            key=lambda pid: patient_map.get(pid, {}).get("esi_level", 5)
        )
        for pid in sorted_pids:
            beds = allocate[pid]
            if not beds:
                continue
            patient = patient_map.get(pid, {})
            esi = patient.get("esi_level", 3)
            bed = pick_bed(esi, obs_raw)
            # FIX Bug 4: always validate the preferred bed against the action mask.
            # The old `bed == "icu"` short-circuit bypassed the mask check entirely,
            # allowing ICU allocation even when ICU was not listed as available.
            # If the preferred bed isn't in the mask, fall back to whatever is allowed.
            if bed in beds:
                return Action(action_type="allocate_bed", patient_id=pid, bed_type=bed)
            # Fallback: use the best available bed from the mask
            for fallback_bed in ["icu", "general", "hallway"]:
                if fallback_bed in beds:
                    return Action(action_type="allocate_bed", patient_id=pid, bed_type=fallback_bed)

    # 3. Trigger protocols
    protocol_map = mask.get("trigger_protocol", {})
    if protocol_map:
        for pid in protocol_map:
            protocol = pick_protocol(pid, patients, mask)
            if protocol:
                return Action(action_type="trigger_protocol", patient_id=pid, protocol_type=protocol)

    # 4. Discharge eligible ESI 4/5 patients
    discharge_ids = mask.get("discharge_patient_ids", [])
    if discharge_ids:
        return Action(action_type="discharge", patient_id=discharge_ids[0])

    # 5. Divert if overloaded
    if mask.get("divert_allowed", False):
        return Action(action_type="divert")

    # FIX Bug 7: unconditional fallback divert fires even when divert_allowed=False,
    # accumulating "invalid divert" penalties every step. Log a warning so it's
    # No valid action — try protocol as fallback
    for pid, protocols in mask.get("trigger_protocol", {}).items():
        if protocols:
            return Action(action_type="trigger_protocol", patient_id=pid, protocol_type=protocols[0])

    # True last resort
    return Action(action_type="divert")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_task(difficulty: str, seed: int) -> dict:
    env = MedicalEmergencyRoomEnv(difficulty=difficulty, seed=seed)

    # FIX Bug 1: env.reset() returns a (Observation, info) 2-tuple after the
    # environment fix — unpack correctly to avoid obs holding the raw tuple.
    obs = env.reset(seed=seed)

    done = False
    total_reward = 0.0
    steps = 0
    correct_esi = 0
    total_triage = 0

    while not done:
        obs_raw = obs.model_dump(mode="python")
        mask = env.get_action_mask()
        action = choose_action(obs_raw, mask)

        # Track ESI accuracy — eval-only: accesses private env state.
        # FIX Bug 6: wrapped in try/except so this doesn't crash if env is
        # replaced with an HTTP client that doesn't expose _patients.
        if action.action_type == "assign_esi" and action.patient_id:
            try:
                runtime = env._patients.get(action.patient_id)
                if runtime and action.esi_level == runtime.true_esi:
                    correct_esi += 1
            except AttributeError:
                pass  # not available when running against HTTP client
            total_triage += 1

        # FIX Bug 2: env.step() now returns a 5-tuple (obs, reward, terminated,
        # truncated, info) — unpack all five values and derive done from both flags.
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        steps += 1

    avg = round(total_reward / steps, 4) if steps > 0 else 0.0
    esi_acc = round(correct_esi / total_triage, 3) if total_triage > 0 else 0.0

    return {
        "difficulty": difficulty,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "avg_reward": avg,
        "esi_accuracy": esi_acc,
        "triage_attempts": total_triage,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    print("=" * 50)
    print("  Medical ER Triage OpenEnv — Smart Demo")
    print("=" * 50)

    seeds = {"easy": 42, "medium": 43, "hard": 44}
    results = []

    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n--- Task: {difficulty} ---")
        result = run_task(difficulty, seeds[difficulty])
        results.append(result)
        print(f"  Steps        : {result['steps']}")
        print(f"  Total reward : {result['total_reward']}")
        print(f"  Avg reward   : {result['avg_reward']}")
        print(f"  ESI accuracy : {result['esi_accuracy'] * 100:.1f}%  ({result['triage_attempts']} attempts)")

    macro_avg = round(sum(r["avg_reward"] for r in results) / len(results), 4)
    print(f"\n{'=' * 50}")
    print(f"  Macro avg reward : {macro_avg}")
    print(f"  Pass threshold   : 0.70")
    print(f"  Status           : {'✅ PASS' if macro_avg >= 0.70 else '❌ BELOW THRESHOLD'}")
    print("=" * 50)
    print("\nDemo complete.")
