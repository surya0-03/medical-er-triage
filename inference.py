from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Literal

import requests
from openai import OpenAI

# ── Required env vars (per hackathon spec) ────────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN: str = os.getenv("HF_TOKEN", os.getenv("GROQ_API_KEY", ""))

# ── Environment endpoint (your HF Space) ──────────────────────────────────────
ENV_BASE_URL: str = os.getenv(
    "ENV_BASE_URL", "https://maldini03-medical-er-triage-env.hf.space"
).rstrip("/")

INFERENCE_SEED: int = int(os.getenv("OPENAI_INFERENCE_SEED", "12345"))
BENCHMARK: str = "medical-er-triage"
TaskName = Literal["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are a medical ER triage policy. Return exactly one JSON object with keys: action_type, patient_id, esi_level, bed_type, protocol_type. Use null for unused fields. Return only the JSON, no explanation.

SCORING RULES (maximise reward):
- assign_esi: exact ESI match=+0.50, off by 1=+0.25, off by 2=+0.10. NEVER assign ESI1 patient as 3/4/5 (-0.50 penalty). NEVER assign ESI2 as 4/5 (-0.30 penalty). Always use the action_mask assign_esi_patient_ids list.
- allocate_bed: ESI1/2 → "icu" (+0.15). ESI3 → "general" (+0.10). ESI4/5 → "general" (+0.10, never "icu" or you get -0.10). ESI1/2 in hallway = -0.50 penalty. Only allocate beds from the action_mask allocate_bed map.
- trigger_protocol: correct match=+0.20, wrong=-0.30. Protocol rules: stroke_code needs [facial droop/slurred speech/unilateral weakness] + ESI≤2 + icu bed. stemi_alert needs [crushing chest pain/diaphoresis/shortness of breath] + ESI≤2 + icu bed. sepsis_alert needs [fever/tachycardia/altered mental status] + ESI≤2 + general bed. trauma_alert needs [polytrauma/hypotension/active bleeding] + ESI≤1 + icu bed.
- discharge: only for ESI4/5 patients who have waited past safe limit. Use action_mask discharge_patient_ids.
- divert: only when overloaded (divert_allowed=true). patient_id must be null.

PRIORITY: Always handle the most critical patients first (lowest ESI number = most critical). Follow the action_mask exactly — only act on IDs listed there."""

TASK_SEEDS: dict[str, int] = {"easy": 42, "medium": 43, "hard": 44}

# ── Symptom/vitals-based ESI inference (used by deterministic fallback) ───────

def infer_esi_from_patient(patient: dict[str, Any]) -> int:
    """Infer ESI level from vitals and symptoms — mirrors demo.py logic."""
    vitals = patient.get("vitals", {})
    symptoms = [s.lower() for s in patient.get("symptoms", [])]
    hr = vitals.get("heart_rate", 80)
    spo2 = float(vitals.get("oxygen_level", 98.0))
    bp_str = str(vitals.get("blood_pressure", "120/80"))
    try:
        systolic = int(bp_str.split("/")[0])
    except Exception:
        systolic = 120

    critical = {
        "stridor", "cyanosis", "severe dyspnea", "respiratory distress",
        "hematemesis", "syncope", "polytrauma", "recurrent seizures",
        "sudden coma", "apneic", "altered mental status",
    }
    if any(any(c in s for c in critical) for s in symptoms):
        return 1
    if spo2 < 88 or hr > 140 or hr < 40 or systolic < 70:
        return 1

    emergent = {
        "chest pain", "crushing chest pain", "chest pressure",
        "facial droop", "slurred speech", "weakness",
        "shortness of breath", "diaphoresis", "left arm pain",
        "wheezing", "chest tightness", "suicidal", "severe headache",
        "melena", "lightheadedness", "somnolence", "pinpoint pupils",
    }
    if any(any(e in s for e in emergent) for s in symptoms):
        return 2
    if spo2 < 92 or hr > 120 or systolic < 85:
        return 2

    urgent = {
        "abdominal pain", "fever", "cough", "vomiting", "diarrhea",
        "flank pain", "hematuria", "fracture", "deformity",
        "productive cough", "headache", "photophobia",
    }
    if any(any(u in s for u in urgent) for s in symptoms):
        return 3
    if hr > 100 or spo2 < 95:
        return 3

    semi_urgent = {"sprained", "ear pain", "laceration", "dysuria", "mild", "minor", "tooth"}
    if any(any(u in s for u in semi_urgent) for s in symptoms):
        return 4

    return 5


def match_protocol(patient: dict[str, Any], available_protocols: list[str]) -> str | None:
    """Return the first protocol whose symptoms match >= 2 keywords, else None."""
    symptoms = [s.lower() for s in patient.get("symptoms", [])]
    protocol_keywords: dict[str, list[str]] = {
        "stroke_code":  ["facial droop", "slurred speech", "weakness"],
        "stemi_alert":  ["chest pain", "diaphoresis", "shortness of breath", "crushing chest"],
        "sepsis_alert": ["fever", "tachycardia", "altered mental status"],
        "trauma_alert": ["polytrauma", "hypotension", "bleeding"],
    }
    for protocol in available_protocols:
        keywords = protocol_keywords.get(protocol, [])
        matches = sum(1 for kw in keywords if any(kw in s for s in symptoms))
        if matches >= 2:
            return protocol
    return None


# ── Logging (mandatory stdout format) ────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


# ── Environment HTTP client ───────────────────────────────────────────────────

class ERTriageEnvClient:
    """Thin HTTP wrapper around the FastAPI environment."""

    def __init__(self, base_url: str) -> None:
        self._base = base_url
        self._session_id: str = "default"

    def reset(self, difficulty: str, seed: int) -> dict[str, Any]:
        resp = requests.post(
            f"{self._base}/reset",
            json={"difficulty": difficulty, "seed": seed},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self._session_id = data.get("session_id", "default")
        return data

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        resp = requests.post(
            f"{self._base}/step",
            json={"session_id": self._session_id, "action": action},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict[str, Any]:
        resp = requests.get(
            f"{self._base}/state",
            params={"session_id": self._session_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def get_action_mask(self) -> dict[str, Any]:
        data = self.state()
        return data.get("action_mask", {})

    def _get(self, path: str) -> dict[str, Any]:
        resp = requests.get(f"{self._base}{path}", timeout=30)
        resp.raise_for_status()
        return resp.json()


# ── LLM agent ────────────────────────────────────────────────────────────────

def build_user_prompt(task: str, obs: dict[str, Any], action_mask: dict[str, Any]) -> str:
    observation = obs.get("observation", obs)
    waiting_queue: list = observation.get("waiting_queue", [])
    patients: list = observation.get("patients", [])

    queue_index = {pid: idx for idx, pid in enumerate(waiting_queue)}
    waiting_patients = []
    for p in patients:
        pid = str(p["patient_id"])
        if pid in queue_index:
            vitals = p.get("vitals", {})
            waiting_patients.append({
                "patient_id": pid,
                "queue_position": queue_index[pid],
                "esi_level": int(p["esi_level"]),
                "wait_time": int(p["wait_time"]),
                "symptoms": list(p["symptoms"]),
                "vitals": {
                    "heart_rate": vitals.get("heart_rate"),
                    "blood_pressure": vitals.get("blood_pressure"),
                    "oxygen_level": vitals.get("oxygen_level"),
                },
                "deterioration_count": int(p.get("deterioration_count", 0)),
                "state": str(p["state"]),
            })

    waiting_patients.sort(key=lambda x: (x["esi_level"], -x["wait_time"]))

    summary = {
        "task": task,
        "time_step": int(observation.get("time_step", 0)),
        "icu_beds_available": int(observation.get("icu_beds_available", 0)),
        "general_beds_available": int(observation.get("general_beds_available", 0)),
        "hallway_used": int(observation.get("hallway_used", 0)),
        "crowding_score": float(observation.get("crowding_score", 0.0)),
        "staff_load": observation.get("staff_load", {}),
        "waiting_patients_top": waiting_patients[:8],
        "waiting_queue_size": len(waiting_queue),
        "action_mask": action_mask,
    }
    return json.dumps(summary, separators=(",", ":"))


def parse_action(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3:
            content = "\n".join(lines[1:-1]).strip()
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("LLM output must be a JSON object")
    return payload


def deterministic_fallback(action_mask: dict[str, Any], obs: dict[str, Any] | None = None) -> dict[str, Any]:
    patient_esi: dict[str, int] = {}
    patient_map: dict[str, dict[str, Any]] = {}
    if obs is not None:
        observation = obs.get("observation", obs)
        for p in observation.get("patients", []):
            pid = str(p["patient_id"])
            patient_map[pid] = p
            patient_esi[pid] = infer_esi_from_patient(p)

    assign_ids = action_mask.get("assign_esi_patient_ids", [])
    if assign_ids:
        pid = min(assign_ids, key=lambda p: (patient_esi.get(p, 3), p))
        esi_level = patient_esi.get(pid, 3)
        return {"action_type": "assign_esi", "patient_id": pid,
                "esi_level": esi_level, "bed_type": None, "protocol_type": None}

    allocate = action_mask.get("allocate_bed", {})
    if allocate:
        sorted_pids = sorted(allocate.keys(), key=lambda p: (patient_esi.get(p, 3), p))
        for pid in sorted_pids:
            choices = set(allocate[pid])
            esi = patient_esi.get(pid, 3)
            if esi <= 2:
                bed = "icu" if "icu" in choices else "general" if "general" in choices else "hallway" if "hallway" in choices else None
            else:
                bed = "general" if "general" in choices else "hallway" if "hallway" in choices else "icu" if "icu" in choices else None
            if bed is not None:
                return {"action_type": "allocate_bed", "patient_id": pid,
                        "bed_type": bed, "esi_level": None, "protocol_type": None}

    protocol_map = action_mask.get("trigger_protocol", {})
    for pid in sorted(protocol_map.keys()):
        protocols = protocol_map[pid]
        if not protocols:
            continue
        patient = patient_map.get(pid, {})
        matched = match_protocol(patient, protocols)
        if matched:
            return {"action_type": "trigger_protocol", "patient_id": pid,
                    "protocol_type": matched, "esi_level": None, "bed_type": None}

    discharge_ids = action_mask.get("discharge_patient_ids", [])
    if discharge_ids:
        return {"action_type": "discharge", "patient_id": discharge_ids[0],
                "esi_level": None, "bed_type": None, "protocol_type": None}

    if action_mask.get("divert_allowed", False):
        return {"action_type": "divert", "patient_id": None,
                "esi_level": None, "bed_type": None, "protocol_type": None}

    return {"action_type": "divert", "patient_id": None,
            "esi_level": None, "bed_type": None, "protocol_type": None}


def choose_action(
    client: OpenAI | None,
    task: str,
    obs: dict[str, Any],
    action_mask: dict[str, Any],
) -> dict[str, Any]:
    if client is None:
        return deterministic_fallback(action_mask, obs)

    prompt = build_user_prompt(task, obs, action_mask)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            seed=INFERENCE_SEED,
            max_tokens=256,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = completion.choices[0].message.content or ""
        return parse_action(content)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True, file=sys.stderr)
        return deterministic_fallback(action_mask, obs)


# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(
    task: TaskName,
    client: OpenAI | None,
    env_client: ERTriageEnvClient,
) -> None:
    seed = TASK_SEEDS[task]
    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    rewards: list[float] = []
    steps_taken = 0
    success = False

    try:
        obs = env_client.reset(difficulty=task, seed=seed)
        done = False
        step = 0

        while not done and step < 500:
            step += 1
            action_mask = env_client.get_action_mask()
            action = choose_action(client, task, obs, action_mask)

            result = env_client.step(action)
            reward = float(result.get("reward", {}).get("value", 0.0))
            done = bool(result.get("terminated", False)) or bool(result.get("truncated", False))
            error = result.get("info", {}).get("error") if isinstance(result.get("info"), dict) else None

            rewards.append(reward)
            steps_taken = step
            action_str = json.dumps(action, separators=(",", ":"))
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            obs = result

        success = sum(rewards) > 0.0

    except Exception as exc:
        print(f"[DEBUG] Task {task} error: {exc}", flush=True, file=sys.stderr)
        success = False

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
    api_base = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")

    client: OpenAI | None = None
    if api_key:
        try:
            client = OpenAI(base_url=api_base, api_key=api_key)
            print(f"Using LLM: {MODEL_NAME} @ {api_base}", flush=True, file=sys.stderr)
        except Exception as exc:
            print(f"[DEBUG] Client init failed: {exc}", flush=True, file=sys.stderr)
    else:
        print("[DEBUG] No API key found. Using deterministic fallback.", file=sys.stderr)

    env_client = ERTriageEnvClient(base_url=ENV_BASE_URL)

    import time
    for attempt in range(9):
        try:
            health = env_client._get("/health")
            if health.get("status") == "healthy":
                print(f"[DEBUG] Space healthy after {attempt} retries", flush=True, file=sys.stderr)
                break
        except Exception:
            pass
        print(f"[DEBUG] Waiting for Space to wake (attempt {attempt + 1}/9)...", flush=True, file=sys.stderr)
        time.sleep(10)

    for task in ("easy", "medium", "hard"):
        run_task(task=task, client=client, env_client=env_client)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
