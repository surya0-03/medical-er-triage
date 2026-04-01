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

SYSTEM_PROMPT = (
    "You are a medical ER triage policy. "
    "Return exactly one JSON object with keys: action_type, patient_id, esi_level, bed_type, protocol_type. "
    "Valid action_type values: assign_esi, allocate_bed, discharge, trigger_protocol, divert. "
    "Use null for fields not required by the selected action. "
    "Return only the JSON object, no explanation."
)

TASK_SEEDS: dict[str, int] = {"easy": 42, "medium": 43, "hard": 44}


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
            waiting_patients.append({
                "patient_id": pid,
                "queue_position": queue_index[pid],
                "esi_level": int(p["esi_level"]),
                "wait_time": int(p["wait_time"]),
                "symptoms": list(p["symptoms"]),
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


def deterministic_fallback(action_mask: dict[str, Any]) -> dict[str, Any]:
    assign_ids = sorted(action_mask.get("assign_esi_patient_ids", []))
    if assign_ids:
        return {"action_type": "assign_esi", "patient_id": assign_ids[0],
                "esi_level": 3, "bed_type": None, "protocol_type": None}

    allocate = action_mask.get("allocate_bed", {})
    for pid in sorted(allocate.keys()):
        choices = set(allocate[pid])
        bed = "icu" if "icu" in choices else "general" if "general" in choices else "hallway"
        if bed in choices:
            return {"action_type": "allocate_bed", "patient_id": pid,
                    "bed_type": bed, "esi_level": None, "protocol_type": None}

    protocol_map = action_mask.get("trigger_protocol", {})
    for pid in sorted(protocol_map.keys()):
        protocols = sorted(protocol_map[pid])
        if protocols:
            return {"action_type": "trigger_protocol", "patient_id": pid,
                    "protocol_type": protocols[0], "esi_level": None, "bed_type": None}

    discharge_ids = sorted(action_mask.get("discharge_patient_ids", []))
    if discharge_ids:
        return {"action_type": "discharge", "patient_id": discharge_ids[0],
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
        return deterministic_fallback(action_mask)

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
        return deterministic_fallback(action_mask)


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
    # Build OpenAI-compatible client (Groq by default, OpenAI if key provided)
    client: OpenAI | None = None
    if HF_TOKEN:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
            print(f"Using LLM: {MODEL_NAME} @ {API_BASE_URL}", flush=True, file=sys.stderr)
        except Exception as exc:
            print(f"[DEBUG] Client init failed: {exc}", flush=True, file=sys.stderr)
    else:
        print("No HF_TOKEN/GROQ_API_KEY found. Using deterministic fallback.", file=sys.stderr)

    env_client = ERTriageEnvClient(base_url=ENV_BASE_URL)

    for task in ("easy", "medium", "hard"):
        run_task(task=task, client=client, env_client=env_client)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
