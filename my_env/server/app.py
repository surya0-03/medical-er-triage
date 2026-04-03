from __future__ import annotations

from threading import Lock
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from .environment import (
    ARRIVAL_BASE_RATE,
    BED_GRADE_ESI1_NO_BED,
    BED_GRADE_ESI45_TO_ICU,
    BED_GRADE_GENERAL_CORRECT,
    BED_GRADE_ICU_CORRECT,
    DEFAULT_RANDOM_SEED,
    DETERIORATION_EVENT_PENALTY,
    DETERIORATION_RULES,
    ESI_GRADE_DIFF_1,
    ESI_GRADE_DIFF_2,
    ESI_GRADE_ESI1_TO_45,
    ESI_GRADE_ESI2_TO_5,
    ESI_GRADE_EXACT,
    ESI12_DELAY_PENALTY_PER_STEP,
    GENERAL_BEDS_FREE_HARD_START,
    GENERAL_BEDS_TOTAL,
    HALLWAY_CAPACITY,
    HARD_DEATH_CAP,
    HARD_START_HOUR,
    ICU_BEDS_FREE_HARD_START,
    ICU_BEDS_TOTAL,
    OUTCOME_BASE_PROB,
    OUTCOME_CRITICAL_REWARD,
    OUTCOME_DEATH_REWARD,
    OUTCOME_DELAY_BY_ESI,
    OUTCOME_DETERIORATION_FACTOR,
    OUTCOME_ESI_ERROR_FACTOR,
    OUTCOME_RECOVER_REWARD,
    OUTCOME_STABLE_REWARD,
    OUTCOME_WAIT_FACTOR,
    OUTCOME_WRONG_BED_FACTOR,
    PROTOCOL_BONUS_CORRECT,
    PROTOCOL_BONUS_WRONG,
    REWARD_CLAMP_MAX,
    REWARD_CLAMP_MIN,
    SAFE_WAIT_LIMITS,
    STAFF_OVERLOAD_PENALTY_FACTOR,
    TASK_STEPS,
    TIME_STEP_MINUTES,
    MedicalEmergencyRoomEnv,
)
from ..models import Action


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    # Accept both "task" (OpenEnv standard) and "difficulty" (legacy) field names
    task: Literal["easy", "medium", "hard"] | None = None
    difficulty: Literal["easy", "medium", "hard"] | None = None
    seed: int = Field(default=DEFAULT_RANDOM_SEED)
    session_id: StrictStr | None = None

    @model_validator(mode="after")
    def resolve_task_or_difficulty(self) -> "ResetRequest":
        if self.task is None and self.difficulty is None:
            # Default to medium if neither provided
            object.__setattr__(self, "difficulty", "medium")
        return self

    @property
    def resolved_difficulty(self) -> Literal["easy", "medium", "hard"]:
        return self.task or self.difficulty or "medium"


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    session_id: StrictStr | None = None
    action: Action


class EnvironmentServerState:
    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[str, dict[str, Any]] = {}

    def reset(self, difficulty: Literal["easy", "medium", "hard"], seed: int, session_id: str | None) -> dict[str, Any]:
        with self._lock:
            sid = session_id if session_id is not None else "default"
            session = self._sessions.get(sid)
            if session is None or session["difficulty"] != difficulty:
                env = MedicalEmergencyRoomEnv(difficulty=difficulty, seed=seed)
                session = {
                    "difficulty": difficulty,
                    "seed": seed,
                    "env": env,
                }
                self._sessions[sid] = session

            session["seed"] = seed
            session["difficulty"] = difficulty
            env = session["env"]
            observation = env.reset(seed=seed)
            return {
                "observation": observation.model_dump(mode="python"),
                "action_mask": env.get_action_mask(),
                "difficulty": difficulty,
                "seed": seed,
                "session_id": sid,
            }

    def step(self, session_id: str | None, action: Action) -> dict[str, Any]:
        with self._lock:
            sid = session_id if session_id is not None else "default"
            session = self._sessions.get(sid)
            if session is None:
                raise KeyError("unknown session_id")

            env = session["env"]
            observation, reward, done, info = env.step(action)
            terminated = bool(info.get("terminated", done))
            truncated = bool(info.get("truncated", False))
            return {
                "observation": observation.model_dump(mode="python"),
                "action_mask": env.get_action_mask(),
                "reward": reward.model_dump(mode="python"),
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
                "session_id": sid,
            }

    def state(self, session_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            sid = session_id if session_id is not None else "default"
            session = self._sessions.get(sid)
            if session is None:
                raise KeyError("unknown session_id")
            env = session["env"]
            state = env.state()
            state["session_id"] = sid
            state["action_mask"] = env.get_action_mask()
            return state


app = FastAPI(title="OpenEnv Medical ER Triage", version="1.0.0")
_runtime = EnvironmentServerState()


@app.post("/reset")
def reset_environment(request: ResetRequest) -> dict[str, Any]:
    return _runtime.reset(difficulty=request.resolved_difficulty, seed=request.seed, session_id=request.session_id)


@app.post("/step")
def step_environment(request: StepRequest) -> dict[str, Any]:
    try:
        return _runtime.step(session_id=request.session_id, action=request.action)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown session_id; call /reset first") from exc


@app.get("/state")
def get_state(session_id: str = Query(default="default", min_length=1)) -> dict[str, Any]:
    try:
        return _runtime.state(session_id=session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown session_id; call /reset first") from exc


@app.get("/grade")
def grade_episode(session_id: str = Query(default="default", min_length=1)) -> dict[str, Any]:
    try:
        with _runtime._lock:
            session = _runtime._sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="unknown session_id; call /reset first")
            env = session["env"]
            s = env.state()
            avg = float(s.get("average_step_reward", 0.0))
            score = round(max(0.0, min(1.0, (avg + 1.45) / 2.90)), 4)
            return {
                "score": score,
                "passed": score >= 0.70,
                "difficulty": s["difficulty"],
                "total_steps": s["time_step"],
                "hard_esi1_deaths": s["hard_esi1_deaths"],
                "average_step_reward": round(avg, 6),
                "done": s["done"],
                "session_id": session_id,
            }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown session_id") from exc

@app.get("/score")
def detailed_score(session_id: str = Query(default="default", min_length=1)) -> dict[str, Any]:
    try:
        with _runtime._lock:
            session = _runtime._sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="unknown session_id; call /reset first")
            env = session["env"]
            s = env.state()
            avg = float(s.get("average_step_reward", 0.0))
            score = round(max(0.0, min(1.0, (avg + 1.45) / 2.90)), 4)

            priority_queue = []
            for runtime in env._patients.values():
                if runtime.patient.state == "waiting":
                    priority_queue.append({
                        "patient_id": runtime.patient.patient_id,
                        "true_esi": runtime.true_esi,
                        "wait_time": runtime.patient.wait_time,
                        "deterioration_count": runtime.patient.deterioration_count,
                        "acuity_score": round(
                            min(100.0,
                                (6 - runtime.patient.esi_level) * 15.0
                                + max(0, runtime.patient.wait_time - SAFE_WAIT_LIMITS[runtime.patient.esi_level]) * 5.0
                                + runtime.patient.deterioration_count * 10.0
                            ), 1
                        ),
                    })
            priority_queue.sort(key=lambda x: (-x["acuity_score"], x["true_esi"]))

            hour = ((env._time_step * TIME_STEP_MINUTES) // 60) % 24
            minute = ((env._time_step * TIME_STEP_MINUTES) % 60)

            return {
                "score": score,
                "passed": score >= 0.70,
                "difficulty": s["difficulty"],
                "total_steps": s["time_step"],
                "hard_esi1_deaths": s["hard_esi1_deaths"],
                "average_step_reward": round(avg, 6),
                "done": s["done"],
                "session_id": session_id,
                "priority_queue": priority_queue[:5],
                "time_of_day": f"{hour:02d}:{minute:02d}",
                "is_peak_hours": 17 <= hour <= 21,
                "waiting_count": len(env._waiting_queue),
                "beds_in_use": len(env._beds),
                "outcomes_pending": len(env._delayed_outcomes),
                "mortality_rate": round(env._mortality_rate(), 4),
                "avg_wait_time": round(env._average_wait_time(), 2),
            }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown session_id") from exc

@app.get("/health")
def health_check() -> dict[str, Any]:
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/tasks")
def get_tasks() -> dict[str, Any]:
    return {
        "time_step_minutes": TIME_STEP_MINUTES,
        "tasks": {
            "easy": {
                "max_steps": TASK_STEPS["easy"],
                "termination": "step_limit",
            },
            "medium": {
                "max_steps": TASK_STEPS["medium"],
                "termination": "step_limit",
            },
            "hard": {
                "max_steps": TASK_STEPS["hard"],
                "max_esi1_deaths": HARD_DEATH_CAP,
                "termination": "step_limit_or_esi1_deaths",
            },
        },
        "hard_start": {
            "hour": HARD_START_HOUR,
            "icu_beds_free": ICU_BEDS_FREE_HARD_START,
            "general_beds_free": GENERAL_BEDS_FREE_HARD_START,
        },
        "resources": {
            "icu_beds_total": ICU_BEDS_TOTAL,
            "general_beds_total": GENERAL_BEDS_TOTAL,
            "hallway_capacity": HALLWAY_CAPACITY,
        },
    }


@app.get("/grader")
def get_grader() -> dict[str, Any]:
    return {
        "reward_formula": "score = clamp01((raw_step_reward + bounded_outcome_component + (1 + outcome_cap)) / (2 * (1 + outcome_cap)))",
        "reward_clamp": [REWARD_CLAMP_MIN, REWARD_CLAMP_MAX],
        "esi": {
            "exact": ESI_GRADE_EXACT,
            "diff_1": ESI_GRADE_DIFF_1,
            "diff_2": ESI_GRADE_DIFF_2,
            "esi1_to_4_or_5": ESI_GRADE_ESI1_TO_45,
            "esi2_to_5": ESI_GRADE_ESI2_TO_5,
        },
        "bed": {
            "icu_correct": BED_GRADE_ICU_CORRECT,
            "general_correct": BED_GRADE_GENERAL_CORRECT,
            "esi1_no_bed": BED_GRADE_ESI1_NO_BED,
            "esi4_or_5_to_icu": BED_GRADE_ESI45_TO_ICU,
        },
        "protocol": {
            "correct": PROTOCOL_BONUS_CORRECT,
            "wrong": PROTOCOL_BONUS_WRONG,
        },
        "deterioration": {
            "event_penalty": DETERIORATION_EVENT_PENALTY,
            "esi1_esi2_delay_penalty_per_step": ESI12_DELAY_PENALTY_PER_STEP,
            "safe_wait_limits": SAFE_WAIT_LIMITS,
            "rules": {
                str(k): {
                    "probability": v[0],
                    "to_esi": v[1],
                }
                for k, v in DETERIORATION_RULES.items()
            },
        },
        "staffing": {
            "overload_penalty_factor": STAFF_OVERLOAD_PENALTY_FACTOR,
        },
        "outcomes": {
            "base_probability": OUTCOME_BASE_PROB,
            "factors": {
                "esi_error": OUTCOME_ESI_ERROR_FACTOR,
                "wait": OUTCOME_WAIT_FACTOR,
                "deterioration": OUTCOME_DETERIORATION_FACTOR,
                "wrong_bed": OUTCOME_WRONG_BED_FACTOR,
            },
            "reward_mapping": {
                "recover": OUTCOME_RECOVER_REWARD,
                "stable": OUTCOME_STABLE_REWARD,
                "critical": OUTCOME_CRITICAL_REWARD,
                "death": OUTCOME_DEATH_REWARD,
            },
            "delays": OUTCOME_DELAY_BY_ESI,
        },
    }


@app.get("/baseline")
def get_baseline() -> dict[str, Any]:
    return {
        "name": "openai_policy_baseline",
        "seed": DEFAULT_RANDOM_SEED,
        "required_env": ["HF_TOKEN", "API_BASE_URL", "MODEL_NAME"],
        "optional_env": {
            "OPENAI_INFERENCE_SEED": "12345",
        },
        "arrival_model": {
            "rate_formula": "0.4 * hourly_multiplier",
            "base_rate": ARRIVAL_BASE_RATE,
            "hard_start_hour": HARD_START_HOUR,
        },
        "policy": {
            "action_source": "OpenAI chat completion with strict JSON action schema",
            "temperature": 0.0,
            "reproducibility": "fixed environment seeds + OPENAI_INFERENCE_SEED",
        },
        "notes": {
            "determinism": "seeded environment dynamics and fixed baseline inference settings",
            "json_contract": "all endpoints return JSON-compatible dict payloads",
        },
    }
