from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

from .models import Action
from .server.environment import DEFAULT_RANDOM_SEED, MedicalEmergencyRoomEnv


TaskName = Literal["easy", "medium", "hard"]


SYSTEM_PROMPT = (
    "You are a medical ER triage baseline policy. "
    "Return exactly one JSON object with keys: action_type, patient_id, esi_level, bed_type, protocol_type. "
    "Valid action_type: assign_esi, allocate_bed, discharge, trigger_protocol, divert. "
    "Use null for fields not required by the selected action."
)


@dataclass
class TaskResult:
    task: TaskName
    steps: int
    total_reward: float
    average_reward: float
    terminated: bool
    truncated: bool


class DeterministicFallbackAgent:
    def choose_action(self, task: TaskName, env_state: dict[str, Any], action_mask: dict[str, Any]) -> Action:  # noqa: ARG002
        assign_ids = sorted(action_mask.get("assign_esi_patient_ids", []))
        if assign_ids:
            patient_map = {
                str(patient["patient_id"]): int(patient["esi_level"])
                for patient in env_state["observation"].get("patients", [])
            }
            pid = assign_ids[0]
            esi_level = patient_map.get(pid, 3)
            return Action(action_type="assign_esi", patient_id=pid, esi_level=esi_level)

        allocate = action_mask.get("allocate_bed", {})
        for pid in sorted(allocate.keys()):
            choices = set(allocate[pid])
            if "icu" in choices:
                return Action(action_type="allocate_bed", patient_id=pid, bed_type="icu")
            if "general" in choices:
                return Action(action_type="allocate_bed", patient_id=pid, bed_type="general")
            if "hallway" in choices:
                return Action(action_type="allocate_bed", patient_id=pid, bed_type="hallway")

        protocol_map = action_mask.get("trigger_protocol", {})
        for pid in sorted(protocol_map.keys()):
            protocols = sorted(protocol_map[pid])
            if protocols:
                return Action(action_type="trigger_protocol", patient_id=pid, protocol_type=protocols[0])

        discharge_ids = sorted(action_mask.get("discharge_patient_ids", []))
        if discharge_ids:
            return Action(action_type="discharge", patient_id=discharge_ids[0])

        return Action(action_type="divert")


class OpenAIBaselineAgent:
    def __init__(self, client: Any, model: str, temperature: float, seed: int) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._seed = seed

    def choose_action(self, task: TaskName, env_state: dict[str, Any], action_mask: dict[str, Any]) -> Action:
        prompt = self._build_user_prompt(task=task, env_state=env_state, action_mask=action_mask)

        completion = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            seed=self._seed,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        content = completion.choices[0].message.content
        if content is None:
            raise RuntimeError("OpenAI response content is empty")

        payload = self._extract_json(content)
        return Action(
            action_type=payload.get("action_type"),
            patient_id=payload.get("patient_id"),
            esi_level=payload.get("esi_level"),
            bed_type=payload.get("bed_type"),
            protocol_type=payload.get("protocol_type"),
        )

    def _build_user_prompt(self, task: TaskName, env_state: dict[str, Any], action_mask: dict[str, Any]) -> str:
        observation = env_state["observation"]
        waiting_queue: list[str] = observation.get("waiting_queue", [])
        patients: list[dict[str, Any]] = observation.get("patients", [])

        queue_index = {patient_id: idx for idx, patient_id in enumerate(waiting_queue)}
        waiting_patients = []
        for patient in patients:
            pid = str(patient["patient_id"])
            if pid in queue_index:
                waiting_patients.append(
                    {
                        "patient_id": pid,
                        "queue_position": queue_index[pid],
                        "esi_level": int(patient["esi_level"]),
                        "wait_time": int(patient["wait_time"]),
                        "symptoms": list(patient["symptoms"]),
                        "state": str(patient["state"]),
                    }
                )

        waiting_patients.sort(key=lambda item: (item["esi_level"], -item["wait_time"], item["queue_position"]))

        summary = {
            "task": task,
            "time_step": int(observation["time_step"]),
            "icu_beds_available": int(observation["icu_beds_available"]),
            "general_beds_available": int(observation["general_beds_available"]),
            "hallway_used": int(observation["hallway_used"]),
            "crowding_score": float(observation["crowding_score"]),
            "staff_load": observation["staff_load"],
            "waiting_patients_top": waiting_patients[:8],
            "waiting_queue_size": len(waiting_queue),
            "action_mask": action_mask,
        }
        return json.dumps(summary, separators=(",", ":"))

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any]:
        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if len(lines) >= 3:
                content = "\n".join(lines[1:-1]).strip()
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise RuntimeError("OpenAI output must be a JSON object")
        return payload


class BaselineRunner:
    def __init__(self, agent: Any) -> None:
        self._agent = agent

    def run_task(self, task: TaskName, seed: int) -> TaskResult:
        env = MedicalEmergencyRoomEnv(difficulty=task, seed=seed)
        env.reset(seed=seed)

        total_reward = 0.0
        steps = 0
        done = False
        info: dict[str, Any] = {}

        while not done:
            env_state = env.state()
            action_mask = env.get_action_mask()
            try:
                action = self._agent.choose_action(task=task, env_state=env_state, action_mask=action_mask)
            except Exception:
                action = Action(action_type="divert")

            _, reward, done, info = env.step(action)
            total_reward += float(reward.value)
            steps += 1
            if steps > 500:
                break

        average_reward = total_reward / float(steps) if steps > 0 else 0.0
        return TaskResult(
            task=task,
            steps=steps,
            total_reward=round(total_reward, 6),
            average_reward=round(average_reward, 6),
            terminated=bool(info.get("terminated", done)),
            truncated=bool(info.get("truncated", False)),
        )


def main() -> None:
    fallback_mode = os.environ.get("OPENAI_BASELINE_FALLBACK", "").strip().lower()
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    inference_seed = int(os.environ.get("OPENAI_INFERENCE_SEED", "12345"))

    run_label = "DeterministicFallback"
    runner: BaselineRunner

    if api_key and fallback_mode != "deterministic":
        try:
            from openai import OpenAI  # Optional dependency

            client = OpenAI(api_key=api_key)
            runner = BaselineRunner(agent=OpenAIBaselineAgent(client=client, model=model, temperature=0.0, seed=inference_seed))
            run_label = "OpenAI"
        except Exception:
            runner = BaselineRunner(agent=DeterministicFallbackAgent())
    else:
        runner = BaselineRunner(agent=DeterministicFallbackAgent())

    task_order: list[TaskName] = ["easy", "medium", "hard"]
    seeds = {
        "easy": DEFAULT_RANDOM_SEED,
        "medium": DEFAULT_RANDOM_SEED + 1,
        "hard": DEFAULT_RANDOM_SEED + 2,
    }

    results = [runner.run_task(task=task, seed=seeds[task]) for task in task_order]

    print(f"=== {run_label} Baseline Scores ===")
    for result in results:
        print(
            f"task={result.task} "
            f"steps={result.steps} "
            f"total_reward={result.total_reward:.6f} "
            f"avg_reward={result.average_reward:.6f} "
            f"terminated={result.terminated} "
            f"truncated={result.truncated}"
        )


if __name__ == "__main__":
    main()
