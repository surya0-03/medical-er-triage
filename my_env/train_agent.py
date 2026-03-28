from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from statistics import mean
from typing import Any, Literal

from .models import Action
from .server.environment import MedicalEmergencyRoomEnv


TaskName = Literal["easy", "medium", "hard"]


CRITICAL_SYMPTOMS = {
    "stridor",
    "cyanosis",
    "severe dyspnea",
    "respiratory distress",
    "unresponsiveness",
    "polytrauma",
    "active bleeding",
    "hypotension",
    "crushing chest pain",
    "facial droop",
    "slurred speech",
    "unilateral weakness",
    "altered mental status",
}

SEVERE_SYMPTOMS = {
    "shortness of breath",
    "diaphoresis",
    "tachypnea",
    "wheezing",
    "chest tightness",
    "fever",
    "tachycardia",
}


@dataclass
class TrainConfig:
    iterations: int = 10
    population: int = 16
    elite_frac: float = 0.25
    noise_std: float = 0.35
    seed: int = 42


@dataclass
class TrainResult:
    best_train_params: list[float]
    best_train_score: float
    selected_params: list[float]
    selected_score: float
    selection_seeds: list[int]


DEFAULT_PARAMS: list[float] = [
    0.4,
    1.2,
    0.03,
    0.5,
    2.5,
    1.0,
    0.04,
    1.1,
    0.2,
    2.8,
    1.1,
    6.5,
    4.0,
    2.2,
    0.8,
    1.3,
    0.35,
]


class ParamHeuristicAgent:
    """Small parametric policy trained via simple evolutionary search."""

    def __init__(self, params: list[float], rng: random.Random) -> None:
        self.p = params
        self.rng = rng

    def choose_action(self, env: MedicalEmergencyRoomEnv) -> Action:
        mask = env.get_action_mask()
        obs = env.state()["observation"]
        patients = {str(p["patient_id"]): p for p in obs.get("patients", [])}

        assign_ids: list[str] = list(mask.get("assign_esi_patient_ids", []))
        if assign_ids:
            pid = max(assign_ids, key=lambda x: self._assign_priority(patients.get(x, {})))
            patient = patients.get(pid, {})
            esi = self._predict_esi(patient)
            return Action(action_type="assign_esi", patient_id=pid, esi_level=esi)

        alloc_map: dict[str, list[str]] = mask.get("allocate_bed", {})
        if alloc_map:
            pid = max(alloc_map.keys(), key=lambda x: self._alloc_priority(patients.get(x, {})))
            allowed = [b for b in alloc_map[pid] if b in {"icu", "general", "hallway"}]
            if allowed:
                patient = patients.get(pid, {})
                esi = int(patient.get("esi_level", 3))
                if esi <= 2 and "icu" in allowed:
                    bed = "icu"
                elif "general" in allowed:
                    bed = "general"
                elif "hallway" in allowed:
                    bed = "hallway"
                else:
                    bed = allowed[0]
                return Action(action_type="allocate_bed", patient_id=pid, bed_type=bed)

        protocol_map: dict[str, list[str]] = mask.get("trigger_protocol", {})
        if protocol_map:
            best_pid = None
            best_proto = None
            best_score = -1.0
            for pid, protocols in protocol_map.items():
                patient = patients.get(pid, {})
                symptoms = set(str(s).lower() for s in patient.get("symptoms", []))
                for proto in protocols:
                    score = self._protocol_match_score(proto, symptoms)
                    if score > best_score:
                        best_score = score
                        best_pid = pid
                        best_proto = proto
            if best_pid is not None and best_proto is not None and best_score > 0:
                return Action(action_type="trigger_protocol", patient_id=best_pid, protocol_type=best_proto)

        discharge_ids: list[str] = list(mask.get("discharge_patient_ids", []))
        if discharge_ids:
            pid = max(discharge_ids, key=lambda x: int(patients.get(x, {}).get("wait_time", 0)))
            return Action(action_type="discharge", patient_id=pid)

        return Action(action_type="divert")

    def _assign_priority(self, patient: dict[str, Any]) -> float:
        esi = float(patient.get("esi_level", 3))
        wait_time = float(patient.get("wait_time", 0))
        vitals = patient.get("vitals", {})
        hr = float(vitals.get("heart_rate", 90))
        oxy = float(vitals.get("oxygen_level", 98))

        symptoms = set(str(s).lower() for s in patient.get("symptoms", []))
        critical_hits = sum(1 for s in symptoms if s in CRITICAL_SYMPTOMS)
        severe_hits = sum(1 for s in symptoms if s in SEVERE_SYMPTOMS)

        return (
            self.p[0] * wait_time
            + self.p[1] * (6.0 - esi)
            + self.p[2] * max(0.0, hr - 100.0)
            + self.p[3] * max(0.0, 95.0 - oxy)
            + self.p[4] * critical_hits
            + self.p[5] * severe_hits
        )

    def _predict_esi(self, patient: dict[str, Any]) -> int:
        vitals = patient.get("vitals", {})
        hr = float(vitals.get("heart_rate", 90))
        oxy = float(vitals.get("oxygen_level", 98))
        wait_time = float(patient.get("wait_time", 0))
        symptoms = set(str(s).lower() for s in patient.get("symptoms", []))

        critical_hits = sum(1 for s in symptoms if s in CRITICAL_SYMPTOMS)
        severe_hits = sum(1 for s in symptoms if s in SEVERE_SYMPTOMS)

        risk = (
            self.p[6] * max(0.0, hr - 100.0)
            + self.p[7] * max(0.0, 95.0 - oxy)
            + self.p[8] * wait_time
            + self.p[9] * critical_hits
            + self.p[10] * severe_hits
        )

        if risk >= self.p[11]:
            return 1
        if risk >= self.p[12]:
            return 2
        if risk >= self.p[13]:
            return 3
        if risk >= self.p[14]:
            return 4
        return 5

    def _alloc_priority(self, patient: dict[str, Any]) -> float:
        esi = float(patient.get("esi_level", 3))
        wait_time = float(patient.get("wait_time", 0))
        return self.p[15] * (6.0 - esi) + self.p[16] * wait_time

    @staticmethod
    def _protocol_match_score(protocol: str, symptoms: set[str]) -> float:
        reqs = {
            "stroke_code": {"facial droop", "slurred speech", "unilateral weakness"},
            "stemi_alert": {"crushing chest pain", "diaphoresis", "shortness of breath"},
            "sepsis_alert": {"fever", "tachycardia", "altered mental status"},
            "trauma_alert": {"polytrauma", "hypotension", "active bleeding"},
        }
        needed = reqs.get(protocol, set())
        if not needed:
            return 0.0
        return float(len(symptoms.intersection(needed))) / float(len(needed))


def _parse_seed_csv(raw: str) -> list[int]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    return [int(x) for x in parts]


def evaluate_params(params: list[float], seed: int) -> float:
    rng = random.Random(seed)
    agent = ParamHeuristicAgent(params=params, rng=rng)

    tasks: list[TaskName] = ["easy", "medium", "hard"]
    task_scores: list[float] = []
    for i, task in enumerate(tasks):
        env = MedicalEmergencyRoomEnv(difficulty=task, seed=seed + i)
        env.reset(seed=seed + i)

        total = 0.0
        steps = 0
        done = False
        while not done and steps < 500:
            action = agent.choose_action(env)
            _, reward, done, _ = env.step(action)
            total += float(reward.value)
            steps += 1

        task_scores.append(total / float(max(steps, 1)))

    return mean(task_scores)


def evaluate_params_by_task(params: list[float], seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    agent = ParamHeuristicAgent(params=params, rng=rng)

    task_scores: dict[str, float] = {}
    for i, task in enumerate(["easy", "medium", "hard"]):
        env = MedicalEmergencyRoomEnv(difficulty=task, seed=seed + i)
        env.reset(seed=seed + i)

        total = 0.0
        steps = 0
        done = False
        while not done and steps < 500:
            action = agent.choose_action(env)
            _, reward, done, _ = env.step(action)
            total += float(reward.value)
            steps += 1

        task_scores[task] = total / float(max(steps, 1))

    task_scores["macro"] = mean([task_scores["easy"], task_scores["medium"], task_scores["hard"]])
    return task_scores


def evaluate_params_over_seeds(params: list[float], seeds: list[int]) -> float:
    if not seeds:
        raise ValueError("seeds must be non-empty")
    return mean([evaluate_params(params, seed) for seed in seeds])


def save_checkpoint(path: str, params: list[float], score: float, cfg: TrainConfig, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "best_score": score,
        "best_params": params,
        "config": cfg.__dict__,
    }
    if extra:
        payload.update(extra)
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_checkpoint(path: str) -> tuple[list[float], float]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    params = [float(x) for x in payload["best_params"]]
    score = float(payload.get("best_score", 0.0))
    return params, score


def train(
    cfg: TrainConfig,
    init_params: list[float] | None = None,
    selection_seeds: list[int] | None = None,
    selection_topk: int = 1,
) -> TrainResult:
    rng = random.Random(cfg.seed)
    selection_seeds = list(selection_seeds or [])
    selection_topk = max(1, selection_topk)

    mean_params = list(init_params) if init_params is not None else list(DEFAULT_PARAMS)

    best_train_params = list(mean_params)
    best_train_score = evaluate_params(best_train_params, cfg.seed)

    if selection_seeds:
        selected_params = list(best_train_params)
        selected_score = evaluate_params_over_seeds(selected_params, selection_seeds)
        print(
            f"iter=0 train_best={best_train_score:.4f} selected_holdout={selected_score:.4f} "
            f"seeds={selection_seeds}"
        )
    else:
        selected_params = list(best_train_params)
        selected_score = best_train_score
        print(f"iter=0 best_score={best_train_score:.4f}")

    elite_count = max(1, int(cfg.population * cfg.elite_frac))

    for it in range(1, cfg.iterations + 1):
        candidates: list[tuple[float, list[float]]] = []
        for j in range(cfg.population):
            cand = [m + rng.gauss(0.0, cfg.noise_std) for m in mean_params]
            # Enforce monotonically descending thresholds for ESI mapping.
            sorted_th = sorted([cand[11], cand[12], cand[13], cand[14]], reverse=True)
            cand[11], cand[12], cand[13], cand[14] = sorted_th

            score = evaluate_params(cand, cfg.seed + 100 * it + j)
            candidates.append((score, cand))

        candidates.sort(key=lambda x: x[0], reverse=True)
        elites = candidates[:elite_count]

        mean_params = [mean([e[1][k] for e in elites]) for k in range(len(mean_params))]
        if elites[0][0] > best_train_score:
            best_train_score = elites[0][0]
            best_train_params = list(elites[0][1])

        if selection_seeds:
            for _, cand in elites[:selection_topk]:
                holdout_score = evaluate_params_over_seeds(cand, selection_seeds)
                if holdout_score > selected_score:
                    selected_score = holdout_score
                    selected_params = list(cand)
            print(
                f"iter={it} top={elites[0][0]:.4f} mean_top={mean([e[0] for e in elites]):.4f} "
                f"train_best={best_train_score:.4f} holdout_best={selected_score:.4f}"
            )
        else:
            print(
                f"iter={it} top={elites[0][0]:.4f} mean_top={mean([e[0] for e in elites]):.4f} "
                f"best={best_train_score:.4f}"
            )

    return TrainResult(
        best_train_params=best_train_params,
        best_train_score=best_train_score,
        selected_params=selected_params,
        selected_score=selected_score,
        selection_seeds=selection_seeds,
    )


def main() -> None:
    cfg = TrainConfig(
        iterations=int(os.environ.get("TRAIN_ITERS", "10")),
        population=int(os.environ.get("TRAIN_POP", "16")),
        elite_frac=float(os.environ.get("TRAIN_ELITE_FRAC", "0.25")),
        noise_std=float(os.environ.get("TRAIN_NOISE_STD", "0.35")),
        seed=int(os.environ.get("TRAIN_SEED", "42")),
    )

    init_checkpoint = os.environ.get("TRAIN_INIT_CHECKPOINT", "").strip()
    init_params = None
    if init_checkpoint:
        init_params, init_score = load_checkpoint(init_checkpoint)
        print(f"loaded_init_checkpoint={init_checkpoint} score={init_score:.4f}")

    selection_seed_csv = os.environ.get("TRAIN_SELECTION_SEEDS", "901,902,903,904,905").strip()
    selection_seeds = _parse_seed_csv(selection_seed_csv) if selection_seed_csv else []
    selection_topk = int(os.environ.get("TRAIN_SELECTION_TOPK", "2"))

    result = train(
        cfg,
        init_params=init_params,
        selection_seeds=selection_seeds,
        selection_topk=selection_topk,
    )

    best_params = list(result.selected_params)
    best_score = float(result.selected_score)

    output_checkpoint = os.environ.get("TRAIN_OUTPUT_CHECKPOINT", "my_env/checkpoints/best_params.json")
    save_checkpoint(
        output_checkpoint,
        best_params,
        best_score,
        cfg,
        extra={
            "best_train_score": result.best_train_score,
            "best_train_params": result.best_train_params,
            "selection_seeds": result.selection_seeds,
            "selection_topk": selection_topk,
        },
    )
    print(f"saved_checkpoint={output_checkpoint}")

    task_eval = evaluate_params_by_task(best_params, cfg.seed + 999)

    summary = {
        "selected_score": best_score,
        "selected_params": best_params,
        "best_train_score": result.best_train_score,
        "selection_seeds": result.selection_seeds,
        "selection_topk": selection_topk,
        "task_eval_seed": cfg.seed + 999,
        "task_eval": task_eval,
        "checkpoint": output_checkpoint,
        "config": cfg.__dict__,
    }

    print("=== TRAINING SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
