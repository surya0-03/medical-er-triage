from __future__ import annotations

import json
import os
from statistics import mean

from my_env.train_agent import DEFAULT_PARAMS, evaluate_params_by_task, load_checkpoint


def _parse_int_list_or_ranges(raw: str) -> list[int]:
    values: list[int] = []
    for part in [p.strip() for p in raw.split(",") if p.strip()]:
        if "-" in part:
            start_str, end_str = [x.strip() for x in part.split("-", 1)]
            start = int(start_str)
            end = int(end_str)
            if end < start:
                start, end = end, start
            values.extend(list(range(start, end + 1)))
        else:
            values.append(int(part))
    dedup_sorted = sorted(set(values))
    if not dedup_sorted:
        raise ValueError("seed list cannot be empty")
    return dedup_sorted


def _evaluate(params: list[float], seeds: list[int]) -> dict[str, float]:
    easy_scores: list[float] = []
    medium_scores: list[float] = []
    hard_scores: list[float] = []
    macro_scores: list[float] = []

    for seed in seeds:
        task_scores = evaluate_params_by_task(params, seed)
        easy_scores.append(task_scores["easy"])
        medium_scores.append(task_scores["medium"])
        hard_scores.append(task_scores["hard"])
        macro_scores.append(task_scores["macro"])

    return {
        "easy": mean(easy_scores),
        "medium": mean(medium_scores),
        "hard": mean(hard_scores),
        "macro": mean(macro_scores),
    }


def _slice_decision(
    baseline: dict[str, float],
    candidate: dict[str, float],
    min_macro_delta: float,
    min_hard_delta: float,
) -> dict[str, object]:
    delta_macro = candidate["macro"] - baseline["macro"]
    delta_hard = candidate["hard"] - baseline["hard"]

    macro_ok = delta_macro >= min_macro_delta
    hard_ok = delta_hard >= min_hard_delta

    return {
        "delta_macro": delta_macro,
        "delta_hard": delta_hard,
        "macro_ok": macro_ok,
        "hard_ok": hard_ok,
        "pass": macro_ok and hard_ok,
    }


def main() -> None:
    checkpoint_path = os.environ.get("CANARY_CHECKPOINT", "my_env/checkpoints/best_params_holdout.json")

    slice_a_raw = os.environ.get("CANARY_SLICE_A", "301-325")
    slice_b_raw = os.environ.get("CANARY_SLICE_B", "401-425")

    min_macro_delta = float(os.environ.get("CANARY_MIN_MACRO_DELTA", "-0.002"))
    min_hard_delta = float(os.environ.get("CANARY_MIN_HARD_DELTA", "-0.002"))

    slice_a_seeds = _parse_int_list_or_ranges(slice_a_raw)
    slice_b_seeds = _parse_int_list_or_ranges(slice_b_raw)

    baseline_params = list(DEFAULT_PARAMS)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    candidate_params, candidate_checkpoint_score = load_checkpoint(checkpoint_path)

    baseline_a = _evaluate(baseline_params, slice_a_seeds)
    baseline_b = _evaluate(baseline_params, slice_b_seeds)
    candidate_a = _evaluate(candidate_params, slice_a_seeds)
    candidate_b = _evaluate(candidate_params, slice_b_seeds)

    decision_a = _slice_decision(baseline_a, candidate_a, min_macro_delta, min_hard_delta)
    decision_b = _slice_decision(baseline_b, candidate_b, min_macro_delta, min_hard_delta)

    overall_pass = bool(decision_a["pass"] and decision_b["pass"])

    payload = {
        "checkpoint": checkpoint_path,
        "checkpoint_score": candidate_checkpoint_score,
        "policy": "promote" if overall_pass else "rollback",
        "overall_pass": overall_pass,
        "thresholds": {
            "min_macro_delta": min_macro_delta,
            "min_hard_delta": min_hard_delta,
        },
        "slice_a": {
            "seeds": slice_a_seeds,
            "baseline": baseline_a,
            "candidate": candidate_a,
            "decision": decision_a,
        },
        "slice_b": {
            "seeds": slice_b_seeds,
            "baseline": baseline_b,
            "candidate": candidate_b,
            "decision": decision_b,
        },
    }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
