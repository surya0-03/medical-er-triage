from __future__ import annotations

import json
import os
from statistics import mean

from my_env.train_agent import DEFAULT_PARAMS, evaluate_params_by_task, load_checkpoint


def _parse_seeds(raw: str) -> list[int]:
    items = [s.strip() for s in raw.split(",") if s.strip()]
    return [int(x) for x in items]


def evaluate_over_seeds(params: list[float], seeds: list[int]) -> dict[str, float]:
    easy_scores: list[float] = []
    medium_scores: list[float] = []
    hard_scores: list[float] = []
    macro_scores: list[float] = []

    for seed in seeds:
        score = evaluate_params_by_task(params, seed)
        easy_scores.append(score["easy"])
        medium_scores.append(score["medium"])
        hard_scores.append(score["hard"])
        macro_scores.append(score["macro"])

    return {
        "easy": mean(easy_scores),
        "medium": mean(medium_scores),
        "hard": mean(hard_scores),
        "macro": mean(macro_scores),
    }


def main() -> None:
    seeds = _parse_seeds(os.environ.get("EVAL_SEEDS", "101,202,303,404,505"))

    baseline_params = list(DEFAULT_PARAMS)
    baseline_result = evaluate_over_seeds(baseline_params, seeds)

    checkpoint_path = os.environ.get("EVAL_CHECKPOINT", "my_env/checkpoints/best_params.json")
    trained_result = None
    trained_params: list[float] | None = None
    if os.path.exists(checkpoint_path):
        trained_params, best_score = load_checkpoint(checkpoint_path)
        trained_result = evaluate_over_seeds(trained_params, seeds)
        print(f"loaded_checkpoint={checkpoint_path} score={best_score:.4f}")
    else:
        print(f"checkpoint_not_found={checkpoint_path}")

    payload: dict[str, object] = {
        "seeds": seeds,
        "baseline": baseline_result,
        "checkpoint": checkpoint_path,
    }

    if trained_result is not None and trained_params is not None:
        payload["trained"] = trained_result
        payload["delta_macro"] = trained_result["macro"] - baseline_result["macro"]

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
