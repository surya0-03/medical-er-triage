from __future__ import annotations

from typing import Final

TIME_STEP_MINUTES: Final[int] = 10

ICU_BEDS_TOTAL: Final[int] = 8
GENERAL_BEDS_TOTAL: Final[int] = 20
HALLWAY_CAPACITY: Final[int] = 3

ICU_BEDS_FREE_HARD_START: Final[int] = 2
GENERAL_BEDS_FREE_HARD_START: Final[int] = 4

ICU_NURSE_RATIO_NUMERATOR: Final[int] = 1
ICU_NURSE_RATIO_DENOMINATOR: Final[int] = 2

ED_NURSE_RATIO_NUMERATOR: Final[int] = 1
ED_NURSE_RATIO_DENOMINATOR: Final[int] = 3

SAFE_WAIT_LIMITS: Final[dict[int, int]] = {
	1: 0,
	2: 1,
	3: 3,
	4: 6,
	5: 12,
}

DETERIORATION_RULES: Final[dict[int, tuple[float, int]]] = {
	3: (0.08, 2),
	4: (0.02, 3),
	5: (0.005, 4),
}

ARRIVAL_BASE_RATE: Final[float] = 0.4
HARD_START_HOUR: Final[int] = 17

HOURLY_ARRIVAL_MULTIPLIERS: Final[dict[int, float]] = {
	0: 0.30, 1: 0.20, 2: 0.15, 3: 0.15, 4: 0.20,
	5: 0.30, 6: 0.50, 7: 0.70, 8: 0.90, 9: 1.20,
	10: 1.40, 11: 1.50, 12: 1.40, 13: 1.30, 14: 1.20,
	15: 1.40, 16: 1.60, 17: 1.70, 18: 1.80, 19: 1.90,
	20: 1.70, 21: 1.40, 22: 1.00, 23: 0.60,
}

TASK_STEPS: Final[dict[str, int]] = {
	"easy": 5,
	"medium": 12,
	"hard": 20,
}
HARD_DEATH_CAP: Final[int] = 3

ESI_GRADE_EXACT: Final[float] = 0.50
ESI_GRADE_DIFF_1: Final[float] = 0.25
ESI_GRADE_DIFF_2: Final[float] = 0.10
ESI_GRADE_ESI1_TO_45: Final[float] = -0.50
ESI_GRADE_ESI2_TO_5: Final[float] = -0.30

BED_GRADE_ICU_CORRECT: Final[float] = 0.15
BED_GRADE_GENERAL_CORRECT: Final[float] = 0.10
BED_GRADE_ESI1_NO_BED: Final[float] = -0.50
BED_GRADE_ESI45_TO_ICU: Final[float] = -0.10

PROTOCOL_BONUS_CORRECT: Final[float] = 0.20
PROTOCOL_BONUS_WRONG: Final[float] = -0.30

DETERIORATION_EVENT_PENALTY: Final[float] = -0.15
ESI12_DELAY_PENALTY_PER_STEP: Final[float] = -0.10

OUTCOME_BASE_PROB: Final[float] = 0.75
OUTCOME_ESI_ERROR_FACTOR: Final[float] = -0.15
OUTCOME_WAIT_FACTOR: Final[float] = -0.05
OUTCOME_DETERIORATION_FACTOR: Final[float] = -0.10
OUTCOME_WRONG_BED_FACTOR: Final[float] = -0.10

OUTCOME_RECOVER_REWARD: Final[float] = 0.15
OUTCOME_STABLE_REWARD: Final[float] = 0.05
OUTCOME_CRITICAL_REWARD: Final[float] = -0.35
OUTCOME_DEATH_REWARD: Final[float] = -0.60

OUTCOME_DELAY_BY_ESI: Final[dict[int, int]] = {
	1: 3,
	2: 4,
	3: 6,
	4: 8,
	5: 10,
}

STAFF_OVERLOAD_PENALTY_FACTOR: Final[float] = -0.05

REWARD_CLAMP_MIN: Final[float] = -1.0
REWARD_CLAMP_MAX: Final[float] = 1.0

DEFAULT_RANDOM_SEED: Final[int] = 42
