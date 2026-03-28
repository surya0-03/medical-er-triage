from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Literal

from .data.constants import (
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
    HOURLY_ARRIVAL_MULTIPLIERS,
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
)
from ..models import Action, Observation, Patient, Reward, calculate_staff_load
from .data.patient_dataset import PatientRecord, get_patient_dataset
from .data.protocols import PROTOCOL_DEFINITIONS, ProtocolType


Difficulty = Literal["easy", "medium", "hard"]
BedType = Literal["icu", "general", "hallway", "none"]


@dataclass
class RuntimePatient:
    patient: Patient
    true_esi: int
    last_assigned_esi: int | None
    bed_type: BedType
    deterioration_events: int
    wrong_bed_events: int


@dataclass
class BedAssignment:
    patient_id: str
    bed_type: BedType
    remaining_steps: int


@dataclass
class DelayedOutcome:
    patient_id: str
    due_step: int
    probability: float


class MedicalEmergencyRoomEnv:
    def __init__(
        self,
        difficulty: Difficulty = "medium",
        seed: int = DEFAULT_RANDOM_SEED,
        *,
        debug: bool = False,
        outcome_smoothing: float = 0.30,
    ) -> None:
        if difficulty not in TASK_STEPS:
            raise ValueError("difficulty must be one of 'easy', 'medium', 'hard'")

        self._difficulty: Difficulty = difficulty
        self._base_seed = seed
        self._debug = debug
        self.rng = random.Random(seed)

        dataset = get_patient_dataset()
        self._dataset: list[PatientRecord] = dataset if dataset else [self._default_patient_record()]
        self._patient_counter = 0

        self._patients: dict[str, RuntimePatient] = {}
        self._beds: list[BedAssignment] = []
        self._delayed_outcomes: list[DelayedOutcome] = []

        self._time_step = 0
        self._icu_beds_available = ICU_BEDS_TOTAL
        self._general_beds_available = GENERAL_BEDS_TOTAL
        self._hallway_used = 0
        self._waiting_queue: list[str] = []

        self._icu_nurses = 4
        self._ed_nurses = 8

        self._hard_esi1_deaths = 0
        self._esi1_total_seen = 0

        self._step_reward_sum = 0.0
        self._step_reward_count = 0
        self._outcome_rewards_accum = 0.0
        self._outcome_smoothing = max(0.0, min(1.0, outcome_smoothing))
        self._outcome_reward_cap = 0.45
        self._previous_crowding_score = 0.0

        self._protocol_last_trigger_step: dict[ProtocolType, int] = {
            protocol: -10_000 for protocol in PROTOCOL_DEFINITIONS
        }
        self._protocol_last_trigger_step_by_patient: dict[str, dict[ProtocolType, int]] = {}
        self._resources: dict[str, bool] = {
            "ct_scanner": True,
            "cath_lab_team": True,
            "broad_spectrum_antibiotics": True,
            "massive_transfusion_pack": True,
        }
        self._resource_busy_until: dict[str, int] = {
            resource: -10_000 for resource in self._resources
        }

        self._ineffective_action_penalty = -0.03
        self._ineffective_penalty_cap = -0.05
        self._invalid_discharge_penalty = -0.03
        self._max_wait_penalty_per_patient = -0.18
        self._hallway_penalty_per_patient = -0.04
        self._max_system_penalty = -0.90

        self._max_waiting_queue = 120
        self._max_active_patients = 300
        self._max_delayed_outcomes = 400
        self._max_patients_total = self._max_active_patients + self._max_delayed_outcomes
        self._max_wait_time = 60
        self._max_deterioration_events_per_patient = 5

        self._sustained_overload_steps = 0
        self._extreme_crowding_steps = 0
        self._contracts_enabled = debug
        self._episode_done = False

        self._last_observation: Observation | None = None

    def reset(self, seed: int | None = None) -> Observation:
        if seed is None:
            seed = self._base_seed

        self._base_seed = seed
        self.rng = random.Random(seed)

        self._patient_counter = 0

        self._patients.clear()
        self._beds.clear()
        self._delayed_outcomes.clear()

        self._time_step = 0
        self._waiting_queue = []
        self._hard_esi1_deaths = 0
        self._esi1_total_seen = 0

        self._step_reward_sum = 0.0
        self._step_reward_count = 0
        self._outcome_rewards_accum = 0.0
        self._previous_crowding_score = 0.0

        self._sustained_overload_steps = 0
        self._extreme_crowding_steps = 0
        self._episode_done = False

        self._protocol_last_trigger_step = {
            protocol: -10_000 for protocol in PROTOCOL_DEFINITIONS
        }
        self._protocol_last_trigger_step_by_patient = {}
        self._resources = {
            "ct_scanner": True,
            "cath_lab_team": True,
            "broad_spectrum_antibiotics": True,
            "massive_transfusion_pack": True,
        }
        self._resource_busy_until = {
            resource: -10_000 for resource in self._resources
        }

        if self._difficulty == "hard":
            self._icu_beds_available = ICU_BEDS_FREE_HARD_START
            self._general_beds_available = GENERAL_BEDS_FREE_HARD_START
        else:
            self._icu_beds_available = ICU_BEDS_TOTAL
            self._general_beds_available = GENERAL_BEDS_TOTAL
        self._hallway_used = 0

        initial_arrivals = 5 if self._difficulty == "hard" else 2 if self._difficulty == "medium" else 1
        for _ in range(initial_arrivals):
            self._enqueue_new_patient()

        self._repair_state()
        self._safe_validate_state_consistency()
        self._assert_contracts("post_reset")
        observation = self._build_observation()
        self._last_observation = observation
        self._previous_crowding_score = observation.crowding_score
        return observation

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, object]]:
        if self._last_observation is None:
            self._warn("step called before reset; auto-resetting with base seed")
            self.reset(seed=self._base_seed)

        if self._episode_done:
            observation = self._last_observation if self._last_observation is not None else self._build_observation()
            reward = Reward(value=0.0, breakdown={"post_done_step": 0.0})
            info: dict[str, object] = {
                "time_step": self._time_step,
                "difficulty": self._difficulty,
                "terminated": True,
                "truncated": False,
                "requires_reset": True,
            }
            return observation, reward, True, info

        self._assert_contracts("pre_step")
        action_breakdown = self._apply_action(action)
        self._assert_contracts("post_action")
        self._update_beds()
        self._assert_contracts("post_bed_update")
        delay_penalty = self._increment_wait_times()
        deterioration_penalty, deterioration_events = self._apply_deterioration()
        self._assert_contracts("post_deterioration")
        new_arrivals = self._process_arrivals()
        self._assert_contracts("post_arrivals")

        outcome_reward_delta = self._update_outcomes()
        self._assert_contracts("post_outcomes")
        self._outcome_rewards_accum = (
            (1.0 - self._outcome_smoothing) * self._outcome_rewards_accum
            + self._outcome_smoothing * outcome_reward_delta
        )

        staffing_penalty, overload = self._compute_staffing_penalty()
        crowding = self._calculate_crowding_score()
        crowding_penalty = -0.05 * crowding

        hallway_penalty = self._hallway_penalty_per_patient * float(self._hallway_used)

        stabilization_bonus = 0.0
        crowding_reduction = self._previous_crowding_score - crowding
        if crowding_reduction > 0.0 and action.action_type in {"allocate_bed", "discharge"}:
            raw_stabilization_bonus = min(0.02, 0.01 * crowding_reduction)
            stabilization_bonus = min(raw_stabilization_bonus, 0.4 * abs(crowding_penalty))

        immediate_reward = (
            action_breakdown["esi_grade"]
            + action_breakdown["bed_grade"]
            + action_breakdown["protocol_bonus"]
            + action_breakdown["ineffective_penalty"]
            + action_breakdown["discharge_penalty"]
        )

        raw_system_penalty = (
            deterioration_penalty
            + delay_penalty
            + staffing_penalty
            + crowding_penalty
            + hallway_penalty
            + stabilization_bonus
        )
        system_penalty = max(self._max_system_penalty, raw_system_penalty)

        step_reward = immediate_reward + system_penalty
        bounded_step_reward = max(-1.0, min(1.0, step_reward))

        self._step_reward_sum += bounded_step_reward
        self._step_reward_count += 1

        average_step_rewards = self._step_reward_sum / float(self._step_reward_count)
        bounded_outcome_component = max(-self._outcome_reward_cap, min(self._outcome_reward_cap, self._outcome_rewards_accum))
        raw_total_reward = bounded_step_reward + bounded_outcome_component
        normalization_span = 1.0 + self._outcome_reward_cap
        normalized_score = (raw_total_reward + normalization_span) / (2.0 * normalization_span)
        final_value = self._clamp_reward(normalized_score)

        reward = Reward(
            value=final_value,
            breakdown={
                "step_reward": float(bounded_step_reward),
                "average_step_rewards": float(average_step_rewards),
                "immediate_reward": float(immediate_reward),
                "system_penalty": float(system_penalty),
                "raw_system_penalty": float(raw_system_penalty),
                "outcome_reward": float(bounded_outcome_component),
                "raw_total_reward": float(raw_total_reward),
                "normalized_score": float(normalized_score),
                "outcome_reward_delta": float(outcome_reward_delta),
                "outcome_rewards_accum": float(self._outcome_rewards_accum),
                "esi_grade": float(action_breakdown["esi_grade"]),
                "bed_grade": float(action_breakdown["bed_grade"]),
                "protocol_bonus": float(action_breakdown["protocol_bonus"]),
                "deterioration_penalty": float(deterioration_penalty),
                "delay_penalty": float(delay_penalty),
                "staffing_penalty": float(staffing_penalty),
                "crowding_penalty": float(crowding_penalty),
                "hallway_penalty": float(hallway_penalty),
                "stabilization_bonus": float(stabilization_bonus),
                "ineffective_penalty": float(action_breakdown["ineffective_penalty"]),
                "discharge_penalty": float(action_breakdown["discharge_penalty"]),
                "deterioration_events": float(deterioration_events),
                "new_arrivals": float(new_arrivals),
                "staff_overload": float(overload),
            },
        )

        if overload > 1.25:
            self._sustained_overload_steps += 1
        else:
            self._sustained_overload_steps = 0

        if crowding > 3.5:
            self._extreme_crowding_steps += 1
        else:
            self._extreme_crowding_steps = 0

        self._time_step += 1
        self._repair_state()
        self._safe_validate_state_consistency()
        self._assert_contracts("post_step")

        observation = self._build_observation()
        self._last_observation = observation
        self._previous_crowding_score = observation.crowding_score

        terminated = self._is_terminated()
        truncated = False
        done = terminated or truncated
        self._episode_done = done

        info: dict[str, object] = {
            "time_step": self._time_step,
            "difficulty": self._difficulty,
            "hard_esi1_deaths": self._hard_esi1_deaths,
            "queue_length": len(self._waiting_queue),
            "beds_in_use": len(self._beds),
            "outcome_reward_delta": outcome_reward_delta,
            "action_type": action.action_type,
            "sustained_overload_steps": self._sustained_overload_steps,
            "extreme_crowding_steps": self._extreme_crowding_steps,
            "avg_wait_time": self._average_wait_time(),
            "mortality_rate": self._mortality_rate(),
            "terminated": terminated,
            "truncated": truncated,
            "actual_bed_allocated": action_breakdown.get("actual_bed_allocated", None),
        }

        return observation, reward, done, info

    def step_gym(self, action: Action) -> tuple[Observation, Reward, bool, bool, dict[str, object]]:
        observation, reward, done, info = self.step(action)
        terminated = bool(info.get("terminated", done))
        truncated = bool(info.get("truncated", False))
        return observation, reward, terminated, truncated, info

    def state(self) -> dict[str, object]:
        observation = self._last_observation if self._last_observation is not None else self._build_observation()
        average_step_reward = self._step_reward_sum / float(self._step_reward_count) if self._step_reward_count > 0 else 0.0
        return {
            "time_step": self._time_step,
            "difficulty": self._difficulty,
            "seed": self._base_seed,
            "observation": observation.model_dump(mode="python"),
            "patients_total": len(self._patients),
            "waiting_queue_count": len(self._waiting_queue),
            "beds_in_use": len(self._beds),
            "delayed_outcomes_pending": len(self._delayed_outcomes),
            "hard_esi1_deaths": self._hard_esi1_deaths,
            "esi1_total_seen": self._esi1_total_seen,
            "average_step_reward": average_step_reward,
            "outcome_rewards_accum": self._outcome_rewards_accum,
            "done": self._episode_done,
        }

    def get_action_mask(self) -> dict[str, Any]:
        untriaged_waiting_ids = [
            pid
            for pid in self._waiting_queue
            if pid in self._patients and self._patients[pid].last_assigned_esi is None
        ]
        triaged_waiting_ids = [
            pid
            for pid in self._waiting_queue
            if pid in self._patients and self._patients[pid].last_assigned_esi is not None
        ]
        protocol_map: dict[str, list[str]] = {}

        for patient_id, runtime in self._patients.items():
            if runtime.patient.state not in {"assigned_bed", "in_treatment"}:
                continue
            valid_protocols: list[str] = []
            for protocol_type, definition in PROTOCOL_DEFINITIONS.items():
                last_step = self._protocol_last_trigger_step_by_patient.get(patient_id, {}).get(protocol_type, -10_000)
                cooldown_elapsed = self._time_step - last_step
                if cooldown_elapsed < definition.cooldown_steps:
                    continue
                valid_protocols.append(protocol_type)
            if valid_protocols:
                protocol_map[patient_id] = valid_protocols

        bed_mask: dict[str, list[str]] = {}
        for pid in triaged_waiting_ids:
            runtime = self._patients[pid]
            allowed: list[str] = []
            if self._icu_beds_available > 0:
                allowed.append("icu")
            if self._general_beds_available > 0:
                allowed.append("general")
            if self._hallway_used < HALLWAY_CAPACITY:
                allowed.append("hallway")
            if runtime.true_esi == 1 and not allowed:
                allowed.append("none_available")
            bed_mask[pid] = allowed

        discharge_ids = [
            pid
            for pid in triaged_waiting_ids
            if self._patients[pid].last_assigned_esi is not None
            and self._patients[pid].last_assigned_esi >= 4
            and self._patients[pid].patient.wait_time
            >= SAFE_WAIT_LIMITS[self._patients[pid].last_assigned_esi]
        ]

        # Progression guard: when triaged patients have a valid next action,
        # suppress new triage recommendations to reduce action-looping behavior.
        assign_ids = untriaged_waiting_ids
        has_bed_progress = any(any(b in {"icu", "general", "hallway"} for b in beds) for beds in bed_mask.values())
        if has_bed_progress or discharge_ids:
            assign_ids = [
                pid
                for pid in untriaged_waiting_ids
                if pid in self._patients and self._patients[pid].patient.esi_level <= 2
            ]

        return {
            "assign_esi_patient_ids": assign_ids,
            "allocate_bed": bed_mask,
            "trigger_protocol": protocol_map,
            "discharge_patient_ids": discharge_ids,
            "divert_allowed": self._is_overloaded(),
        }

    @property
    def action_space(self) -> dict[str, Any]:
        return {
            "action_type": ["assign_esi", "allocate_bed", "discharge", "trigger_protocol", "divert"],
            "esi_level": [1, 2, 3, 4, 5],
            "bed_type": ["icu", "general", "hallway"],
            "protocol_type": list(PROTOCOL_DEFINITIONS.keys()),
        }

    @property
    def observation_space(self) -> dict[str, Any]:
        return {
            "time_step": "int>=0",
            "patients": "list[Patient]",
            "waiting_queue": "list[str]",
            "icu_beds_available": f"int in [0,{ICU_BEDS_TOTAL}]",
            "general_beds_available": f"int in [0,{GENERAL_BEDS_TOTAL}]",
            "hallway_used": f"int in [0,{HALLWAY_CAPACITY}]",
            "crowding_score": "float>=0",
            "staff_load": "dict",
        }

    def _apply_action(self, action: Action) -> dict[str, Any]:
        esi_grade = 0.0
        bed_grade = 0.0
        protocol_bonus = 0.0
        ineffective_penalty = 0.0
        discharge_penalty = 0.0
        actual_bed_allocated: str | None = None

        if action.action_type == "assign_esi":
            runtime = self._patients.get(action.patient_id or "")
            if (
                runtime is not None
                and runtime.patient.state == "waiting"
                and runtime.last_assigned_esi is None
                and action.esi_level is not None
            ):
                runtime.last_assigned_esi = action.esi_level
                esi_grade = self._grade_esi(action.esi_level, runtime.true_esi)
            else:
                ineffective_penalty += self._ineffective_action_penalty

        elif action.action_type == "allocate_bed":
            runtime = self._patients.get(action.patient_id or "")
            if (
                runtime is not None
                and runtime.patient.state == "waiting"
                and runtime.last_assigned_esi is not None
                and action.bed_type is not None
            ):
                requested_bed = action.bed_type
                actual_bed = self._select_bed_with_fallback(runtime, requested_bed)
                if actual_bed is not None:
                    allocated = self._allocate_bed(runtime, actual_bed)
                    if allocated:
                        actual_bed_allocated = actual_bed
                        bed_grade = self._grade_bed(runtime, actual_bed)
                    else:
                        actual_bed_allocated = "none"
                        if runtime.true_esi != 1:
                            ineffective_penalty += self._ineffective_action_penalty
                else:
                    actual_bed_allocated = "none"
                    if runtime.true_esi != 1:
                        ineffective_penalty += self._ineffective_action_penalty
            else:
                ineffective_penalty += self._ineffective_action_penalty

        elif action.action_type == "trigger_protocol":
            runtime = self._patients.get(action.patient_id or "")
            if runtime is not None and action.protocol_type is not None:
                protocol_bonus = self._trigger_protocol(runtime, action.protocol_type)
            else:
                ineffective_penalty += self._ineffective_action_penalty

        elif action.action_type == "discharge":
            runtime = self._patients.get(action.patient_id or "")
            if (
                runtime is not None
                and runtime.patient.state == "waiting"
                and runtime.last_assigned_esi is not None
                and runtime.last_assigned_esi >= 4
                and runtime.patient.wait_time >= SAFE_WAIT_LIMITS[runtime.last_assigned_esi]
            ):
                self._complete_patient(runtime.patient.patient_id)
            else:
                discharge_penalty += self._invalid_discharge_penalty

        elif action.action_type == "divert":
            if self._is_overloaded():
                divert_patient_id: str | None = None
                triaged_low_priority: list[RuntimePatient] = [
                    self._patients[pid]
                    for pid in self._waiting_queue
                    if pid in self._patients
                    and self._patients[pid].last_assigned_esi is not None
                    and self._patients[pid].last_assigned_esi >= 4
                ]
                if triaged_low_priority:
                    selected = max(
                        triaged_low_priority,
                        key=lambda runtime: (
                            int(runtime.last_assigned_esi or 0),
                            int(runtime.patient.wait_time),
                        ),
                    )
                    divert_patient_id = selected.patient.patient_id
                else:
                    untriaged: list[RuntimePatient] = [
                        self._patients[pid]
                        for pid in self._waiting_queue
                        if pid in self._patients and self._patients[pid].last_assigned_esi is None
                    ]
                    if untriaged:
                        selected = max(
                            untriaged,
                            key=lambda runtime: (
                                int(runtime.patient.esi_level),
                                int(runtime.patient.wait_time),
                            ),
                        )
                        divert_patient_id = selected.patient.patient_id

                if divert_patient_id is not None:
                    self._divert_patient(divert_patient_id)
                    esi_grade += 0.05
                else:
                    ineffective_penalty += self._ineffective_action_penalty
            else:
                ineffective_penalty += self._ineffective_action_penalty

        ineffective_penalty = max(self._ineffective_penalty_cap, ineffective_penalty)

        return {
            "esi_grade": esi_grade,
            "bed_grade": bed_grade,
            "protocol_bonus": protocol_bonus,
            "ineffective_penalty": ineffective_penalty,
            "discharge_penalty": discharge_penalty,
            "actual_bed_allocated": actual_bed_allocated,
        }

    def _select_bed_with_fallback(self, runtime: RuntimePatient, requested_bed: BedType) -> BedType | None:
        if requested_bed == "icu" and self._icu_beds_available > 0:
            return "icu"
        if requested_bed == "general" and self._general_beds_available > 0:
            return "general"
        if requested_bed == "hallway" and self._hallway_used < HALLWAY_CAPACITY:
            return "hallway"

        if runtime.true_esi == 1:
            if self._icu_beds_available > 0:
                return "icu"
            if self._general_beds_available > 0:
                return "general"
            if self._hallway_used < HALLWAY_CAPACITY:
                return "hallway"

        return None

    def _allocate_bed(self, runtime: RuntimePatient, bed_type: BedType) -> bool:
        patient = runtime.patient

        if bed_type == "icu":
            if self._icu_beds_available <= 0:
                return False
            self._icu_beds_available -= 1
        elif bed_type == "general":
            if self._general_beds_available <= 0:
                return False
            self._general_beds_available -= 1
        elif bed_type == "hallway":
            if self._hallway_used >= HALLWAY_CAPACITY:
                return False
            self._hallway_used += 1
        else:
            return False

        if patient.patient_id in self._waiting_queue:
            self._waiting_queue.remove(patient.patient_id)

        patient.state = "assigned_bed"
        runtime.bed_type = bed_type

        remaining_steps = self._treatment_duration_steps(runtime.true_esi)
        self._beds.append(BedAssignment(patient_id=patient.patient_id, bed_type=bed_type, remaining_steps=remaining_steps))
        return True

    def _trigger_protocol(self, runtime: RuntimePatient, protocol_type: ProtocolType) -> float:
        patient = runtime.patient
        definition = PROTOCOL_DEFINITIONS.get(protocol_type)
        if definition is None:
            return PROTOCOL_BONUS_WRONG

        patient_id = patient.patient_id
        last_step = self._protocol_last_trigger_step_by_patient.get(patient_id, {}).get(protocol_type, -10_000)
        cooldown_elapsed = self._time_step - last_step
        if cooldown_elapsed < definition.cooldown_steps:
            return PROTOCOL_BONUS_WRONG

        if patient.state not in {"assigned_bed", "in_treatment"}:
            return PROTOCOL_BONUS_WRONG

        matches = len(set(patient.symptoms).intersection(set(definition.required_symptoms)))
        has_minimum_symptoms = matches >= 2
        correct_esi = runtime.last_assigned_esi is not None and runtime.last_assigned_esi == runtime.true_esi
        meets_esi_requirement = runtime.true_esi <= definition.minimum_esi_requirement
        correct_bed = runtime.bed_type == definition.required_bed_type
        resource_name = definition.required_resource
        resource_available = self._resources.get(resource_name, False) and self._time_step >= self._resource_busy_until.get(resource_name, -10_000)
        if not resource_available:
            return PROTOCOL_BONUS_WRONG

        self._protocol_last_trigger_step_by_patient.setdefault(patient_id, {})[protocol_type] = self._time_step

        if has_minimum_symptoms and correct_esi and meets_esi_requirement and correct_bed:
            self._resource_busy_until[resource_name] = self._time_step + definition.cooldown_steps
            return PROTOCOL_BONUS_CORRECT

        return PROTOCOL_BONUS_WRONG

    def _update_beds(self) -> None:
        active: list[BedAssignment] = []

        for assignment in self._beds:
            runtime = self._patients.get(assignment.patient_id)
            if runtime is None:
                self._release_bed_resources(assignment.bed_type)
                continue

            if runtime.patient.state == "assigned_bed":
                runtime.patient.state = "in_treatment"

            assignment.remaining_steps = max(0, assignment.remaining_steps - 1)

            if assignment.remaining_steps == 0:
                runtime.patient.state = "completed"
                probability = self._outcome_probability(runtime)
                # Use true_esi to prevent deterioration side-effects from changing outcome delay timing.
                delay_steps = OUTCOME_DELAY_BY_ESI[runtime.true_esi]
                self._delayed_outcomes.append(
                    DelayedOutcome(
                        patient_id=runtime.patient.patient_id,
                        due_step=self._time_step + delay_steps,
                        probability=probability,
                    )
                )
                runtime.patient.state = "outcome_pending"
                self._release_bed_resources(assignment.bed_type)
            else:
                active.append(assignment)

        self._beds = active

        if len(self._delayed_outcomes) > self._max_delayed_outcomes:
            self._delayed_outcomes.sort(key=lambda item: item.due_step)
            dropped = self._delayed_outcomes[self._max_delayed_outcomes :]
            self._delayed_outcomes = self._delayed_outcomes[: self._max_delayed_outcomes]
            for item in dropped:
                self._warn("delayed outcome cap reached; force-removing overflow patient to prevent state leak")
                self._finalize_patient_removal(item.patient_id)

    def _increment_wait_times(self) -> float:
        penalty = 0.0
        for patient_id in self._waiting_queue:
            runtime = self._patients.get(patient_id)
            if runtime is None:
                continue

            runtime.patient.wait_time = min(self._max_wait_time, runtime.patient.wait_time + 1)
            safe_limit = SAFE_WAIT_LIMITS[runtime.patient.esi_level]
            overdue = max(0, runtime.patient.wait_time - safe_limit)
            if overdue <= 0:
                continue

            smooth_factor = min(1.0, float(overdue) / 4.0)
            patient_penalty = ESI12_DELAY_PENALTY_PER_STEP * smooth_factor

            if runtime.patient.esi_level == 1:
                # ESI-1 receives stronger urgency penalty, but capped to avoid double-penalty explosions.
                patient_penalty *= 1.5

            patient_penalty = max(self._max_wait_penalty_per_patient, patient_penalty)
            penalty += patient_penalty

        return penalty

    def _apply_deterioration(self) -> tuple[float, int]:
        events = 0

        for patient_id in list(self._waiting_queue):
            runtime = self._patients.get(patient_id)
            if runtime is None:
                continue

            if runtime.deterioration_events >= self._max_deterioration_events_per_patient:
                continue

            esi = runtime.patient.esi_level
            safe_limit = SAFE_WAIT_LIMITS[esi]
            if runtime.patient.wait_time <= safe_limit:
                continue
            if esi not in DETERIORATION_RULES:
                continue

            probability, target_esi = DETERIORATION_RULES[esi]
            decay = max(0.25, 1.0 - 0.1 * float(runtime.deterioration_events))
            adjusted_probability = probability * decay

            if self.rng.random() < adjusted_probability:
                runtime.patient.esi_level = target_esi
                runtime.patient.deterioration_count += 1
                runtime.deterioration_events += 1
                events += 1

        penalty = DETERIORATION_EVENT_PENALTY * float(events)
        return penalty, events

    def _process_arrivals(self) -> int:
        hour_of_day = ((self._time_step * TIME_STEP_MINUTES) // 60) % 24
        hourly_multiplier = self._hourly_multiplier(hour_of_day)
        rate = ARRIVAL_BASE_RATE * hourly_multiplier
        new_count = self._sample_poisson(rate)

        accepted = 0
        for _ in range(new_count):
            patient_id = self._enqueue_new_patient()
            if patient_id is not None:
                accepted += 1
            else:
                break

        return accepted

    def _sample_poisson(self, rate: float) -> int:
        if rate <= 0.0:
            return 0

        # Fast normal approximation for high-rate regime.
        if rate >= 10.0:
            sample = int(round(self.rng.gauss(rate, math.sqrt(rate))))
            return max(0, sample)

        threshold = math.exp(-rate)
        k = 0
        p = 1.0
        max_loops = 256
        while p > threshold and k < max_loops:
            k += 1
            p *= self.rng.random()

        if k >= max_loops:
            self._warn("poisson loop cap reached; using capped result")
            return max(0, int(rate))

        return max(0, k - 1)

    def _hourly_multiplier(self, hour_of_day: int) -> float:
        base = HOURLY_ARRIVAL_MULTIPLIERS.get(hour_of_day, 1.0)
        if self._difficulty == "hard":
            return base * 1.5
        if self._difficulty == "medium":
            return base * 1.2
        return base

    def _enqueue_new_patient(self) -> str | None:
        if not self._can_accept_new_patient():
            return None

        if not self._dataset:
            self._dataset = [self._default_patient_record()]

        record = self.rng.choice(self._dataset)

        patient_id = f"p_{self._patient_counter:05d}"
        self._patient_counter += 1

        vitals_data = record.get("vitals", {})
        true_esi = int(record.get("true_esi", 3))
        true_esi = true_esi if 1 <= true_esi <= 5 else 3
        if true_esi == 1:
            self._esi1_total_seen += 1

        symptoms = list(record.get("symptoms", ["general discomfort"]))
        if not symptoms:
            symptoms = ["general discomfort"]

        if len(self._patients) >= self._max_patients_total:
            return None

        patient = Patient(
            patient_id=patient_id,
            age=max(0, int(record.get("age", 40))),
            symptoms=symptoms,
            vitals={
                "heart_rate": max(1, int(vitals_data.get("heart_rate", 80))),
                "blood_pressure": str(vitals_data.get("blood_pressure", "120/80")),
                "oxygen_level": max(0.0, min(100.0, float(vitals_data.get("oxygen_level", 98.0)))),
            },
            esi_level=true_esi,
            wait_time=0,
            state="waiting",
            deterioration_count=0,
        )

        self._patients[patient_id] = RuntimePatient(
            patient=patient,
            true_esi=true_esi,
            last_assigned_esi=None,
            bed_type="none",
            deterioration_events=0,
            wrong_bed_events=0,
        )
        self._waiting_queue.append(patient_id)
        return patient_id

    def _compute_staffing_penalty(self) -> tuple[float, float]:
        staff = calculate_staff_load(
            patients=[runtime.patient for runtime in self._patients.values()],
            icu_nurses=self._icu_nurses,
            ed_nurses=self._ed_nurses,
        )

        overload_icu = max(0.0, float(staff["icu_ratio_load"]) - 1.0)
        overload_ed = max(0.0, float(staff["ed_ratio_load"]) - 1.0)
        overload = overload_icu + overload_ed

        penalty = STAFF_OVERLOAD_PENALTY_FACTOR * overload
        return penalty, overload

    def _calculate_crowding_score(self) -> float:
        queue_component = float(len(self._waiting_queue)) / 10.0
        hallway_component = float(self._hallway_used) / float(HALLWAY_CAPACITY)
        occupied = (ICU_BEDS_TOTAL - self._icu_beds_available) + (GENERAL_BEDS_TOTAL - self._general_beds_available)
        occupied_component = float(occupied) / float(ICU_BEDS_TOTAL + GENERAL_BEDS_TOTAL)
        return queue_component + hallway_component + occupied_component

    def _is_overloaded(self) -> bool:
        return self._calculate_crowding_score() > 1.0 or self._hallway_used >= HALLWAY_CAPACITY

    def _grade_esi(self, assigned_esi: int, true_esi: int) -> float:
        diff = abs(assigned_esi - true_esi)
        if diff == 0:
            raw_grade = ESI_GRADE_EXACT
        elif diff == 1:
            raw_grade = ESI_GRADE_DIFF_1
        elif diff == 2:
            raw_grade = ESI_GRADE_DIFF_2
        else:
            raw_grade = 0.0

        if true_esi == 1 and assigned_esi in {4, 5}:
            raw_grade += ESI_GRADE_ESI1_TO_45
        if true_esi == 1 and assigned_esi == 3:
            raw_grade += ESI_GRADE_ESI1_TO_45
        if true_esi == 2 and assigned_esi == 5:
            raw_grade += ESI_GRADE_ESI2_TO_5
        if true_esi == 2 and assigned_esi in {3, 4}:
            raw_grade += ESI_GRADE_ESI2_TO_5

        return raw_grade

    def _grade_bed(self, runtime: RuntimePatient, bed_type: BedType) -> float:
        true_esi = runtime.true_esi
        grade = 0.0

        correct_bed = False
        if true_esi <= 2 and bed_type == "icu":
            grade += BED_GRADE_ICU_CORRECT
            correct_bed = True
        elif true_esi >= 3 and bed_type == "general":
            grade += BED_GRADE_GENERAL_CORRECT
            correct_bed = True

        if true_esi == 1 and bed_type == "none":
            grade += BED_GRADE_ESI1_NO_BED

        wrong_bed = False
        if true_esi in {4, 5} and bed_type == "icu":
            grade += BED_GRADE_ESI45_TO_ICU
            wrong_bed = True
        if true_esi <= 2 and bed_type in {"general", "hallway"}:
            wrong_bed = True
        if true_esi <= 2 and bed_type == "hallway":
            grade += BED_GRADE_ESI1_NO_BED
        if true_esi >= 3 and bed_type == "hallway":
            wrong_bed = True
            grade += BED_GRADE_ESI45_TO_ICU

        if wrong_bed:
            runtime.wrong_bed_events = min(runtime.wrong_bed_events + 1, 10)
        elif correct_bed:
            runtime.wrong_bed_events = 0

        return grade

    def _outcome_probability(self, runtime: RuntimePatient) -> float:
        assigned_esi = runtime.last_assigned_esi if runtime.last_assigned_esi is not None else runtime.true_esi
        esi_error = abs(assigned_esi - runtime.true_esi)

        safe_limit = SAFE_WAIT_LIMITS[runtime.patient.esi_level]
        wait_time_over_limit = max(0, runtime.patient.wait_time - safe_limit)

        capped_esi_error = min(esi_error, 4)
        capped_wait = min(wait_time_over_limit, 10)
        capped_deterioration = min(runtime.deterioration_events, self._max_deterioration_events_per_patient)
        wrong_bed_flag = 1 if runtime.wrong_bed_events > 0 else 0

        probability = OUTCOME_BASE_PROB
        probability += OUTCOME_ESI_ERROR_FACTOR * float(capped_esi_error)
        probability += OUTCOME_WAIT_FACTOR * float(capped_wait)
        probability += OUTCOME_DETERIORATION_FACTOR * float(capped_deterioration)
        probability += OUTCOME_WRONG_BED_FACTOR * float(wrong_bed_flag)

        # Hallway placements are a degraded care environment.
        if runtime.bed_type == "hallway":
            probability -= 0.08

        return max(0.0, min(1.0, probability))

    def _update_outcomes(self) -> float:
        reward_delta = 0.0
        pending: list[DelayedOutcome] = []
        resolved_ids: list[str] = []

        for item in self._delayed_outcomes:
            if item.due_step <= self._time_step:
                runtime = self._patients.get(item.patient_id)
                if runtime is None:
                    self._warn("orphan delayed outcome encountered; dropping outcome entry")
                    continue

                if item.probability > 0.65:
                    outcome_reward = OUTCOME_RECOVER_REWARD
                elif item.probability >= 0.45:
                    outcome_reward = OUTCOME_STABLE_REWARD
                elif item.probability > 0.25:
                    outcome_reward = OUTCOME_CRITICAL_REWARD
                else:
                    outcome_reward = OUTCOME_DEATH_REWARD
                    if runtime.true_esi == 1:
                        self._hard_esi1_deaths += 1

                runtime.patient.state = "resolved"
                reward_delta += outcome_reward
                resolved_ids.append(item.patient_id)
            else:
                pending.append(item)

        self._delayed_outcomes = pending

        for patient_id in resolved_ids:
            self._finalize_patient_removal(patient_id)

        return reward_delta

    def _treatment_duration_steps(self, esi_level: int) -> int:
        if esi_level == 1:
            base = 5
        elif esi_level == 2:
            base = 4
        elif esi_level == 3:
            base = 3
        elif esi_level == 4:
            base = 2
        else:
            base = 1

        noise = self.rng.randint(-1, 1)
        return max(1, base + noise)

    def _complete_patient(self, patient_id: str) -> None:
        runtime = self._patients.get(patient_id)
        if runtime is None:
            return

        runtime.patient.state = "resolved"
        self._remove_patient_from_all_containers(patient_id, release_resources=True)

    def _divert_patient(self, patient_id: str) -> None:
        runtime = self._patients.get(patient_id)
        if runtime is None:
            return

        runtime.patient.state = "completed"
        self._remove_patient_from_all_containers(patient_id, release_resources=True)

    def _remove_patient_from_all_containers(self, patient_id: str, release_resources: bool) -> None:
        self._waiting_queue = [pid for pid in self._waiting_queue if pid != patient_id]

        active_beds: list[BedAssignment] = []
        for assignment in self._beds:
            if assignment.patient_id != patient_id:
                active_beds.append(assignment)
                continue

            if release_resources:
                self._release_bed_resources(assignment.bed_type)
            else:
                self._warn("patient appeared in bed container during outcome resolution; releasing resources safely")
                self._release_bed_resources(assignment.bed_type)
        self._beds = active_beds

        self._delayed_outcomes = [item for item in self._delayed_outcomes if item.patient_id != patient_id]
        self._protocol_last_trigger_step_by_patient.pop(patient_id, None)
        self._patients.pop(patient_id, None)

    def _finalize_patient_removal(self, patient_id: str) -> None:
        in_waiting = patient_id in self._waiting_queue
        in_beds = any(item.patient_id == patient_id for item in self._beds)
        in_outcomes = any(item.patient_id == patient_id for item in self._delayed_outcomes)

        if in_waiting or in_beds or in_outcomes:
            self._warn("patient removal inconsistency detected; repairing references before deletion")
            self._remove_patient_from_all_containers(patient_id, release_resources=True)
        else:
            self._patients.pop(patient_id, None)

    def _repair_state(self) -> None:
        valid_ids = set(self._patients.keys())

        dedup_waiting: list[str] = []
        seen_waiting: set[str] = set()
        for patient_id in self._waiting_queue:
            if patient_id not in valid_ids:
                continue
            if patient_id in seen_waiting:
                continue
            dedup_waiting.append(patient_id)
            seen_waiting.add(patient_id)
        self._waiting_queue = dedup_waiting

        valid_beds: list[BedAssignment] = []
        for assignment in self._beds:
            runtime = self._patients.get(assignment.patient_id)
            if runtime is None:
                self._release_bed_resources(assignment.bed_type)
                continue
            runtime.bed_type = assignment.bed_type
            if runtime.patient.state == "waiting":
                runtime.patient.state = "assigned_bed"
                if assignment.patient_id in self._waiting_queue:
                    self._waiting_queue.remove(assignment.patient_id)
            valid_beds.append(assignment)
        self._beds = valid_beds

        valid_outcomes: list[DelayedOutcome] = []
        for item in self._delayed_outcomes:
            runtime = self._patients.get(item.patient_id)
            if runtime is None:
                continue
            if runtime.patient.state not in {"outcome_pending", "resolved"}:
                runtime.patient.state = "outcome_pending"
            valid_outcomes.append(item)
        self._delayed_outcomes = valid_outcomes

        outcome_ids = {item.patient_id for item in self._delayed_outcomes}
        orphan_outcome_pending = [
            patient_id
            for patient_id, runtime in self._patients.items()
            if runtime.patient.state == "outcome_pending" and patient_id not in outcome_ids
        ]
        for patient_id in orphan_outcome_pending:
            self._warn("orphan outcome_pending patient detected; finalizing removal")
            self._finalize_patient_removal(patient_id)

        self._icu_beds_available = ICU_BEDS_TOTAL - sum(1 for bed in self._beds if bed.bed_type == "icu")
        self._general_beds_available = GENERAL_BEDS_TOTAL - sum(1 for bed in self._beds if bed.bed_type == "general")
        self._hallway_used = sum(1 for bed in self._beds if bed.bed_type == "hallway")

        self._icu_beds_available = max(0, min(ICU_BEDS_TOTAL, self._icu_beds_available))
        self._general_beds_available = max(0, min(GENERAL_BEDS_TOTAL, self._general_beds_available))
        self._hallway_used = max(0, min(HALLWAY_CAPACITY, self._hallway_used))

        if len(self._patients) > self._max_patients_total:
            self._warn("patient container exceeded cap; trimming oldest IDs deterministically")
            sorted_ids = sorted(self._patients.keys())
            for patient_id in sorted_ids[self._max_patients_total :]:
                self._remove_patient_from_all_containers(patient_id, release_resources=True)

    def _safe_validate_state_consistency(self) -> None:
        try:
            self._validate_state_consistency()
        except ValueError as exc:
            self._warn(f"state consistency violation repaired: {exc}")
            self._repair_state()
            try:
                self._validate_state_consistency()
            except ValueError as second_exc:
                self._warn(f"state remains inconsistent after repair: {second_exc}")

    def _release_bed_resources(self, bed_type: BedType) -> None:
        if bed_type == "icu":
            self._icu_beds_available = min(ICU_BEDS_TOTAL, self._icu_beds_available + 1)
        elif bed_type == "general":
            self._general_beds_available = min(GENERAL_BEDS_TOTAL, self._general_beds_available + 1)
        elif bed_type == "hallway":
            self._hallway_used = max(0, self._hallway_used - 1)

    def _validate_state_consistency(self) -> None:
        waiting_ids = set(self._waiting_queue)
        bed_ids = {bed.patient_id for bed in self._beds}
        pending_ids = {item.patient_id for item in self._delayed_outcomes}

        if len(waiting_ids) != len(self._waiting_queue):
            raise ValueError("duplicate patient_id in waiting queue")

        overlap = (waiting_ids & bed_ids) | (waiting_ids & pending_ids) | (bed_ids & pending_ids)
        if overlap:
            raise ValueError("patient exists in multiple state containers")

        for patient_id in waiting_ids:
            runtime = self._patients.get(patient_id)
            if runtime is None:
                raise ValueError("waiting queue references invalid patient_id")
            if runtime.patient.state != "waiting":
                raise ValueError("waiting queue patient must be in waiting state")

        for patient_id in bed_ids:
            runtime = self._patients.get(patient_id)
            if runtime is None:
                raise ValueError("bed references invalid patient_id")
            if runtime.patient.state not in {"assigned_bed", "in_treatment"}:
                raise ValueError("bed-assigned patient must be assigned_bed or in_treatment")

        for patient_id in pending_ids:
            runtime = self._patients.get(patient_id)
            if runtime is None:
                raise ValueError("outcome queue references invalid patient_id")
            if runtime.patient.state != "outcome_pending":
                raise ValueError("outcome queue patient must be in outcome_pending state")

        for patient_id, runtime in self._patients.items():
            patient_state = runtime.patient.state
            in_waiting = patient_id in waiting_ids
            in_beds = patient_id in bed_ids
            in_outcomes = patient_id in pending_ids

            if patient_state == "waiting":
                if int(in_waiting) + int(in_beds) + int(in_outcomes) != 1 or not in_waiting:
                    raise ValueError("waiting-state patient must exist only in waiting queue")
            elif patient_state in {"assigned_bed", "in_treatment"}:
                if int(in_waiting) + int(in_beds) + int(in_outcomes) != 1 or not in_beds:
                    raise ValueError("bed-state patient must exist only in beds")
            elif patient_state == "outcome_pending":
                if int(in_waiting) + int(in_beds) + int(in_outcomes) != 1 or not in_outcomes:
                    raise ValueError("outcome-pending patient must exist only in delayed outcomes")
            elif patient_state in {"resolved", "completed"}:
                if in_waiting or in_beds or in_outcomes:
                    raise ValueError("resolved/completed patient cannot exist in active containers")
            else:
                raise ValueError("unknown patient state")

        if self._icu_beds_available < 0 or self._icu_beds_available > ICU_BEDS_TOTAL:
            raise ValueError("ICU bed availability is out of bounds")
        if self._general_beds_available < 0 or self._general_beds_available > GENERAL_BEDS_TOTAL:
            raise ValueError("general bed availability is out of bounds")
        if self._hallway_used < 0 or self._hallway_used > HALLWAY_CAPACITY:
            raise ValueError("hallway usage is out of bounds")

        icu_in_use = sum(1 for bed in self._beds if bed.bed_type == "icu")
        general_in_use = sum(1 for bed in self._beds if bed.bed_type == "general")
        hallway_in_use = sum(1 for bed in self._beds if bed.bed_type == "hallway")

        if self._icu_beds_available != ICU_BEDS_TOTAL - icu_in_use:
            raise ValueError("ICU availability does not match bed assignments")
        if self._general_beds_available != GENERAL_BEDS_TOTAL - general_in_use:
            raise ValueError("general availability does not match bed assignments")
        if self._hallway_used != hallway_in_use:
            raise ValueError("hallway usage does not match bed assignments")

    def _build_observation(self) -> Observation:
        visible_states = {"waiting", "assigned_bed", "in_treatment", "completed", "outcome_pending"}
        visible_patients: list[Patient] = []
        for runtime in self._patients.values():
            if runtime.patient.state not in visible_states:
                continue

            public_esi = runtime.last_assigned_esi if runtime.last_assigned_esi is not None else 3
            visible_patients.append(runtime.patient.model_copy(update={"esi_level": public_esi}))

        crowding_score = self._calculate_crowding_score()
        staff_load = calculate_staff_load(
            patients=visible_patients,
            icu_nurses=self._icu_nurses,
            ed_nurses=self._ed_nurses,
        )

        return Observation(
            time_step=self._time_step,
            patients=visible_patients,
            waiting_queue=list(self._waiting_queue),
            icu_beds_available=self._icu_beds_available,
            general_beds_available=self._general_beds_available,
            hallway_used=self._hallway_used,
            crowding_score=crowding_score,
            staff_load=staff_load,
        )

    def _is_terminated(self) -> bool:
        if self._difficulty == "hard":
            if self._time_step >= TASK_STEPS["hard"]:
                return True
            if self._hard_esi1_deaths >= HARD_DEATH_CAP:
                return True

        if self._time_step >= TASK_STEPS[self._difficulty]:
            return True

        if self._extreme_crowding_steps >= 3:
            return True

        if self._sustained_overload_steps >= 6:
            return True

        return False

    @staticmethod
    def _clamp_reward(value: float) -> float:
        return max(REWARD_CLAMP_MIN, min(REWARD_CLAMP_MAX, value))

    def _default_patient_record(self) -> PatientRecord:
        return {
            "symptoms": ["fatigue", "dizziness"],
            "vitals": {"heart_rate": 82, "blood_pressure": "120/80", "oxygen_level": 98.0},
            "age": 40,
            "true_esi": 3,
        }

    def _active_patient_count(self) -> int:
        active_states = {"waiting", "assigned_bed", "in_treatment", "outcome_pending"}
        return sum(1 for runtime in self._patients.values() if runtime.patient.state in active_states)

    def _can_accept_new_patient(self) -> bool:
        if len(self._waiting_queue) >= self._max_waiting_queue:
            return False
        if self._active_patient_count() >= self._max_active_patients:
            return False
        if len(self._delayed_outcomes) >= self._max_delayed_outcomes:
            return False
        if len(self._patients) >= self._max_patients_total:
            return False
        return True

    def _average_wait_time(self) -> float:
        waits = [runtime.patient.wait_time for runtime in self._patients.values() if runtime.patient.state == "waiting"]
        if not waits:
            return 0.0
        return float(sum(waits)) / float(len(waits))

    def _mortality_rate(self) -> float:
        if self._esi1_total_seen <= 0:
            return 0.0
        return float(self._hard_esi1_deaths) / float(self._esi1_total_seen)

    def _warn(self, message: str) -> None:
        if self._debug:
            print(f"[MedicalEmergencyRoomEnv][warning] {message}")

    def _assert_contracts(self, phase: str) -> None:
        if not self._contracts_enabled:
            return

        self._validate_state_consistency()

        if self._time_step < 0:
            raise AssertionError(f"[{phase}] time_step must be non-negative")
        if not (0 <= self._icu_beds_available <= ICU_BEDS_TOTAL):
            raise AssertionError(f"[{phase}] icu bed bounds violated")
        if not (0 <= self._general_beds_available <= GENERAL_BEDS_TOTAL):
            raise AssertionError(f"[{phase}] general bed bounds violated")
        if not (0 <= self._hallway_used <= HALLWAY_CAPACITY):
            raise AssertionError(f"[{phase}] hallway bounds violated")

        if len(self._patients) > self._max_patients_total:
            raise AssertionError(f"[{phase}] patients cap exceeded")
        if len(self._delayed_outcomes) > self._max_delayed_outcomes:
            raise AssertionError(f"[{phase}] delayed outcomes cap exceeded")
