from __future__ import annotations

import random

from hypothesis import HealthCheck, given, settings, strategies as st

from my_env.models import Action
from my_env.server.environment import MedicalEmergencyRoomEnv


def _choose_mask_action(env: MedicalEmergencyRoomEnv, rng: random.Random) -> Action:
    mask = env.get_action_mask()

    if mask["assign_esi_patient_ids"]:
        pid = rng.choice(mask["assign_esi_patient_ids"])
        return Action(action_type="assign_esi", patient_id=pid, esi_level=rng.randint(1, 5))

    alloc_items = list(mask["allocate_bed"].items())
    if alloc_items:
        pid, beds = rng.choice(alloc_items)
        valid_beds = [b for b in beds if b in {"icu", "general", "hallway"}]
        if valid_beds:
            return Action(action_type="allocate_bed", patient_id=pid, bed_type=rng.choice(valid_beds))

    protocol_items = list(mask["trigger_protocol"].items())
    if protocol_items:
        pid, protocols = rng.choice(protocol_items)
        if protocols:
            return Action(action_type="trigger_protocol", patient_id=pid, protocol_type=rng.choice(protocols))

    if mask["discharge_patient_ids"]:
        return Action(action_type="discharge", patient_id=rng.choice(mask["discharge_patient_ids"]))

    return Action(action_type="divert")


def _choose_semantically_invalid_action(env: MedicalEmergencyRoomEnv, rng: random.Random) -> Action:
    # Valid schema, intentionally wrong semantics (non-existent patient or invalid phase action).
    bogus_id = f"bogus_{rng.randint(0, 99999)}"
    choice = rng.randint(0, 2)
    if choice == 0:
        return Action(action_type="assign_esi", patient_id=bogus_id, esi_level=5)
    if choice == 1:
        return Action(action_type="allocate_bed", patient_id=bogus_id, bed_type="icu")
    return Action(action_type="trigger_protocol", patient_id=bogus_id, protocol_type="stroke_code")


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=10_000), steps=st.integers(min_value=30, max_value=220))
def test_invariants_hold_under_generated_trajectories(seed: int, steps: int) -> None:
    env = MedicalEmergencyRoomEnv(difficulty="hard", seed=seed, debug=False)
    env._contracts_enabled = True  # noqa: SLF001
    rng = random.Random(seed)
    env.reset(seed=seed)

    for i in range(steps):
        action = _choose_mask_action(env, rng) if i % 7 != 0 else _choose_semantically_invalid_action(env, rng)
        _, reward, done, _ = env.step(action)

        # Bounded reward contract for training stability.
        assert -1.0 <= float(reward.value) <= 1.0

        # State consistency and caps are hard contracts.
        env._validate_state_consistency()  # noqa: SLF001
        assert len(env._patients) <= env._max_patients_total  # noqa: SLF001
        assert len(env._delayed_outcomes) <= env._max_delayed_outcomes  # noqa: SLF001

        if done:
            env.reset(seed=seed)


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_mask_partition_and_progression(seed: int) -> None:
    env = MedicalEmergencyRoomEnv(difficulty="hard", seed=seed, debug=False)
    env._contracts_enabled = True  # noqa: SLF001
    env.reset(seed=seed)

    # Warm-up a few steps to create mixed states.
    rng = random.Random(seed)
    for _ in range(20):
        env.step(_choose_mask_action(env, rng))

    mask = env.get_action_mask()
    assign_ids = set(mask["assign_esi_patient_ids"])
    alloc_ids = set(mask["allocate_bed"].keys())

    # Mutually exclusive action sets enforce progression stage.
    assert assign_ids.isdisjoint(alloc_ids)

    for pid in assign_ids:
        runtime = env._patients[pid]  # noqa: SLF001
        assert runtime.patient.state == "waiting"
        assert runtime.last_assigned_esi is None

    for pid in alloc_ids:
        runtime = env._patients[pid]  # noqa: SLF001
        assert runtime.patient.state == "waiting"
        assert runtime.last_assigned_esi is not None


def test_repeated_esi_assignment_penalized() -> None:
    env = MedicalEmergencyRoomEnv(difficulty="medium", seed=42, debug=False)
    env._contracts_enabled = True  # noqa: SLF001
    env.reset(seed=42)

    mask = env.get_action_mask()
    assert mask["assign_esi_patient_ids"]
    pid = mask["assign_esi_patient_ids"][0]

    _, reward1, _, _ = env.step(Action(action_type="assign_esi", patient_id=pid, esi_level=3))
    _, reward2, _, _ = env.step(Action(action_type="assign_esi", patient_id=pid, esi_level=2))

    # Second reassignment should not grant positive triage reward.
    assert reward2.breakdown.get("esi_grade", 0.0) == 0.0
    assert reward2.breakdown.get("ineffective_penalty", 0.0) <= 0.0

    # Prevents exploit where repeated triage outperforms progression.
    assert float(reward2.value) <= float(reward1.value)
