from __future__ import annotations

from my_env.models import Action, Patient
from my_env.server.data.constants import PROTOCOL_BONUS_CORRECT, PROTOCOL_BONUS_WRONG
from my_env.server.environment import MedicalEmergencyRoomEnv, RuntimePatient


def _build_treatment_ready_sepsis_pair(env: MedicalEmergencyRoomEnv) -> tuple[RuntimePatient, RuntimePatient]:
    env.reset(seed=31)

    base_pid: str | None = None
    for pid in env._waiting_queue:  # noqa: SLF001
        runtime = env._patients[pid]  # noqa: SLF001
        symptoms = set(runtime.patient.symptoms)
        if {"fever", "altered mental status", "tachycardia"}.issubset(symptoms):
            base_pid = pid
            break

    if base_pid is None:
        base_pid = env._waiting_queue[0]  # noqa: SLF001
        runtime = env._patients[base_pid]  # noqa: SLF001
        runtime.patient.symptoms = ["fever", "altered mental status", "tachycardia"]
        runtime.true_esi = 2
        runtime.last_assigned_esi = 2

    r1 = env._patients[base_pid]  # noqa: SLF001
    r1.patient.state = "assigned_bed"
    r1.bed_type = "general"
    r1.last_assigned_esi = r1.true_esi

    clone_id = "p_clone"
    clone_patient = Patient(
        patient_id=clone_id,
        age=r1.patient.age,
        symptoms=list(r1.patient.symptoms),
        vitals=dict(r1.patient.vitals),
        esi_level=r1.patient.esi_level,
        wait_time=0,
        state="assigned_bed",
        deterioration_count=0,
    )
    r2 = RuntimePatient(
        patient=clone_patient,
        true_esi=r1.true_esi,
        last_assigned_esi=r1.true_esi,
        bed_type="general",
        deterioration_events=0,
        wrong_bed_events=0,
    )
    env._patients[clone_id] = r2  # noqa: SLF001

    return r1, r2


def test_divert_overloaded_no_candidate_gets_penalty() -> None:
    env = MedicalEmergencyRoomEnv(difficulty="easy", seed=11)
    env.reset(seed=11)

    env._waiting_queue = []  # noqa: SLF001
    env._hallway_used = 3  # noqa: SLF001

    _, reward, _, _ = env.step(Action(action_type="divert"))

    assert reward.breakdown.get("ineffective_penalty", 0.0) < 0.0


def test_actual_bed_allocated_reports_none_on_failed_allocation() -> None:
    env = MedicalEmergencyRoomEnv(difficulty="easy", seed=12)
    env.reset(seed=12)

    pid = env._waiting_queue[0]  # noqa: SLF001
    runtime = env._patients[pid]  # noqa: SLF001
    runtime.last_assigned_esi = 3

    env._icu_beds_available = 0  # noqa: SLF001
    env._general_beds_available = 0  # noqa: SLF001
    env._hallway_used = 3  # noqa: SLF001

    _, _, _, info = env.step(Action(action_type="allocate_bed", patient_id=pid, bed_type="icu"))

    assert info.get("actual_bed_allocated") == "none"


def test_protocol_cooldown_is_patient_scoped_when_resource_is_free() -> None:
    env = MedicalEmergencyRoomEnv(difficulty="easy", seed=31)
    r1, r2 = _build_treatment_ready_sepsis_pair(env)

    b1 = env._trigger_protocol(r1, "sepsis_alert")  # noqa: SLF001
    env._resource_busy_until["broad_spectrum_antibiotics"] = -10_000  # noqa: SLF001
    b2 = env._trigger_protocol(r2, "sepsis_alert")  # noqa: SLF001

    assert b1 == PROTOCOL_BONUS_CORRECT
    assert b2 == PROTOCOL_BONUS_CORRECT


def test_protocol_resource_contention_blocks_second_patient_same_step() -> None:
    env = MedicalEmergencyRoomEnv(difficulty="easy", seed=31)
    r1, r2 = _build_treatment_ready_sepsis_pair(env)

    env._time_step = 10  # noqa: SLF001
    env._protocol_last_trigger_step_by_patient = {}  # noqa: SLF001
    env._resource_busy_until["broad_spectrum_antibiotics"] = -10_000  # noqa: SLF001

    c1 = env._trigger_protocol(r1, "sepsis_alert")  # noqa: SLF001
    c2 = env._trigger_protocol(r2, "sepsis_alert")  # noqa: SLF001

    assert c1 == PROTOCOL_BONUS_CORRECT
    assert c2 == PROTOCOL_BONUS_WRONG


def test_divert_removes_target_patient() -> None:
    env = MedicalEmergencyRoomEnv(difficulty="hard", seed=13)
    env.reset(seed=13)

    for _ in range(200):
        if env._is_overloaded() and env._waiting_queue:  # noqa: SLF001
            break
        if env._enqueue_new_patient() is None:  # noqa: SLF001
            break

    target = env._waiting_queue[0]  # noqa: SLF001
    env._patients[target].last_assigned_esi = 5  # noqa: SLF001

    before_count = len(env._patients)  # noqa: SLF001
    env.step(Action(action_type="divert"))
    after_count = len(env._patients)  # noqa: SLF001

    assert after_count == before_count - 1
    assert target not in env._patients  # noqa: SLF001
    assert target not in env._waiting_queue  # noqa: SLF001
