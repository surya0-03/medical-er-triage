from __future__ import annotations

import math
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, field_validator, model_validator

# Keep these duplicated locally to avoid circular imports from environment.
ICU_BEDS_TOTAL = 8
GENERAL_BEDS_TOTAL = 20
HALLWAY_CAPACITY = 3
ICU_NURSE_RATIO_NUMERATOR = 1
ICU_NURSE_RATIO_DENOMINATOR = 2
ED_NURSE_RATIO_NUMERATOR = 1
ED_NURSE_RATIO_DENOMINATOR = 3

ProtocolType = Literal["stroke_code", "stemi_alert", "sepsis_alert", "trauma_alert"]


class OpenEnvModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class VitalsDict(TypedDict):
    heart_rate: StrictInt
    blood_pressure: StrictStr
    oxygen_level: StrictFloat


class StaffLoadDict(TypedDict):
    icu_nurses: StrictInt
    ed_nurses: StrictInt
    icu_patients: StrictInt
    ed_patients: StrictInt
    icu_nurse_load: StrictFloat
    ed_nurse_load: StrictFloat
    icu_ratio_load: StrictFloat
    ed_ratio_load: StrictFloat
    total_nurse_load: StrictFloat


PatientState = Literal["waiting", "assigned_bed", "in_treatment", "completed", "outcome_pending", "resolved"]
ActionType = Literal["assign_esi", "allocate_bed", "discharge", "trigger_protocol", "divert"]
BedType = Literal["icu", "general", "hallway", "none"]


class Patient(OpenEnvModel):
    patient_id: StrictStr = Field(min_length=1)
    age: StrictInt = Field(ge=0, le=130)
    symptoms: list[StrictStr]
    vitals: VitalsDict
    esi_level: StrictInt = Field(ge=1, le=5)
    wait_time: StrictInt = Field(ge=0)
    state: PatientState
    deterioration_count: StrictInt = Field(ge=0)

    @field_validator("symptoms")
    @classmethod
    def validate_symptoms(cls, value: list[StrictStr]) -> list[StrictStr]:
        if len(value) == 0:
            raise ValueError("symptoms must contain at least one symptom")
        if any(not symptom.strip() for symptom in value):
            raise ValueError("symptoms cannot contain empty strings")
        return value

    @field_validator("vitals")
    @classmethod
    def validate_vitals(cls, value: VitalsDict) -> VitalsDict:
        heart_rate = value["heart_rate"]
        oxygen_level = value["oxygen_level"]
        blood_pressure = value["blood_pressure"]

        if heart_rate <= 0:
            raise ValueError("vitals.heart_rate must be greater than 0")
        if oxygen_level < 0.0 or oxygen_level > 100.0:
            raise ValueError("vitals.oxygen_level must be within [0, 100]")

        parts = blood_pressure.split("/")
        if len(parts) != 2:
            raise ValueError("vitals.blood_pressure must be in the format SYS/DIA")
        if not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError("vitals.blood_pressure values must be numeric")

        return value


class Observation(OpenEnvModel):
    time_step: StrictInt = Field(ge=0)
    patients: list[Patient]
    waiting_queue: list[StrictStr]
    icu_beds_available: StrictInt = Field(ge=0)
    general_beds_available: StrictInt = Field(ge=0)
    hallway_used: StrictInt = Field(ge=0)
    crowding_score: StrictFloat = Field(ge=0.0)
    staff_load: StaffLoadDict

    @field_validator("staff_load")
    @classmethod
    def validate_staff_load(cls, value: StaffLoadDict) -> StaffLoadDict:
        if value["icu_nurses"] < 0:
            raise ValueError("staff_load.icu_nurses must be non-negative")
        if value["ed_nurses"] < 0:
            raise ValueError("staff_load.ed_nurses must be non-negative")
        if value["icu_patients"] < 0:
            raise ValueError("staff_load.icu_patients must be non-negative")
        if value["ed_patients"] < 0:
            raise ValueError("staff_load.ed_patients must be non-negative")
        if value["icu_nurse_load"] < 0.0:
            raise ValueError("staff_load.icu_nurse_load must be non-negative")
        if value["ed_nurse_load"] < 0.0:
            raise ValueError("staff_load.ed_nurse_load must be non-negative")
        if value["icu_ratio_load"] < 0.0:
            raise ValueError("staff_load.icu_ratio_load must be non-negative")
        if value["ed_ratio_load"] < 0.0:
            raise ValueError("staff_load.ed_ratio_load must be non-negative")
        if value["total_nurse_load"] < 0.0:
            raise ValueError("staff_load.total_nurse_load must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_queue_consistency(self) -> Observation:
        if self.icu_beds_available > ICU_BEDS_TOTAL:
            raise ValueError("icu_beds_available cannot exceed ICU_BEDS_TOTAL")
        if self.general_beds_available > GENERAL_BEDS_TOTAL:
            raise ValueError("general_beds_available cannot exceed GENERAL_BEDS_TOTAL")
        if self.hallway_used > HALLWAY_CAPACITY:
            raise ValueError("hallway_used cannot exceed HALLWAY_CAPACITY")

        patient_map = {patient.patient_id: patient for patient in self.patients}

        if len(patient_map) != len(self.patients):
            raise ValueError("patients must have unique patient_id values")

        seen_in_queue: set[str] = set()
        for patient_id in self.waiting_queue:
            if patient_id in seen_in_queue:
                raise ValueError("waiting_queue cannot contain duplicate patient_id values")
            seen_in_queue.add(patient_id)

            patient = patient_map.get(patient_id)
            if patient is None:
                raise ValueError("waiting_queue contains patient_id not present in patients")
            if patient.state != "waiting":
                raise ValueError("waiting_queue patient must have state='waiting'")

        expected = calculate_staff_load(
            patients=self.patients,
            icu_nurses=self.staff_load["icu_nurses"],
            ed_nurses=self.staff_load["ed_nurses"],
        )
        float_fields = [
            "icu_nurse_load", "ed_nurse_load",
            "icu_ratio_load", "ed_ratio_load", "total_nurse_load"
        ]
        int_fields = ["icu_nurses", "ed_nurses", "icu_patients", "ed_patients"]
        for field in int_fields:
            if self.staff_load[field] != expected[field]:
                raise ValueError(f"staff_load.{field} is inconsistent with dynamic computation")
        for field in float_fields:
            if abs(float(self.staff_load[field]) - float(expected[field])) > 1e-6:
                raise ValueError(f"staff_load.{field} is inconsistent with dynamic computation")

        return self


class Action(OpenEnvModel):
    action_type: ActionType
    patient_id: StrictStr | None = None
    esi_level: StrictInt | None = Field(default=None, ge=1, le=5)
    bed_type: BedType | None = None
    protocol_type: ProtocolType | None = None

    @model_validator(mode="after")
    def validate_action_payload(self) -> Action:
        action_type = self.action_type

        if action_type == "assign_esi":
            if self.patient_id is None:
                raise ValueError("patient_id is required for assign_esi")
            if self.esi_level is None:
                raise ValueError("esi_level is required for assign_esi")

        if action_type == "allocate_bed":
            if self.patient_id is None:
                raise ValueError("patient_id is required for allocate_bed")
            if self.bed_type is None:
                raise ValueError("bed_type is required for allocate_bed")
            if self.bed_type == "none":
                raise ValueError("bed_type cannot be 'none' for allocate_bed")

        if action_type == "discharge":
            if self.patient_id is None:
                raise ValueError("patient_id is required for discharge")

        if action_type == "trigger_protocol":
            if self.patient_id is None:
                raise ValueError("patient_id is required for trigger_protocol")
            if self.protocol_type is None:
                raise ValueError("protocol_type is required for trigger_protocol")

        if action_type == "divert":
            if self.patient_id is not None:
                raise ValueError("patient_id must be None for divert")
            if self.esi_level is not None:
                raise ValueError("esi_level must be None for divert")
            if self.protocol_type is not None:
                raise ValueError("protocol_type must be None for divert")

        return self


class Reward(OpenEnvModel):
    value: StrictFloat
    breakdown: dict[StrictStr, StrictFloat]

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: StrictFloat) -> StrictFloat:
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        return value

    @field_validator("breakdown")
    @classmethod
    def validate_breakdown(cls, value: dict[StrictStr, StrictFloat]) -> dict[StrictStr, StrictFloat]:
        for key, component in value.items():
            if not key.strip():
                raise ValueError("breakdown keys must be non-empty")
            if not math.isfinite(component):
                raise ValueError("breakdown values must be finite")
        return value


def calculate_staff_load(patients: list[Patient], icu_nurses: int, ed_nurses: int) -> StaffLoadDict:
    if icu_nurses < 0:
        raise ValueError("icu_nurses must be non-negative")
    if ed_nurses < 0:
        raise ValueError("ed_nurses must be non-negative")

    active_states = {"waiting", "assigned_bed", "in_treatment"}
    active_patients = [patient for patient in patients if patient.state in active_states]
    icu_patients = sum(1 for patient in active_patients if patient.esi_level <= 2)
    ed_patients = sum(1 for patient in active_patients if patient.esi_level >= 3)

    icu_nurse_load = float(icu_patients) if icu_nurses == 0 else float(icu_patients) / float(icu_nurses)
    ed_nurse_load = float(ed_patients) if ed_nurses == 0 else float(ed_patients) / float(ed_nurses)
    icu_capacity_patients = (icu_nurses * ICU_NURSE_RATIO_DENOMINATOR) / ICU_NURSE_RATIO_NUMERATOR
    ed_capacity_patients = (ed_nurses * ED_NURSE_RATIO_DENOMINATOR) / ED_NURSE_RATIO_NUMERATOR
    icu_ratio_load = float(icu_patients) if icu_capacity_patients == 0 else float(icu_patients) / icu_capacity_patients
    ed_ratio_load = float(ed_patients) if ed_capacity_patients == 0 else float(ed_patients) / ed_capacity_patients
    total_nurse_load = icu_nurse_load + ed_nurse_load

    return {
        "icu_nurses": icu_nurses,
        "ed_nurses": ed_nurses,
        "icu_patients": icu_patients,
        "ed_patients": ed_patients,
        "icu_nurse_load": icu_nurse_load,
        "ed_nurse_load": ed_nurse_load,
        "icu_ratio_load": icu_ratio_load,
        "ed_ratio_load": ed_ratio_load,
        "total_nurse_load": total_nurse_load,
    }
