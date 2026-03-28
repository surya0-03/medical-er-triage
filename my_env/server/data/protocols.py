from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator


ProtocolType = Literal["stroke_code", "stemi_alert", "sepsis_alert", "trauma_alert"]
ProtocolBedType = Literal["icu", "general", "hallway", "none"]


class ProtocolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    required_symptoms: list[StrictStr]
    minimum_esi_requirement: StrictInt = Field(ge=1, le=5)
    required_bed_type: ProtocolBedType
    required_resource: StrictStr = Field(min_length=1)
    cooldown_steps: StrictInt = Field(ge=1)

    @field_validator("required_symptoms")
    @classmethod
    def validate_required_symptoms(cls, value: list[StrictStr]) -> list[StrictStr]:
        if len(value) < 2:
            raise ValueError("required_symptoms must include at least 2 symptoms")
        if any(not symptom.strip() for symptom in value):
            raise ValueError("required_symptoms cannot contain empty values")
        return value


PROTOCOL_DEFINITIONS: Final[dict[ProtocolType, ProtocolDefinition]] = {
    "stroke_code": ProtocolDefinition(
        required_symptoms=["facial droop", "slurred speech", "unilateral weakness"],
        minimum_esi_requirement=2,
        required_bed_type="icu",
        required_resource="ct_scanner",
        cooldown_steps=4,
    ),
    "stemi_alert": ProtocolDefinition(
        required_symptoms=["crushing chest pain", "diaphoresis", "shortness of breath"],
        minimum_esi_requirement=2,
        required_bed_type="icu",
        required_resource="cath_lab_team",
        cooldown_steps=3,
    ),
    "sepsis_alert": ProtocolDefinition(
        required_symptoms=["fever", "tachycardia", "altered mental status"],
        minimum_esi_requirement=2,
        required_bed_type="general",
        required_resource="broad_spectrum_antibiotics",
        cooldown_steps=2,
    ),
    "trauma_alert": ProtocolDefinition(
        required_symptoms=["polytrauma", "hypotension", "active bleeding"],
        minimum_esi_requirement=1,
        required_bed_type="icu",
        required_resource="massive_transfusion_pack",
        cooldown_steps=5,
    ),
}


class ProtocolCooldownTracker(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    last_trigger_step: dict[ProtocolType, StrictInt] = Field(
        default_factory=lambda: {
            "stroke_code": -10_000,
            "stemi_alert": -10_000,
            "sepsis_alert": -10_000,
            "trauma_alert": -10_000,
        }
    )

    @field_validator("last_trigger_step")
    @classmethod
    def validate_last_trigger_step(cls, value: dict[ProtocolType, StrictInt]) -> dict[ProtocolType, StrictInt]:
        expected_keys = set(PROTOCOL_DEFINITIONS.keys())
        if set(value.keys()) != expected_keys:
            raise ValueError("last_trigger_step must include exactly all defined protocols")
        return value

    def cooldown_remaining(self, protocol: ProtocolType, current_time_step: int) -> int:
        if current_time_step < 0:
            raise ValueError("current_time_step must be non-negative")

        definition = PROTOCOL_DEFINITIONS[protocol]
        elapsed = current_time_step - self.last_trigger_step[protocol]
        remaining = definition.cooldown_steps - elapsed
        return remaining if remaining > 0 else 0

    def can_trigger(self, protocol: ProtocolType, current_time_step: int) -> bool:
        return self.cooldown_remaining(protocol, current_time_step) == 0

    def mark_triggered(self, protocol: ProtocolType, current_time_step: int) -> None:
        if current_time_step < 0:
            raise ValueError("current_time_step must be non-negative")
        self.last_trigger_step[protocol] = current_time_step
