"""Input contracts shared by API validation and dashboard CSV validation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class SensorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_type: Literal["L", "M", "H"]
    air_temperature_k: FiniteFloat = Field(gt=0)
    process_temperature_k: FiniteFloat = Field(gt=0)
    rotational_speed_rpm: FiniteFloat = Field(gt=0)
    torque_nm: FiniteFloat = Field(ge=0)
    tool_wear_min: FiniteFloat = Field(ge=0)


class BatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    readings: list[SensorInput] = Field(min_length=1, max_length=1000)


class ReviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["pending", "confirmed", "unconfirmed"]
    note: str = Field(default="", max_length=2000)
