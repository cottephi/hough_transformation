from pydantic import BaseModel, NonNegativeFloat
from pydantic import validate_call
import pydantic_core

from .types import COORDINATES, SIGNAL


class Points:
    @validate_call
    def __init__(
        self,
        binned_coordinates: COORDINATES | None,
        signal: SIGNAL | None,
    ):
        self.binned_coordinates = binned_coordinates
        self.signal = signal

    @classmethod
    def __get_pydantic_core_schema__(cls, _, __):
        def validate(value):
            return value

        schema = pydantic_core.core_schema.union_schema(
            [
                pydantic_core.core_schema.is_instance_schema(cls),
            ]
        )

        return pydantic_core.core_schema.no_info_after_validator_function(
            validate, schema
        )


class Deviations(BaseModel):
    r: NonNegativeFloat
    theta: NonNegativeFloat
    spread: NonNegativeFloat

    def __repr__(self):
        return f"sigma_r={self.r},sigma_theta={self.theta},spread={self.spread}"


class Line(Points):
    @validate_call
    def __init__(
        self,
        r: float,
        theta: float,
        binned_coordinates: COORDINATES | None,
        signal: SIGNAL | None,
    ):
        super().__init__(binned_coordinates, signal)
        self.r = r
        self.theta = theta
