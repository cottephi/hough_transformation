from pydantic import BaseModel, NonNegativeFloat
from pydantic import validate_call
import pydantic_core

from src.functions import r

from .types import COORDINATES, IMAGE, POINTS, SIGNAL


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
    """The standard deviations of r, theta; the Gaussian used
    to spread the signal across X and Y, the gaussian to generate the signal
    value"""

    r: NonNegativeFloat
    theta: NonNegativeFloat
    spread: NonNegativeFloat
    signal: NonNegativeFloat


class XYBins(BaseModel):
    """Number of pixels in the x-y space"""
    x: int
    y: int


class RThetaBins(BaseModel):
    """Number of pixels in the r-theta space"""
    r: int
    theta: int


class Spreads(BaseModel):
    """Zone around a point in x-y and r-theta spaces which will be used as
    the kernel for the edge-sharpening convolution filter. The kernel's
    dimensions will be 2*spread+1 x 2*spread+1, with a value of (2*spread+1)**2
    at the center bin and -1 everywhere else."""
    xy: int
    rtheta: int


class Thresholds(BaseModel):
    """Threshold above which we consider data is signal in x-y space and
    threshold above which we consider data is a line in r-theta space."""
    xy: float
    rtheta: float


class Line(Points):
    @validate_call
    def __init__(
        self,
        r: float,
        theta: float,
        binned_coordinates: COORDINATES | None = None,
        signal: SIGNAL | None = None,
    ):
        super().__init__(binned_coordinates, signal)
        self.r = r
        self.theta = theta
        self.max_points: POINTS

    @validate_call
    def points_on_line(self, points: POINTS, width: float, image: IMAGE):
        rs = r(xs=points[:, 0], ys=points[:, 1], thetas=self.theta)
        mask = ((self.r - width / 2.0) < rs) & ((self.r + width / 2.0) > rs)
        self.max_points = points[mask]
