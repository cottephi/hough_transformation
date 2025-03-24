from pydantic import BaseModel, NonNegativeFloat
import numpy as np


class Points:
    def __init__(
        self,
        binned_coordinates: np.ndarray | None,
        signal: np.ndarray | None,
    ):
        self.binned_coordinates = binned_coordinates
        self.signal = signal


class Deviations(BaseModel):
    r: NonNegativeFloat
    theta: NonNegativeFloat
    spread: NonNegativeFloat

    def __repr__(self):
        return f"sigma_r={self.r},sigma_theta={self.theta},spread={self.spread}"


class Line(Points):
    def __init__(
        self,
        r: float,
        theta: float,
        binned_coordinates: np.ndarray | None,
        signal: np.ndarray | None,
    ):
        super().__init__(binned_coordinates, signal)
        self.r = r
        self.theta = theta
