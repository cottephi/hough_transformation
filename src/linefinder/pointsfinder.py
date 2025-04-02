import numpy as np
import scipy.ndimage.filters as filters
from pydantic import validate_call

from ..types import COORDINATE, R_THETA, IMAGE, POINTS


class PointsFinder:
    @validate_call
    def __init__(
        self, data: IMAGE, threshold: float, spread: int
    ):
        self.threshold = threshold
        self.data = data
        self.spread = spread

    @validate_call
    def _set_data(self, data: IMAGE):
        self.data = data

    def find(self) -> POINTS | R_THETA:
        size = self.spread * 2 + 1
        kernel = -np.ones((size, size))
        kernel[self.spread, self.spread] = size ** 2
        convoluted = filters.convolve(self.data, kernel, mode="constant")
        maxima = convoluted == filters.maximum_filter(convoluted, self.spread)
        mask = (self.data > self.threshold) & maxima

        xs, ys = np.where(mask)
        return np.concatenate(
            [np.array(xs).reshape(-1, 1), np.array(ys).reshape(-1, 1)], axis=1
        ) + 0.5  # +0.5 to center the bins

    @staticmethod
    def distance(reference: COORDINATE, points: POINTS) -> np.ndarray:
        diff = points - reference
        return (diff[:, 0] ** 2 + diff[:, 1] ** 2) ** (1/2)