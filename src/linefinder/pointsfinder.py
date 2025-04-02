import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from pydantic import validate_call

from ..types import R_THETA, IMAGE, POINTS


class PointsFinder:
    @validate_call
    def __init__(
        self, data: IMAGE, threshold: float, spread: float
    ):
        self.threshold = threshold
        self.data = data
        self.spread = spread

    @validate_call
    def _set_data(self, data: IMAGE):
        self.data = data

    def find(self) -> POINTS | R_THETA:
        neighborhood_size = 5
        data_max = filters.maximum_filter(self.data, neighborhood_size)
        maxima = self.data == data_max
        data_min = filters.minimum_filter(self.data, neighborhood_size)
        diff = (data_max - data_min) > self.threshold
        maxima[~diff] = 0

        labeled, _ = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        xs, ys = [], []
        for dx, dy in slices:
            x_center = (dx.start + dx.stop) / 2
            xs.append(x_center)
            y_center = (dy.start + dy.stop) / 2
            ys.append(y_center)
        return np.concatenate(
            [np.array(xs).reshape(-1, 1), np.array(ys).reshape(-1, 1)], axis=1
        )
