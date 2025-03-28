from pathlib import Path
import numpy as np
import h5py
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
from pydantic import validate_call

from ..types import X, Y, IMAGE, POINTS


class PointsFinder:
    @validate_call
    def __init__(self, data: IMAGE, threshold: float, output: Path):
        self.threshold = threshold
        self.output = output
        self.data = data

    @validate_call
    def _set_data(self, data: IMAGE):
        self.data = data

    def find(self) -> POINTS:
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
        self._plot_points(xs, ys)
        return np.concatenate(
            [np.array(xs).reshape(-1, 1), np.array(ys).reshape(-1, 1)], axis=1
        )

    @validate_call
    def _plot_points(self, xs: X, ys: Y):
        fig, ax = plt.subplots(
            figsize=(10, 10 * self.data.shape[1] / self.data.shape[0])
        )
        image = ax.imshow(
            self.data.T,
            origin="lower",
            extent=[0, self.data.shape[0], 0, self.data.shape[1]],
        )
        fig.colorbar(image, ax=ax, shrink=0.8)
        plt.scatter(
            xs,
            ys,
            marker="o",
            color="r",
            s=0.3,
        )
        ax = plt.gca()
        ax.set_xlim([0, self.data.shape[0]])
        ax.set_ylim([0, self.data.shape[1]])
        fig.tight_layout()
        plt.savefig(self.output / "found_points.pdf")
