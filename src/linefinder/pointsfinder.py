from pathlib import Path
import numpy as np
import h5py
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
from pydantic import validate_call

from ..types import X, Y, IMAGE


class PointsFinder:
    @validate_call
    def __init__(self, data: IMAGE | Path, threshold: float, output: Path):
        self.threshold = threshold
        self.output = output
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            if not data.suffix == ".hdf5":
                raise ValueError("Can only read HDF5 files")
            with h5py.File(data, "r") as f:
                if not "data" in f.keys():
                    raise ValueError("HDF5 file must contain the 'data' key")
                self._set_data(f["data"][()])
        if len(self.data.shape) != 2:
            raise ValueError("'points' must be a 2D array")

    @validate_call
    def _set_data(self, data: IMAGE):
        self.data = data

    def find(self):
        xs, ys = self._find_points()
        self._plot_points(xs, ys)

    def _find_points(self) -> tuple[X, Y]:
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
            x_center = (dx.start + dx.stop - 1) / 2
            xs.append(x_center)
            y_center = (dy.start + dy.stop - 1) / 2
            ys.append(y_center)
        return np.array(xs), np.array(ys)

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
