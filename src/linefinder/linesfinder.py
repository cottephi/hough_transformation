import math
from pathlib import Path
from functools import partial
import numpy as np
import h5py
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
from pydantic import validate_call

from ..types import IMAGE
from .pointsfinder import PointsFinder
from ..functions import r


class LinesFinder:
    THETA_RANGE = (0, math.pi)

    @validate_call
    def __init__(
        self, data: IMAGE | Path, threshold: float, output: Path, bins: tuple[int, int]
    ):
        self.output = output
        self.bins = bins
        self.thetas = np.linspace(
            self.THETA_RANGE[0], self.THETA_RANGE[1], self.bins[1]
        ).reshape(-1, 1)
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            if not data.suffix == ".hdf5":
                raise ValueError("Can only read HDF5 files")
            with h5py.File(data, "r") as f:
                if not "data" in f.keys():
                    raise ValueError("HDF5 file must contain the 'data' key")
                self._set_data(f["data"][()])
        self.pointsfinder = PointsFinder(self.data, threshold, output)

    @validate_call
    def _set_data(self, data: IMAGE):
        self.data = data

    @validate_call
    def find(self):
        points = self.pointsfinder.find()
        rs = np.apply_along_axis(
            partial(r, xs=points[:, 0], ys=points[:, 1]), 1, self.thetas
        )
        r_range = [rs.min().min(), rs.max().max()]
        r_bins = np.linspace(r_range[0], r_range[1], self.bins[0])
        binned_rs = np.digitize(rs, r_bins) - 1
        binned_thetas = np.digitize(self.thetas[:, 0], self.thetas[:, 0]) - 1
        image = np.zeros(self.bins)
        rs_thetas = np.concatenate(
            [
                binned_rs.reshape(binned_rs.shape[0] * binned_rs.shape[1], 1),
                np.concatenate([binned_thetas] * binned_rs.shape[1]).reshape(-1, 1),
            ],
            1,
        )
        uniques = np.unique(rs_thetas, axis=0, return_counts=True)
        image[uniques[0][:, 0], uniques[0][:, 1]] = uniques[1]
        self._plot(image)

    @validate_call
    def _plot(self, data: IMAGE):
        fig, ax = plt.subplots(figsize=(10, 10 * data.shape[1] / data.shape[0]))
        image = ax.imshow(
            data.T,
            origin="lower",
            extent=[0, data.shape[0], 0, data.shape[1]],
        )
        fig.colorbar(image, ax=ax, shrink=0.8)
        ax = plt.gca()
        ax.set_xlim([0, data.shape[0]])
        ax.set_ylim([0, data.shape[1]])
        fig.tight_layout()
        plt.savefig(self.output / "r_theta.pdf")
