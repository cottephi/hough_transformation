import math
from pathlib import Path
from functools import partial
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pydantic import validate_call

from ..types import IMAGE, POINTS, R
from .pointsfinder import PointsFinder
from ..functions import r
from ..plotter import Plotter
from ..objects import Line


class LinesFinder:
    THETA_RANGE = (0, math.pi)

    @validate_call
    def __init__(
        self,
        data: IMAGE | Path,
        xy_threshold: float,
        rtheta_threshold: int,
        output: Path,
        bins: tuple[int, int],
        line_width: float,
        xy_spread: int,
        rtheta_spread: int,
    ):
        self.output = output
        self.bins = bins
        self.line_width = line_width
        self.rtheta_threshold = rtheta_threshold
        self.rtheta_spread = rtheta_spread
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
        self.xy_bins = self.data.shape
        self.pointsfinder = PointsFinder(self.data, xy_threshold, xy_spread)

    @validate_call
    def _set_data(self, data: IMAGE):
        self.data = data

    @validate_call
    def _create_accumulator(self, points: POINTS) -> tuple[IMAGE, R]:
        rs = np.apply_along_axis(
            partial(r, xs=points[:, 0], ys=points[:, 1]), 1, self.thetas
        ).T
        r_range = (rs.min().min(), rs.max().max())
        r_bins = np.linspace(r_range[0], r_range[1], self.bins[0])
        binned_rs = np.digitize(rs, r_bins) - 1
        binned_thetas = np.digitize(self.thetas[:, 0], self.thetas[:, 0]) - 1
        image = np.zeros(self.bins)
        rs_thetas = np.concatenate(
            [
                binned_rs.reshape(binned_rs.shape[0] * binned_rs.shape[1], 1),
                np.concatenate([binned_thetas] * binned_rs.shape[0]).reshape(-1, 1),
            ],
            1,
        )
        uniques = np.unique(rs_thetas, axis=0, return_counts=True)
        image[uniques[0][:, 0], uniques[0][:, 1]] = uniques[1]
        return image, r_bins

    def find(self):
        points = self.pointsfinder.find()
        lines = []
        if points.size == 0:
            print("No points found")
            return
        accumulator, r_bins = self._create_accumulator(points)
        pointsfinder = PointsFinder(accumulator, self.rtheta_threshold, self.rtheta_spread)
        rs_thetas = pointsfinder.find()
        plotter_r_theta = Plotter(accumulator, None, rs_thetas.astype(int))
        plotter_r_theta.plot(self.output / "found_rtheta.pdf")
        for r_, theta_ in rs_thetas:
            line = Line(r_bins[int(r_)], self.thetas[int(theta_)])
            line.points_on_line(points, self.line_width, self.data)
            lines.append(line)

        with h5py.File(self.output / "found_rtheta.hdf5", "w") as ofile:
            ofile["lines"] = rs_thetas
        plotter = Plotter(self.data, lines, points.astype(int))
        plotter.plot(self.output / "found_lines.pdf")

    @validate_call
    def _plot(self, data: IMAGE, r_bins: list[float]):
        fig, ax = plt.subplots(figsize=(10, 10 * data.shape[1] / data.shape[0]))
        image = ax.imshow(
            data.T,
            origin="lower",
            extent=[0, data.shape[0], 0, data.shape[1]],
        )
        r_step = self.bins[0] / 5
        theta_step = self.bins[1] / 4
        fig.colorbar(image, ax=ax, shrink=0.8)
        ax = plt.gca()
        plt.xticks(
            np.arange(0, self.bins[0] + 1, r_step),
            r_bins[:: int(r_step)] + [r_bins[-1]],
        )
        theta_labels = [
            "0",
            "$\\frac{\\pi}{4}$",
            "$\\frac{\\pi}{2}$",
            "$\\frac{3\\pi}{4}$",
            "$\\pi$",
        ]
        plt.yticks(np.arange(0, self.bins[1] + 1, theta_step), theta_labels)
        plt.xlabel("$r$")
        plt.ylabel("$\\theta$")
        fig.tight_layout()
        plt.savefig(self.output / "r_theta.pdf")
