import math
from pathlib import Path
import shutil

import h5py
import numpy as np
import scipy.stats as stats
from pydantic import BaseModel, validate_call

from ..functions import x, y, r
from ..objects import Deviations, Points, Line
from ..types import (
    IMAGE,
    POINTS,
    COORDINATES,
    SIGNAL,
    BINS_LIMITS,
    POINT,
    POINTS_AND_SIGNAL,
    R_THETA,
)


class PointsSpreadGenerator(Points):
    @validate_call
    def __init__(
        self,
        points: POINTS,
        bins: tuple[int, int],
        deviations: Deviations,
        stddev: float,
    ):
        self.bins = bins
        self.deviations = deviations
        self.stddev = stddev
        self.points = points
        self.signal: SIGNAL
        self.x_bins: BINS_LIMITS
        self.y_bins: BINS_LIMITS
        self._handle_bins()
        # Do not use super() here as GeneratedLine derives on this class and on
        # Line, and super() can then get confuse
        Points.__init__(self, *self._generate_signal_and_bin())

    def _handle_bins(self):
        self.x_bins = np.linspace(0, self.bins[0], self.bins[0])
        self.y_bins = np.linspace(0, self.bins[1], self.bins[1])

    @validate_call
    def _spread_one_point(self, x_y: POINT) -> POINTS_AND_SIGNAL:
        s = self.deviations.spread
        xmax, ymax = self.bins
        xs, ys = np.mgrid[
            max(x_y[0] - 3 * s, 0) : min(x_y[0] + 3 * s, xmax - 1) : 10j,
            max(x_y[1] - 3 * s, 0) : min(x_y[1] + 3 * s, ymax - 1) : 10j,
        ]
        xs_ys = np.vstack((xs.flatten(), ys.flatten())).T
        spread = stats.multivariate_normal.pdf(
            xs_ys,
            mean=x_y,
            cov=[s] * 2,
        )
        spread = np.random.normal(1, self.stddev) * spread / spread.sum()
        return np.concatenate([xs_ys, spread.reshape(-1, 1)], 1)

    def _generate_signal_and_bin(self) -> tuple[POINTS, SIGNAL]:
        if self.deviations.spread > 0:
            points = np.apply_along_axis(
                self._spread_one_point,
                1,
                self.points,
            ).reshape(-1, 3)
            self.points = points[:, :2]
            signal = points[:, -1]
        else:
            signal = np.random.normal(1, self.stddev, size=self.points.shape[0])

        xs = np.digitize(self.points[:, 0], self.x_bins).reshape(-1, 1)
        ys = np.digitize(self.points[:, 1], self.y_bins).reshape(-1, 1)
        return np.concatenate([xs, ys], 1), signal


class GeneratedLine(PointsSpreadGenerator, Line):
    @validate_call
    def __init__(
        self,
        r: float,
        theta: float,
        points: POINTS,
        bins: tuple[int, int],
        deviations: Deviations,
        stddev: float,
    ):
        Line.__init__(self, r, theta, None, None)
        PointsSpreadGenerator.__init__(self, points, bins, deviations, stddev)


class LineGenerator:
    THETA_RANGE = (0, math.pi)

    @validate_call
    def __init__(
        self,
        bins: tuple[int, int],
        length: int,
        deviations: Deviations,
        stddev: float,
    ):
        self.bins = bins
        self.length = length
        self.deviations = deviations
        self.stddev = stddev

    @validate_call
    def _point_on_line(self, r_theta: R_THETA) -> tuple[float, float]:
        r_, theta_ = r_theta
        y_range = y(np.array([0, self.bins[0]]), r_, theta_)
        y_range.sort()
        y_ = np.random.uniform(
            max(0, y_range[0]),
            min(y_range[1], self.bins[1]),
        )
        return x(y_, r_, theta_), y_

    def _generate_line(self) -> GeneratedLine:
        shape = (self.length, 1)
        theta_ = np.random.uniform(
            low=self.THETA_RANGE[0],
            high=self.THETA_RANGE[1],
        )
        thetas = (
            stats.truncnorm(
                (self.THETA_RANGE[0] - theta_) / self.deviations.theta,
                (self.THETA_RANGE[1] - theta_) / self.deviations.theta,
                loc=theta_,
                scale=self.deviations.theta,
            ).rvs(shape)
            if self.deviations.theta > 0
            else np.ones(shape) * theta_
        )
        r_range = r(
            theta_,
            xs=np.array([0, self.bins[0], 0, self.bins[0]]),
            ys=np.array([0, self.bins[1], self.bins[1], 0]),
        )
        r_range = [
            max(0, r_range.min()),
            min(min(self.bins), r_range.max()),
        ]
        r_ = np.random.uniform(low=r_range[0], high=r_range[1])
        rs = (
            stats.truncnorm(
                (r_range[0] - r_) / self.deviations.r,
                (r_range[1] - r_) / self.deviations.r,
                loc=r_,
                scale=self.deviations.r,
            ).rvs(shape)
            if self.deviations.r > 0
            else np.ones(shape) * r_
        )
        return GeneratedLine(
            r_,
            theta_,
            np.apply_along_axis(
                self._point_on_line, 1, np.concatenate([rs, thetas], 1)
            ),
            self.bins,
            self.deviations,
            self.stddev,
        )

    @validate_call
    def generate(self, n: int) -> tuple[GeneratedLine]:
        return tuple(self._generate_line() for _ in range(n))


class DataGenerator:
    @validate_call
    def __init__(self, config: BaseModel):
        self.config = config

    def _create_image(self) -> tuple[IMAGE, tuple[GeneratedLine]]:
        image = (
            np.random.normal(
                0,
                self.config.background_level,
                (self.config.bins[0], self.config.bins[1]),
            )
            if self.config.background_level > 0
            else np.zeros((self.config.bins[0], self.config.bins[1]))
        )
        noise, lines = self._create_points()
        binned_coordinates = np.concatenate(
            [
                noise.binned_coordinates,
                *(line.binned_coordinates for line in lines),
            ]
        )
        signal = np.concatenate([noise.signal, *(line.signal for line in lines)])
        if (
            binned_coordinates[:, 0].max() >= image.shape[0]
            or binned_coordinates[:, 1].max() >= image.shape[1]
        ):
            self._dumps(binned_coordinates, noise, lines, image)
        image[binned_coordinates[:, 0], binned_coordinates[:, 1]] += signal
        return image, lines

    @validate_call
    def _dumps(
        self,
        binned_coordinates: COORDINATES,
        noise: PointsSpreadGenerator,
        lines: tuple[GeneratedLine],
        image: IMAGE,
    ):
        dumpdir = Path("dumped_data")
        if dumpdir.is_file():
            dumpdir.unlink()
        if dumpdir.is_dir():
            shutil.rmtree(dumpdir)
        dumpdir.mkdir(parents=True)

        np.savetxt(dumpdir / "coordinates.csv", binned_coordinates, delimiter=",")
        if noise.points:
            np.savetxt(dumpdir / f"points_noise.csv", noise.points, delimiter=",")
        for i, line in enumerate(lines):
            np.savetxt(dumpdir / f"points_line_{i}.csv", line.points, delimiter=",")
        points = np.concatenate([noise.points, *(line.points for line in lines)])
        np.savetxt(
            dumpdir / "r_theta.csv",
            np.array([[line.r, line.theta] for line in lines]),
            delimiter=",",
        )

        out_of_bound_x = (binned_coordinates[:, 0] >= image.shape[0]) | (
            binned_coordinates[:, 0] < 0
        )
        out_of_bound_y = (binned_coordinates[:, 1] >= image.shape[1]) | (
            binned_coordinates[:, 1] < 0
        )

        np.savetxt(
            dumpdir / "x_out_of_bound.csv",
            np.concatenate(
                [
                    points[out_of_bound_x],
                    binned_coordinates[out_of_bound_x].astype(int),
                ],
                1,
            ),
            delimiter=",",
        )
        np.savetxt(
            dumpdir / "y_out_of_bound.csv",
            np.concatenate(
                [
                    points[out_of_bound_y],
                    binned_coordinates[out_of_bound_y].astype(int),
                ],
                1,
            ),
            delimiter=",",
        )
        raise ValueError(
            f"Some points are out of bound. Dumped points can be seen in {dumpdir}/"
        )

    def _create_noise_points_coordinates(self) -> PointsSpreadGenerator:
        return PointsSpreadGenerator(
            np.random.uniform(
                low=(0, 0),
                high=(self.config.bins[0], self.config.bins[1]),
                size=(self.config.n_lines * self.config.outside_points, 2),
            ),
            self.config.bins,
            self.config.deviations,
            self.config.stddev,
        )

    def _create_points(
        self,
    ) -> tuple[PointsSpreadGenerator, tuple[GeneratedLine]]:
        noise = self._create_noise_points_coordinates()
        line_generator = LineGenerator(
            self.config.bins,
            self.config.points_per_line,
            self.config.deviations,
            self.config.stddev,
        )
        lines = line_generator.generate(self.config.n_lines)
        return noise, lines

    def generate(self) -> tuple[IMAGE, tuple[GeneratedLine]]:
        image, lines = self._create_image()
        rs_thetas = [(line.r, line.theta) for line in lines]
        with h5py.File(self.config.output, "w") as ofile:
            ofile["data"] = image
            ofile["lines"] = rs_thetas
        return image, lines
