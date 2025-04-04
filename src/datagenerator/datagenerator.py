import math
from pathlib import Path
import shutil

import h5py
import numpy as np
import scipy.stats as stats
from pydantic import BaseModel, validate_call

from ..plotter.plotter import Plotter
from ..functions import x, y, r
from ..objects import Deviations, Points, Line, XYBins
from ..types import (
    IMAGE,
    POINTS,
    COORDINATES,
    SIGNAL,
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
    ):
        self.bins = bins
        self.deviations = deviations
        self.max_points = points
        self.points: POINTS
        self.signal: SIGNAL
        # Do not use super() here as GeneratedLine derives on this class and on
        # Line, and super() can then get confuse
        Points.__init__(self, *self._generate_signal_and_bin())

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
        spread = np.random.normal(1, self.deviations.signal) * spread / spread.sum()
        return np.concatenate([xs_ys, spread.reshape(-1, 1)], 1)

    def _generate_signal_and_bin(self) -> tuple[COORDINATES, SIGNAL]:
        if self.max_points.size == 0:
            return np.zeros(shape=(0, 2), dtype=int), np.array([], dtype=float)
        if self.deviations.spread > 0:
            points = np.apply_along_axis(
                self._spread_one_point,
                1,
                self.max_points,
            ).reshape(-1, 3)
            self.points = points[:, :2]
            signal = points[:, -1]
        else:
            self.points = self.max_points
            signal = np.random.normal(1, self.stddev, size=self.points.shape[0])

        xs, ys = self.points.T.astype(int)
        xs = xs.reshape(-1, 1)
        ys = ys.reshape(-1, 1)
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
    ):
        Line.__init__(self, r, theta, None, None)
        PointsSpreadGenerator.__init__(self, points, bins, deviations)


class LineGenerator:
    THETA_RANGE = (0, math.pi)

    @validate_call
    def __init__(
        self,
        bins: XYBins,
        length: int,
        deviations: Deviations,
    ):
        self.bins = bins
        self.length = length
        self.deviations = deviations

    @validate_call
    def _point_on_line(self, r_theta: R_THETA) -> tuple[float, float]:
        r_, theta_ = r_theta
        y_range = y(np.array([0, self.bins.x], dtype=float), r_, theta_)
        y_range.sort()
        y_ = np.random.uniform(
            max(0, y_range[0]),
            min(y_range[1], self.bins.y),
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
            xs=np.array([0, self.bins.x, 0, self.bins.x], dtype=float),
            ys=np.array([0, self.bins.y, self.bins.y, 0], dtype=float),
        )
        r_range = [
            max(0, r_range.min()),
            min(min((self.bins.x, self.bins.y)), r_range.max()),
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
            (self.bins.x, self.bins.y),
            self.deviations,
        )

    @validate_call
    def generate(self, n: int) -> tuple[GeneratedLine]:
        return tuple(self._generate_line() for _ in range(n))


class DataGenerator:
    @validate_call
    def __init__(self, config: BaseModel):
        self.config = config

    def _create_image(self) -> tuple[IMAGE, tuple[GeneratedLine], COORDINATES]:
        image = (
            np.random.normal(
                0,
                self.config.background_level,
                (self.config.bins.x, self.config.bins.y),
            )
            if self.config.background_level > 0
            else np.zeros((self.config.bins.y, self.config.bins.x))
        )
        noise, lines = self._create_points()
        binned_coordinates = np.concatenate(
            [
                noise.binned_coordinates,
                *(line.binned_coordinates for line in lines),
            ]
        )
        max_coorindates = np.concatenate(
            [
                noise.max_points,
                *(line.max_points for line in lines),
            ]
        ).astype(int)
        signal = np.concatenate([noise.signal, *(line.signal for line in lines)])
        if (
            binned_coordinates[:, 0].max() >= image.shape[0]
            or binned_coordinates[:, 1].max() >= image.shape[1]
        ):
            self._dumps(binned_coordinates, noise, lines, image)

        uniques = np.unique(binned_coordinates, axis=0)

        def sum_signal(bins):
            x_coo, y_coo = bins
            mask = np.where(
                (binned_coordinates[:, 0] == x_coo)
                & (binned_coordinates[:, 1] == y_coo)
            )
            return signal[mask].sum()

        image[uniques[:, 0], uniques[:, 1]] += np.apply_along_axis(
            sum_signal, 1, uniques
        )
        return image, lines, max_coorindates

    @validate_call
    def _dumps(
        self,
        binned_coordinates: COORDINATES,
        noise: PointsSpreadGenerator,
        lines: tuple[GeneratedLine, ...],
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
        points = (
            np.zeros(shape=(0, 2), dtype=float)
            if self.config.outside_points == 0
            else np.random.uniform(
                low=(0, 0),
                high=(self.config.bins.x, self.config.bins.y),
                size=(self.config.n_lines * self.config.outside_points, 2),
            )
        )
        return PointsSpreadGenerator(
            points,
            (self.config.bins.x, self.config.bins.y),
            self.config.deviations,
        )

    def _create_points(
        self,
    ) -> tuple[PointsSpreadGenerator, tuple[GeneratedLine]]:
        noise = self._create_noise_points_coordinates()
        line_generator = LineGenerator(
            self.config.bins,
            self.config.points_per_line,
            self.config.deviations,
        )
        lines = line_generator.generate(self.config.n_lines)
        return noise, lines

    def generate(self) -> tuple[IMAGE, tuple[GeneratedLine]]:
        image, lines, coordinates = self._create_image()
        rs_thetas = [(line.r, line.theta) for line in lines]
        with h5py.File(self.config.output, "w") as ofile:
            ofile["data"] = image
            ofile["lines"] = rs_thetas

        plotter = Plotter(image, lines, coordinates)
        plotter.plot(self.config.output.with_suffix(".pdf"))
        return image, lines
