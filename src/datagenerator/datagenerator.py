import math
from pathlib import Path
import shutil

import h5py
import numpy as np
import scipy.stats as stats
from pydantic import BaseModel

from ..functions import x, y, r


class DataGenerator:
    THETA_RANGE = (0, math.pi)

    def __init__(self, config: BaseModel):
        self.config = config
        self.x_bins: np.ndarray
        self.y_bins: np.ndarray
        self._handle_bins()

    def _handle_bins(self):
        self.x_bins = np.linspace(0, self.config.bins[0], self.config.bins[0])
        self.y_bins = np.linspace(0, self.config.bins[1], self.config.bins[1])

    def _create_image(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = (
            np.random.normal(
                0,
                self.config.background_level,
                (self.config.bins[0], self.config.bins[1]),
            )
            if self.config.background_level > 0
            else np.zeros((self.config.bins[0], self.config.bins[1]))
        )
        points_coordinates, rs, thetas, points = self._create_points()
        if (
            points_coordinates[:, 0].max() >= image.shape[0]
            or points_coordinates[:, 1].max() >= image.shape[1]
        ):
            self._dumps(points, points_coordinates, rs, thetas, image)
        image[points_coordinates[:, 0], points_coordinates[:, 1]] += np.random.normal(
            1, self.config.stddev, size=points_coordinates.shape[0]
        )
        return image, rs, thetas

    def _dumps(self, points, points_coordinates, rs, thetas, image):
        dumpdir = Path("dumped_data")
        if dumpdir.is_file():
            dumpdir.unlink()
        if dumpdir.is_dir():
            shutil.rmtree(dumpdir)
        dumpdir.mkdir()

        np.savetxt(dumpdir / "coordinates.csv", points_coordinates, delimiter=",")
        start_at = 0
        if len(points) != self.config.n_lines:
            start_at = 1
            np.savetxt(dumpdir / f"points_noise.csv", points[0], delimiter=",")
        for i, points_l in enumerate(points[start_at:]):
            np.savetxt(dumpdir / f"points_line_{i}.csv", points_l, delimiter=",")
        points = np.concatenate(points)
        np.savetxt(dumpdir / "rs.csv", rs, delimiter=",")
        np.savetxt(dumpdir / "thetas.csv", thetas, delimiter=",")

        out_of_bound_x = (points_coordinates[:, 0] >= image.shape[0]) | (
            points_coordinates[:, 0] < 0
        )
        out_of_bound_y = (points_coordinates[:, 1] >= image.shape[1]) | (
            points_coordinates[:, 1] < 0
        )

        np.savetxt(
            dumpdir / "x_out_of_bound.csv",
            np.concatenate(
                [
                    points[out_of_bound_x],
                    points_coordinates[out_of_bound_x].astype(int),
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
                    points_coordinates[out_of_bound_y].astype(int),
                ],
                1,
            ),
            delimiter=",",
        )
        raise ValueError(
            f"Some points are out of bound. Dumped points can be seen in {dumpdir}/"
        )

    def _point_on_line(self, r_theta) -> tuple[float, float]:
        r_, theta_ = r_theta
        y_range = y(np.array([0, self.config.bins[0]]), r_, theta_)
        y_range.sort()
        y_ = np.random.uniform(
            max(0, y_range[0]),
            min(y_range[1], self.config.bins[1]),
        )
        return x(y_, r_, theta_), y_

    def _points_on_line(self) -> tuple[np.ndarray, float, float]:
        shape = (self.config.points_per_line, 1)
        theta_ = np.random.uniform(
            low=self.THETA_RANGE[0],
            high=self.THETA_RANGE[1],
        )
        thetas = (
            stats.truncnorm(
                (self.THETA_RANGE[0] - theta_) / self.config.deviations.theta,
                (self.THETA_RANGE[1] - theta_) / self.config.deviations.theta,
                loc=theta_,
                scale=self.config.deviations.theta,
            ).rvs(shape)
            if self.config.deviations.theta > 0
            else np.ones(shape) * theta_
        )
        r_range = r(
            theta_,
            xs=np.array([0, self.config.bins[0], 0, self.config.bins[0]]),
            ys=np.array([0, self.config.bins[1], self.config.bins[1], 0]),
        )
        r_range = [
            max(0, r_range.min()),
            min(min(self.config.bins), r_range.max()),
        ]
        r_ = np.random.uniform(low=r_range[0], high=r_range[1])
        rs = (
            stats.truncnorm(
                (r_range[0] - r_) / self.config.deviations.r,
                (r_range[1] - r_) / self.config.deviations.r,
                loc=r_,
                scale=self.config.deviations.r,
            ).rvs(shape)
            if self.config.deviations.r > 0
            else np.ones(shape) * r_
        )
        points = np.apply_along_axis(
            self._point_on_line, 1, np.concatenate([rs, thetas], 1)
        )
        return points, r_, theta_

    def _create_noise_points_coordinates(self) -> np.ndarray:
        return np.random.uniform(
            low=(0, 0),
            high=(self.config.bins[0], self.config.bins[1]),
            size=(self.config.n_lines * self.config.outside_points, 2),
        )

    def _create_points(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
        noise = self._create_noise_points_coordinates()
        points, rs, thetas = list(
            zip(*(self._points_on_line() for _ in range(self.config.n_lines)))
        )
        points = [noise, *points] if noise.size != 0 else list(points)

        return (
            self._coordinates_to_pixel(np.concatenate(points)),
            np.array(rs),
            np.array(thetas),
            points,
        )

    def _coordinates_to_pixel(self, points: np.ndarray) -> np.ndarray:
        xs = np.digitize(points[:, 0], self.x_bins)
        ys = np.digitize(points[:, 1], self.y_bins)
        return np.concatenate([xs.reshape(-1, 1), ys.reshape(-1, 1)], 1)

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        image, rs, thetas = self._create_image()
        lines = np.concatenate([rs.reshape(-1, 1), thetas.reshape(-1, 1)], 1)
        with h5py.File(self.config.output, "w") as ofile:
            ofile["data"] = image
            ofile["lines"] = lines
        return image, lines
