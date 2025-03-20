import math

import h5py
import numpy as np
import scipy.stats as stats
from pydantic import BaseModel

from ..functions import x, y, r


class DataGenerator:
    THETA_RANGE = (0, math.pi)
    X_Y_RANGE = np.array([0, 1])

    def __init__(self, config: BaseModel):
        self.config = config
        self.x_bins = np.linspace(
            self.X_Y_RANGE[0], self.X_Y_RANGE[1], self.config.bins
        )
        self.y_bins = np.linspace(
            self.X_Y_RANGE[0], self.X_Y_RANGE[1], self.config.bins
        )

    def _create_image(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = (
            np.random.normal(
                0,
                self.config.background_level,
                [self.config.bins] * 2,
            )
            if self.config.background_level > 0
            else np.zeros([self.config.image_size] * 2)
        )
        points_coordinates, rs, thetas = self._create_points()
        image[points_coordinates[:, 0], points_coordinates[:, 1]] += np.random.normal(
            1, self.config.stddev, size=points_coordinates.shape[0]
        )
        return image, rs, thetas

    def _points_on_line(self) -> tuple[np.ndarray, float, float]:
        theta_ = np.random.uniform(
            low=self.THETA_RANGE[0],
            high=self.THETA_RANGE[1],
        )
        r_range = r(
            theta_,
            xs=np.concatenate([self.X_Y_RANGE, self.X_Y_RANGE]),
            ys=np.concatenate([self.X_Y_RANGE, self.X_Y_RANGE[::-1]]),
        )
        r_range = [
            max(self.X_Y_RANGE[0], r_range.min()),
            min(self.X_Y_RANGE[1], r_range.max()),
        ]
        r_ = np.random.uniform(
            low=r_range[0],
            high=r_range[1],
        )
        y_range = y(self.X_Y_RANGE, r_, theta_)
        y_range.sort()
        ys = np.random.uniform(
            max(self.X_Y_RANGE[0], y_range[0]),
            min(y_range[1], self.X_Y_RANGE[1]),
            self.config.points_per_line,
        )
        thetas = (
            stats.truncnorm(
                (self.THETA_RANGE[0] - theta_) / self.config.deviations.theta,
                (self.THETA_RANGE[1] - theta_) / self.config.deviations.theta,
                loc=theta_,
                scale=self.config.deviations.theta,
            ).rvs(self.config.points_per_line)
            if self.config.deviations.theta > 0
            else theta_
        )
        rs = (
            stats.truncnorm(
                (r_range[0] - r_) / self.config.deviations.r,
                (r_range[1] - r_) / self.config.deviations.r,
                loc=r_,
                scale=self.config.deviations.r,
            ).rvs(self.config.points_per_line)
            if self.config.deviations.theta > 0
            else r_
        )
        return (
            np.concatenate([x(ys, rs, thetas).reshape(-1, 1), ys.reshape(-1, 1)], 1),
            r_,
            theta_,
        )

    def _create_noise_points_coordinates(self) -> np.ndarray:
        return np.random.uniform(
            low=(self.X_Y_RANGE[0], self.X_Y_RANGE[0]),
            high=(self.X_Y_RANGE[1], self.X_Y_RANGE[1]),
            size=(self.config.n_lines * self.config.outside_points, 2),
        )

    def _create_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        noise = self._create_noise_points_coordinates()
        points, rs, thetas = list(
            zip(*(self._points_on_line() for _ in range(self.config.n_lines)))
        )

        return (
            self._coordinates_to_pixel(
                np.concatenate([noise, *points] if noise.size != 0 else list(points))
            ),
            np.array(rs),
            np.array(thetas),
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
