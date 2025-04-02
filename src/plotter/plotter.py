import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from pydantic import validate_call

from ..functions import y
from ..objects import Line
from ..types import COORDINATES, IMAGE

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})


class Plotter:
    DEFAULT_KWARG = {
        "cmap": "viridis",
    }

    @validate_call
    def __init__(
        self,
        image: IMAGE,
        lines: tuple[Line, ...] | None,
        points: COORDINATES | None,
    ):
        self.image = image
        self.lines = lines if lines is not None else []
        self.points = points if points is not None else np.array([])
        self.bins = self.image.shape

    def _add_lines_and_points(self):
        xs = np.linspace(0, self.bins[0], 100)
        points_on_a_line = np.zeros(shape=(0, 2), dtype=int)
        for line in self.lines:
            ys = y(xs, line.r, line.theta).reshape(-1)
            plt.plot(
                xs,
                ys,
                linewidth=1,
                linestyle="--",
                label=f"$r={round(line.r, 2)}$, "
                f"$\\theta={round(line.theta / math.pi, 2)}\\pi$",
            )
            if (
                line.binned_coordinates is not None
                and line.binned_coordinates.size != 0
            ):
                points_on_this_line = line.max_points.astype(int)
                plt.scatter(
                    points_on_this_line[:, 0],
                    points_on_this_line[:, 1],
                    marker="o",
                    color="b",
                    s=0.3,
                )
                points_on_a_line = np.concatenate(
                    [points_on_a_line, points_on_this_line]
                )
        if self.points.size != 0:
            mask = np.isin(self.points[:, 0], points_on_a_line[:, 0]) * np.isin(
                self.points[:, 1], points_on_a_line[:, 1]
            )
            not_on_a_line = self.points[~mask]
            plt.scatter(
                not_on_a_line[:, 0],
                not_on_a_line[:, 1],
                marker="o",
                color="r",
                s=0.4,
                label="Points not on a line" if self.lines else "Found points",
            )
        ax = plt.gca()
        ax.set_xlim([0, self.bins[0]])
        ax.set_ylim([0, self.bins[1]])
        plt.legend()

    @validate_call
    def plot(
        self,
        save_as: str | Path,
        **kwargs,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 10 * self.bins[1] / self.bins[0]))
        plt_kwargs = copy(self.DEFAULT_KWARG)
        plt_kwargs.update(kwargs)
        image = ax.imshow(
            self.image.T,
            origin="lower",
            extent=[0, self.bins[0], 0, self.bins[1]],
            **plt_kwargs,
        )
        fig.colorbar(image, ax=ax, shrink=0.8)
        self._add_lines_and_points()
        fig.tight_layout()
        plt.savefig(save_as)
        plt.close("all")
