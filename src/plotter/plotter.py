import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from pydantic import validate_call

from ..functions import y
from ..objects import Line
from ..types import IMAGE

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
    ):
        self.image = image
        self.lines = lines
        self.bins = self.image.shape

    def _add_lines(self):
        xs = np.linspace(0, self.bins[0], 100)
        for line in self.lines:
            ys = y(xs, line.r, line.theta).reshape(-1)
            plt.plot(
                xs,
                ys,
                linewidth=1,
                label=f"$r={round(line.r, 2)}$, "
                f"$\\theta={round(line.theta / math.pi, 2)}\\pi$",
            )
            plt.legend()
            ax = plt.gca()
            ax.set_xlim([0, self.bins[0]])
            ax.set_ylim([0, self.bins[1]])

    @validate_call
    def plot(
        self,
        save_as: str | Path,
        include_lines: bool = True,
        **kwargs,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 10 * self.bins[1] / self.bins[0]))
        include_lines = include_lines and self.lines is not None
        plt_kwargs = copy(self.DEFAULT_KWARG)
        plt_kwargs.update(kwargs)
        image = ax.imshow(
            self.image.T,
            origin="lower",
            extent=[0, self.bins[0], 0, self.bins[1]],
            **plt_kwargs,
        )
        fig.colorbar(image, ax=ax, shrink=0.8)
        if include_lines:
            self._add_lines()
        fig.tight_layout()
        plt.savefig(save_as)
        plt.close("all")
