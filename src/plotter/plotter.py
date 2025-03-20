import numpy as np
import matplotlib.pyplot as plt
from copy import copy

from ..functions import y

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})


class Plotter:
    DEFAULT_KWARG = {
        "cmap": "viridis",
    }

    def __init__(
        self,
        image: np.ndarray,
        limits: tuple[tuple[int, int], tuple[int, int]],
        lines: np.ndarray | None,
    ):
        self.image = image
        self.lines = lines
        self.limits = [v for values in limits for v in values]

    def _add_lines(self):
        xs = np.linspace(self.limits[0], self.limits[1], 100)
        for r_, theta_ in self.lines:
            ys = y(xs, r_, theta_).reshape(-1)
            plt.plot(
                xs,
                ys,
                linewidth=1,
                label=f"$r={round(r_, 2)}$, $\\theta={round(theta_, 2)}$",
            )
            plt.legend()
            ax = plt.gca()
            ax.set_xlim(self.limits[0:2])
            ax.set_ylim(self.limits[2:4])

    def plot(
        self,
        save_as: str,
        include_lines: bool = True,
        **kwargs,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 10))
        include_lines = include_lines and self.lines is not None
        plt_kwargs = copy(self.DEFAULT_KWARG)
        plt_kwargs.update(kwargs)
        image = ax.imshow(
            self.image.T,
            origin="lower",
            extent=self.limits,
            **plt_kwargs,
        )
        fig.colorbar(image, ax=ax, shrink=0.8)
        if include_lines:
            self._add_lines()
        fig.tight_layout()
        plt.savefig(save_as)
        plt.close("all")
