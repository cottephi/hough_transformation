import numpy as np


def x(
    ys: np.ndarray | float, rs: np.ndarray | float, thetas: np.ndarray | float
) -> np.ndarray | float:
    return (rs - ys * np.sin(thetas)) / np.cos(thetas)


def y(
    xs: np.ndarray | float, rs: np.ndarray | float, thetas: np.ndarray | float
) -> np.ndarray | float:
    return (rs - xs * np.cos(thetas)) / np.sin(thetas)


def r(
    thetas: np.ndarray | float, xs: np.ndarray | float, ys: np.ndarray | float
) -> np.ndarray | float:
    return xs * np.cos(thetas) + ys * np.sin(thetas)
