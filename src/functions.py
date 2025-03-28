import numpy as np
from pydantic import validate_call

from .types import X, Y, X, THETA, R


@validate_call
def x(ys: Y | float, rs: R | float, thetas: THETA | float) -> X | float:
    return (rs - ys * np.sin(thetas)) / np.cos(thetas)


@validate_call
def y(xs: X | float, rs: R | float, thetas: THETA | float) -> Y | float:
    return (rs - xs * np.cos(thetas)) / np.sin(thetas)


@validate_call
def r(thetas: THETA | float, xs: X | float, ys: Y | float) -> R | float:
    return xs * np.cos(thetas) + ys * np.sin(thetas)
