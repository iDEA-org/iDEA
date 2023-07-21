"""Contains some pre-defined electron-electron interations"""


import numpy as np


__all__ = [
    "softened_interaction",
    "softened_interaction_alternative",
]


def softened_interaction(
    x: np.ndarray, strength: float = 1.0, softening: float = 1.0
) -> np.ndarray:
    r"""
    Constructs the softened interaction potential.

    .. math:: v_\mathrm{int}(x,x') = \frac{s}{|x-x'| + a}


    | Args:
    |     x: np.ndarray, x grid.
    |     strength: float, Strength of the interaction .. math:: s. (default = 1.0)
    |     softening: float, Softening parameter of the interaction .. math:: a. (default = 1.0)

    | Returns:
    |     v_int: np.ndarray, Softened interaction potential on x grid of the System.
    """
    v_int = np.zeros((x.shape[0], x.shape[0]), dtype="float")
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            v_int[i, j] = strength / (abs(x[i] - x[j]) + softening)
    return v_int


def softened_interaction_alternative(
    x: np.ndarray, strength: float = 1.0, softening: float = 1.0
) -> np.ndarray:
    r"""
    Constructs the alternative softened interaction potential.

    .. math:: v_\mathrm{int}(x,x') = \frac{s}{{(\sqrt{x-x'} + a)}^{2}}

    | Args:
    |     x: np.ndarray, x grid.
    |     strength: float, Strength of the interaction .. math:: s. (default = 1.0)
    |     softening: float, Softening parameter of the interaction .. math:: a. (default = 1.0)

    | Returns:
    |     v_int: np.ndarray, Softened interaction potential on x grid of the System.
    """
    v_int = np.zeros((x.shape[0], x.shape[0]), dtype="float")
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            v_int[i, j] = strength / np.sqrt(((x[i] - x[j]) ** 2 + softening))
    return v_int
