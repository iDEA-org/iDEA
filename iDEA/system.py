"""Contains all functionality to define and manage definitions of model systems."""

import pickle
import warnings

import numpy as np

import iDEA.interactions
import iDEA.utilities

__all__ = ["System", "save_system", "load_system", "systems"]


class System:
    r"""Model system, containing all defining properties."""

    def __init__(
        self,
        x: np.ndarray,
        v_ext: np.ndarray,
        v_int: np.ndarray,
        electrons: str,
        stencil: int = 13,
    ):
        r"""
        Model system, containing all defining properties.

        | Args:
        |     x: np.ndarray, Grid of x values in 1D space.
        |     v_ext: np.ndarray, External potential on the grid of x values.
        |     v_int: np.ndarray, Interaction potential on the grid of x values.
        |     electrons: string, Electrons contained in the system.
        |     stencil: int, Stencil to use for derivatives on the grid of x values. (default = 13)

        | Raises:
        |     AssertionError.
        """
        self.__x = x
        self.__dx = self.x[1] - self.x[0]
        self.v_ext = v_ext
        self.v_int = v_int
        self.__electrons = electrons
        self.count = len(electrons)
        self.up_count = electrons.count("u")
        self.down_count = electrons.count("d")
        self.stencil = stencil
        self.check()

    def check(self):
        r"""Performs checks on system properties. Raises AssertionError if any check fails."""
        assert self.x is np.ndarray, f"x grid is not of type np.ndarray, got {type(self.x)} instead."
        assert self.v_ext is np.ndarray, f"v_ext is not of type np.ndarray, got {type(self.v_ext)} instead."
        assert self.v_int is np.ndarray, f"v_int is not of type np.ndarray, got {type(self.v_int)} instead."
        assert self.count is int, f"count is not of type int, got {type(self.count)} instead."
        assert len(self.x.shape) == 1, f"x grid is not a 1D array, got {len(self.x.shape)}D array instead."
        assert len(self.v_ext.shape) == 1, f"v_ext is not a 1D array, got {len(self.v_ext.shape)}D array instead."
        assert len(self.v_int.shape) == 2, f"v_int is not a 2D array, got {len(self.v_int.shape)}D array instead."
        assert (
            self.x.shape == self.v_ext.shape
        ), f"x grid and v_ext arrays are not the same shape, got x.shape = {self.x.shape} and v_ext.shape = {self.v_ext.shape} instead."
        assert (
            self.x.shape[0] == self.v_int.shape[0] and self.x.shape[0] == self.v_int.shape[1]
        ), "v_int is not of the correct shape, got shape {self.v_int.shape} instead."
        assert self.count >= 0, "count is not positive."
        assert set(self.electrons).issubset(
            set(["u", "d"])
        ), f"Electrons must have only up or down spin, e.g 'uudd'. Got {self.electrons} instead"
        assert self.count == self.up_count + self.down_count, "Electrons must obay up_count + down_count = count."
        assert self.stencil in [
            3,
            5,
            7,
            9,
            11,
            13,
        ], f"stencil must be one of [3,5,7,9,11,13], got {self.stencil} instead."

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, value):
        self.__x = value
        self.__dx = self.__x[1] - self.__x[0]
        warnings.warn("x grid has been changed: dx has been recomputed, please update v_ext and v_int on this grid.")

    @x.deleter
    def x(self):
        del self.__x

    @property
    def dx(self):
        return self.__dx

    @dx.setter
    def dx(self, value):
        raise AttributeError("cannot set dx directly: set the x grid and dx will be updated automatically.")

    @dx.deleter
    def dx(self):
        del self.__dx

    @property
    def electrons(self):
        return self.__electrons

    @electrons.setter
    def electrons(self, value):
        self.__electrons = value
        self.count = len(value)
        self.up_count = value.count("u")
        self.down_count = value.count("d")

    @electrons.deleter
    def electrons(self):
        del self.__electrons

    def __str__(self):
        return f"iDEA.system.System: x = np.array([{self.x[0]:.3f},...,{self.x[-1]:.3f}]), dx = {self.dx:.4f}..., v_ext = np.array([{self.v_ext[0]:.3f},...,{self.v_ext[-1]:.3f}]), electrons = {self.electrons}"


def save_system(s: System, file_name: str) -> None:
    r"""
    Save a system to an system file.

    | Args:
    |     system: iDEA.system.System, System object to save.
    |     file_name: str, file name.
    """
    pickle.dump(s, open(file_name, "wb"))


def load_system(file_name: str) -> System:
    r"""
    Load a system from an system file.

    | Args:
    |     file_name: str, file name.

    | Returns
    |     system: iDEA.system.System, Loaded System object.
    """
    return pickle.load(open(file_name, "rb"))


# Define some default built in systems.
__x1 = np.linspace(-10, 10, 300)
systems = iDEA.utilities.Container()
systems.qho = System(
    __x1,
    0.5 * (0.25**2) * (__x1**2),
    iDEA.interactions.softened_interaction(__x1),
    "uu",
)
__x2 = np.linspace(-20, 20, 300)
systems.atom = System(__x2, -2.0 / (abs(__x2) + 1.0), iDEA.interactions.softened_interaction(__x2), "ud")
