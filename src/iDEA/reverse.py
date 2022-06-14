"""Contains all reverse-engineering functionality."""


import copy
import time
import itertools
from tqdm import tqdm
from typing import Union
from collections.abc import Callable, Container
import numpy as np
import scipy as sp
import scipy.sparse as sps
import numpy.linalg as npla
import scipy.linalg as spla
import scipy.sparse.linalg as spsla
import iDEA.system
import iDEA.state
import iDEA.observables
import matplotlib.pyplot as plt


# Reverse function
def reverse(s: iDEA.state.State, target_n: np.ndarray, method: Container, v_guess: np.ndarray = None, mu: float = 1.0, pe: float = 0.1, tol: float = 1e-12, silent: bool = False, **kwargs):
    r"""
    Determines what ficticious external potential is needed for a given method, when solving the system, to produce a given target density.
    If the given target density is from solving the interacting electron problem (iDEA.methods.interacting), and the method is the non-interacting electron solver (iDEA.methods.non_interacting)
    the output is the Kohn-Sham potential.

    The iterative method used is defined by the following formula:
    V_\mathrm{ext} \rightarrow \mu * (\mathrm{n}^p - \mathrm{target_n}^p)

    Args:
        s: iDEA.system.System, System object.
        target_n: np.ndarray, Target density to reverse engineer.
        method: Container, The method used to solve the system.
        v_guess: np.ndarray, The initial guess of the fictious potential. (default = None)
        mu: float = 1.0, Reverse engineering parameter mu. (default = 1.0)
        pe: float = 0.1, Reverse engineering parameter p. (default = 0.1)
        tol: float, Tollerance of convergence. (default = 1e-12)
        silent: bool, Set to true to prevent printing. (default = False)
        kwargs: Other arguments that will be given to the method's solve function.

    Returns:
        s_fictious.v_ext: iDEA.state.SingleBodyState, Solved state.
    """
    s_fictious = copy.deepcopy(s)
    if v_guess is not None:
        s_fictious.v_ext = v_guess
    n = np.zeros(shape=s.x.shape)
    up_n = np.zeros(shape=s.x.shape)
    down_n = np.zeros(shape=s.x.shape)
    p = np.zeros(shape=s.x.shape*2)
    up_p = np.zeros(shape=s.x.shape*2)
    down_p = np.zeros(shape=s.x.shape*2)
    convergence = 1.0
    while convergence > tol:
        if silent is False:
            print(r"iDEA.reverse.reverse: convergence = {0:.5}, tollerance = {1:.5}".format(convergence, tol), end="\r")
        state = method.solve(s_fictious, initial=(n, up_n, down_n, p, up_p, down_p), silent=True, **kwargs)
        n, up_n, down_n = iDEA.observables.density(s_fictious, state=state, return_spins=True)
        p, up_p, down_p = iDEA.observables.density_matrix(s_fictious, state=state, return_spins=True)
        s_fictious.v_ext += mu * (n**pe - target_n**pe)
        convergence = np.sum(abs(n - target_n))*s.dx
    if silent is False:
        print()
    return s_fictious.v_ext

