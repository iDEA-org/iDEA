"""Contains all reverse-engineering functionality."""


import copy
import warnings
from collections.abc import Container
from tqdm import tqdm
import numpy as np
import scipy.optimize as spo
import scipy.sparse as sps
import iDEA.system
import iDEA.state
import iDEA.observables


def reverse(
    s: iDEA.system.System,
    target_n: np.ndarray,
    method: Container,
    v_guess: np.ndarray = None,
    mu: float = 1.0,
    pe: float = 0.1,
    tol: float = 1e-12,
    silent: bool = False,
    **kwargs
) -> iDEA.state.State:
    r"""
    Determines what ficticious system is needed for a given method, when solving the system, to produce a given target density.
    If the given target density is from solving the interacting electron problem (iDEA.methods.interacting), and the method is the non-interacting electron solver (iDEA.methods.non_interacting)
    the output is the Kohn-Sham system.

    The iterative method used is defined by the following formula:
    .. math:: \mathrm{V}_\mathrm{ext} \rightarrow \mu * (\mathrm{n}^p - \mathrm{target_n}^p)

    | Args:
    |     s: iDEA.system.System, System object.
    |     target_n: np.ndarray, Target density to reverse engineer.
    |     method: Container, The method used to solve the system.
    |     v_guess: np.ndarray, The initial guess of the fictitious potential. (default = None)
    |     mu: float = 1.0, Reverse engineering parameter mu. (default = 1.0)
    |     pe: float = 0.1, Reverse engineering parameter p. (default = 0.1)
    |     tol: float, Tollerance of convergence. (default = 1e-12)
    |     silent: bool, Set to true to prevent printing. (default = False)
    |     kwargs: Other arguments that will be given to the method's solve function.

    | Returns:
    |     s_fictitious: iDEA.system.System, fictitious system object.
    """
    s_fictitious = copy.deepcopy(s)
    if v_guess is not None:
        s_fictitious.v_ext = v_guess
    n = np.zeros(shape=s.x.shape)
    up_n = np.zeros(shape=s.x.shape)
    down_n = np.zeros(shape=s.x.shape)
    p = np.zeros(shape=s.x.shape * 2)
    up_p = np.zeros(shape=s.x.shape * 2)
    down_p = np.zeros(shape=s.x.shape * 2)
    convergence = 1.0
    while convergence > tol:
        if silent is False:
            print(
                r"iDEA.reverse_engineering.reverse: convergence = {0:.5}, tolerance = {1:.5}".format(
                    convergence, tol
                ),
                end="\r",
            )
        state = method.solve(
            s_fictitious,
            initial=(n, up_n, down_n, p, up_p, down_p),
            silent=True,
            **kwargs
        )
        n, up_n, down_n = iDEA.observables.density(
            s_fictitious, state=state, return_spins=True
        )
        p, up_p, down_p = iDEA.observables.density_matrix(
            s_fictitious, state=state, return_spins=True
        )
        s_fictitious.v_ext += mu * (n**pe - target_n**pe)
        convergence = np.sum(abs(n - target_n)) * s.dx
    if silent is False:
        print()
    return s_fictitious


def _residual(
    v: np.ndarray,
    s_fictitious: iDEA.system.System,
    evolution_fictitious: iDEA.state.Evolution,
    j: int,
    method: Container,
    v_ptrb: np.ndarray,
    dt: float,
    restricted: bool,
    target_n: np.ndarray,
):
    r"""
    The residual function used to optimise each time step of the time dependent reverse propagation.

    | Args:
    |     v: iDEA.system.System, Potential adjusted during optimisation.
    |     s_fictitious: iDEA.system.System, Fictitious system.
    |     evolution_fictitious: iDEA.system.Evolution, Fictitious evolution.
    |     j: int float = 1.0, Time index.
    |     method: Container: float = 0.1, The method used to solve the system.
    |     v_ptrb: np.ndarray, Local perturbing potential on the grid of t and x values, indexed as v_ptrb[time,space].
    |     dt: float, bool, Timestep.
    |     restricted: bool, Is the calculation restricted (r) on unrestricted (u).
    |     target_n: np.ndarray, Target density.

    | Returns:
    |     residual: np.ndarray, Error in propagation to be minimised.
    """
    v_td = np.zeros_like(v_ptrb)
    v_td[j, :] = v[:]
    evolution = method.propagate_step(
        s_fictitious, evolution_fictitious, j, method.hamiltonian, v_td, dt, restricted
    )
    n = iDEA.observables.density(
        s_fictitious,
        evolution=evolution,
        time_indices=np.array([j]),
        return_spins=False,
    )[0, :]
    residual = n - target_n
    return residual


def reverse_propagation(
    s_fictitious: iDEA.system.System,
    state_fictitious: iDEA.state.State,
    target_n: np.ndarray,
    method: Container,
    v_ptrb: np.ndarray,
    t: np.ndarray,
    restricted: bool = False,
    tol: float = 1e-10,
    **kwargs
) -> iDEA.state.Evolution:
    r"""
    Determines what ficticious evolution is needed for a given method, when solving the system, to produce a given time dependent target density.
    If the given target density is from solving the interacting electron problem (iDEA.methods.interacting), and the method is the non-interacting electron solver (iDEA.methods.non_interacting)
    the output is the Kohn-Sham system.

    | Args:
    |     s_fictitious: iDEA.system.System, System object.
    |     state_fictitious: iDEA.state.State, Fictitious initial state.
    |     target_n: np.ndarray, Target density to reverse engineer.
    |     method: Container, The method used to solve the system.
    |     v_ptrb: np.ndarray, Local perturbing potential on the grid of t and x values, indexed as v_ptrb[time,space].
    |     t: np.ndarray, Grid of time values.
    |     restricted: bool, Is the calculation restricted (r) on unrestricted (u). (default = False)
    |     tol: float, Tollerance of convergence. (default = 1e-10)
    |     kwargs: Other arguments that will be given to the method's solve function.

    | Returns:
    |     evolution_fictitious, error: iDEA.system.Evolution, fictitious evolution object along with time dependent error.
    """
    # Determine the Hamiltonian function.
    hamiltonian_function = method.hamiltonian

    # Construct the unperturbed Hamiltonian.
    n, up_n, down_n = iDEA.observables.density(
        s_fictitious, state=state_fictitious, return_spins=True
    )
    p, up_p, down_p = iDEA.observables.density_matrix(
        s_fictitious, state=state_fictitious, return_spins=True
    )
    H, up_H, down_H = hamiltonian_function(
        s_fictitious, up_n, down_n, up_p, down_p, **kwargs
    )
    H = sps.csc_matrix(H)
    up_H = sps.csc_matrix(up_H)
    down_H = sps.csc_matrix(down_H)
    down_H += sps.spdiags(
        1e-12 * s_fictitious.x,
        np.array([0]),
        s_fictitious.x.shape[0],
        s_fictitious.x.shape[0],
    )

    # Apply restriction.
    if restricted:
        up_H = H
        down_H = H

    # Compute timestep.
    dt = t[1] - t[0]

    # Initilise the single-body time-dependent evolution.
    evolution_fictitious = iDEA.state.SingleBodyEvolution(state_fictitious)
    evolution_fictitious.up.td_orbitals = np.zeros(
        shape=(
            t.shape[0],
            s_fictitious.x.shape[0],
            state_fictitious.up.occupied.shape[0],
        ),
        dtype=np.complex,
    )
    evolution_fictitious.down.td_orbitals = np.zeros(
        shape=(
            t.shape[0],
            s_fictitious.x.shape[0],
            state_fictitious.down.occupied.shape[0],
        ),
        dtype=np.complex,
    )
    evolution_fictitious.up.td_orbitals[0, :, :] = state_fictitious.up.orbitals[
        :, state_fictitious.up.occupied
    ]
    evolution_fictitious.down.td_orbitals[0, :, :] = state_fictitious.down.orbitals[
        :, state_fictitious.down.occupied
    ]
    evolution_fictitious.v_ptrb = copy.deepcopy(v_ptrb)
    evolution_fictitious.t = copy.deepcopy(t)

    # Initialise error.
    error = np.zeros_like(t)

    # Reverse propagation.
    for j, ti in enumerate(
        tqdm(t, desc="iDEA.reverse_engineering.reverse_propagation: reversing propagation")
    ):
        if j != 0:

            # Determine ficticious perturbing potential.
            v_guess = np.zeros_like(evolution_fictitious.v_ptrb[j, :])
            result = spo.root(
                _residual,
                v_guess,
                args=(
                    s_fictitious,
                    evolution_fictitious,
                    j,
                    method,
                    v_ptrb,
                    dt,
                    restricted,
                    target_n[j, :],
                ),
                method="hybr",
                tol=tol,
                options={"maxfev": 200},
            )
            evolution_fictitious.v_ptrb[j, :] = result.x

            # Perform propagation.
            evolution_fictitious = method.propagate_step(
                s_fictitious,
                evolution_fictitious,
                j,
                method.hamiltonian,
                evolution_fictitious.v_ptrb,
                dt,
                restricted,
            )
            n = iDEA.observables.density(
                s_fictitious,
                evolution=evolution_fictitious,
                time_indices=np.array([j]),
                return_spins=False,
            )[0]

            # Compute mae.
            error[j] = np.mean(np.abs(n - target_n[j, :]))

    return evolution_fictitious, error
