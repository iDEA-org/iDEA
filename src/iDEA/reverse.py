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


def reverse(s: iDEA.system.System, target_n: np.ndarray, method: Container, v_guess: np.ndarray = None, mu: float = 1.0, pe: float = 0.1, tol: float = 1e-12, silent: bool = False, **kwargs) -> iDEA.state.State:
    r"""
    Determines what ficticious system is needed for a given method, when solving the system, to produce a given target density.
    If the given target density is from solving the interacting electron problem (iDEA.methods.interacting), and the method is the non-interacting electron solver (iDEA.methods.non_interacting)
    the output is the Kohn-Sham system.

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
        s_fictious: iDEA.system.System, Fictious system object.
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
    return s_fictious


def _residual(v, s_fictious, evolution_fictious, j, method, v_ptrb, dt, restricted, target_n):
    v_td = np.zeros_like(v_ptrb)
    v_td[j,:] = v[:]
    evolution = method.propagate_step(s_fictious, evolution_fictious, j, method.hamiltonian, v_td, dt, restricted)
    n = iDEA.observables.density(s_fictious, evolution=evolution, time_indices=np.array([j]), return_spins=False)[0,:]
    residual = n - target_n
    return residual


def reverse_propigation(s_fictious, state_fictious, target_n, method, v_ptrb, t, restricted=False, tol=1e-10, silent=False, **kwargs) -> iDEA.state.Evolution:
    # Determine the Hamiltonian function.
    hamiltonian_function = method.hamiltonian

    # Construct the unperturbed Hamiltonian.
    n, up_n, down_n = iDEA.observables.density(s_fictious, state=state_fictious, return_spins=True)
    p, up_p, down_p = iDEA.observables.density_matrix(s_fictious, state=state_fictious, return_spins=True)
    H, up_H, down_H = hamiltonian_function(s_fictious, up_n, down_n, up_p, down_p, **kwargs)
    H = sps.csc_matrix(H)
    up_H = sps.csc_matrix(up_H)
    down_H = sps.csc_matrix(down_H)
    down_H += sps.spdiags(1e-12*s_fictious.x, np.array([0]), s_fictious.x.shape[0], s_fictious.x.shape[0])

    # Apply restriction.
    if restricted:
        up_H = H
        down_H = H

    # Compute timestep.
    dt = t[1] - t[0]

    # Initilise the single-body time-dependent evolution.
    evolution_fictious = iDEA.state.SingleBodyEvolution(state_fictious)
    evolution_fictious.up.td_orbitals = np.zeros(shape=(t.shape[0], s_fictious.x.shape[0], state_fictious.up.occupied.shape[0]), dtype=np.complex)
    evolution_fictious.down.td_orbitals = np.zeros(shape=(t.shape[0], s_fictious.x.shape[0], state_fictious.down.occupied.shape[0]), dtype=np.complex)
    evolution_fictious.up.td_orbitals[0, :, :] = state_fictious.up.orbitals[:, state_fictious.up.occupied]
    evolution_fictious.down.td_orbitals[0, :, :] = state_fictious.down.orbitals[:, state_fictious.down.occupied]
    evolution_fictious.v_ptrb = v_ptrb
    evolution_fictious.t = t

    # Initialise error.
    error = np.zeros_like(t)

    # Reverse propigation.
    for j, ti in enumerate(tqdm(t, desc="iDEA.reverse.reverse_propigation: reversing propagation")):
        if j != 0:

            # Determine ficticious perturbing potential.
            v_guess = evolution_fictious.v_ptrb[j,:]
            result = spo.root(_residual, v_guess, args=(s_fictious, evolution_fictious, j, method, v_ptrb, dt, restricted, target_n[j,:]), method='hybr', tol=tol) 
            if result.success == False:
                    warnings.warn('iDEA.reverse.reverse_propigation: continuing after error in root')
            evolution_fictious.v_ptrb[j,:] = result.x
            
            # Perform propigation.
            evolution_fictious = method.propagate_step(s_fictious, evolution_fictious, j, method.hamiltonian, evolution_fictious.v_ptrb, dt, restricted)
            n = iDEA.observables.density(s_fictious, evolution=evolution_fictious, time_indices=np.array([j]), return_spins=False)[0]

            # Compute mae.
            error[j] = np.mean(np.abs(n - target_n[j,:]))

    return evolution_fictious, error