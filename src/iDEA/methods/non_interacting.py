"""Contains all non-interacting functionality and solvers."""


import copy
import time
import itertools
from typing import Union
from collections.abc import Callable
import numpy as np
import scipy as sp
import scipy.sparse as sps
import numpy.linalg as npla
import scipy.linalg as spla
import scipy.sparse.linalg as spsla
import iDEA.system
import iDEA.state
import iDEA.observables


# Method name.
name = "non_interacting"


def kinetic_energy_operator(s: iDEA.system.System) -> np.ndarray:
    """
    Compute single-particle kinetic energy operator as a matrix.

    This is built using a given number of finite differences to represent the second derivative.
    The number of differences taken is defined in s.stencil.

    Args:
        s: iDEA.system.System, System object.

    Returns:
        K: np.ndarray, Kintetic energy operator.
    """
    if s.stencil == 3:
        sd = 1.0 * np.array([1, -2, 1], dtype=np.float) / s.dx ** 2
        sdi = (-1, 0, 1)
    elif s.stencil == 5:
        sd = 1.0 / 12.0 * np.array([-1, 16, -30, 16, -1], dtype=np.float) / s.dx ** 2
        sdi = (-2, -1, 0, 1, 2)
    elif s.stencil == 7:
        sd = (1.0 / 180.0 * np.array([2, -27, 270, -490, 270, -27, 2], dtype=np.float) / s.dx ** 2)
        sdi = (-3, -2, -1, 0, 1, 2, 3)
    elif s.stencil == 9:
        sd = (1.0 / 5040.0 * np.array([-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9], dtype=np.float) / s.dx ** 2)
        sdi = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
    elif s.stencil == 11:
        sd = (1.0 / 25200.0 * np.array([8, -125, 1000, -6000, 42000, -73766, 42000, -6000, 1000, -125, 8], dtype=np.float,) / s.dx ** 2)
        sdi = (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
    elif s.stencil == 13:
        sd = (1.0 / 831600.0 * np.array([-50, 864, -7425,44000, -222750, 1425600, -2480478, 1425600, -222750, 44000, -7425, 864, -50], dtype=np.float) / s.dx ** 2)
        sdi = (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6)
    second_derivative = np.zeros((s.x.shape[0], s.x.shape[0]))
    for i in range(len(sdi)):
        second_derivative += np.diag(np.full(np.diag(np.zeros((s.x.shape[0], s.x.shape[0])), k=sdi[i]).shape[0], sd[i]), k=sdi[i])
    K = -0.5 * second_derivative
    return K


def external_potential_operator(s: iDEA.system.System) -> np.ndarray:
    """
    Compute the external potential operator.

    Args;
        s: iDEA.system.System, System object.

    Returns:
        Vext: np.ndarray, External potential energy operator.
    """
    Vext = np.diag(s.v_ext)
    return Vext


def hamiltonian(s: iDEA.system.System, up_n: np.ndarray = None, down_n: np.ndarray = None, up_p: np.ndarray = None, down_p: np.ndarray = None, K: np.ndarray = None, V: np.ndarray = None) -> np.ndarray:
    """
    Compute the Hamiltonian from the kinetic and potential terms.

    Args:
        s: iDEA.system.System, System object.
        up_n: np.ndarray, Charge density of up electrons.
        down_n: np.ndarray, Charge density of down electrons.
        up_p: np.ndarray, Charge density matrix of up electrons.
        down_p: np.ndarray, Charge density matrix of down electrons.
        K: np.ndarray, Single-particle kinetic energy operator [If None this will be computed from s]. (default = None)
        V: np.ndarray, Potential energy operator [If None this will be computed from s]. (default = None)

    Returns:
        H: np.ndarray, Hamiltonian, up Hamiltonian, down Hamiltonian.
    """
    if K is None:
        K = kinetic_energy_operator(s)
    if V is None:
        V = external_potential_operator(s)
    H = K + V
    return H, H, H


def total_energy(s: iDEA.system.System, state: iDEA.state.SingleBodyState) -> float:
    """
    Compute the total energy of a non_interacting state.

    Args:
        s: iDEA.system.System, System object.
        state: iDEA.state.SingleBodyState, State. (default = None)

    Returns:
        E: float, Total energy.
    """
    return iDEA.observables.single_particle_energy(s, state)


def add_occupations(s: iDEA.system.System, state: iDEA.state.SingleBodyState, k: int) -> iDEA.state.SingleBodyState:
    """
    Calculate the occpuations of a state in a given energy excitation.

    Args:
        s: iDEA.system.System, System object.
        state: iDEA.state.SingleBodyState, State.
        k: int, Excitation state [k = 0 is the ground-state]. 

    Returns:
        state: iDEA.state.SingleBodyState, State with occupations added.
    """
    # Calculate the max level or orbitals needed to achieve required state and only use these.
    max_level = (k + 1) * int(np.ceil(s.count))
    up_energies = state.up.energies[:max_level]
    down_energies = state.down.energies[:max_level]

    # Calculate all possible combinations of spin indices.
    up_indices = list(itertools.combinations(range(max_level), s.up_count))
    down_indices = list(itertools.combinations(range(max_level), s.down_count))

    # Construct all possible occupations.
    up_occupations = []
    for up_index in up_indices:
        up_occupation = np.zeros(shape=up_energies.shape, dtype=np.float)
        np.put(up_occupation, up_index, [1.0]*s.up_count)
        up_occupations.append(up_occupation)
    down_occupations = []
    for down_index in down_indices:
        down_occupation = np.zeros(shape=down_energies.shape, dtype=np.float)
        np.put(down_occupation, down_index, [1.0]*s.down_count)
        down_occupations.append(down_occupation)
    occupations = list(itertools.product(up_occupations, down_occupations))

    # Calculate the energies of all possible occpuations.
    energies = []
    for occupation in occupations:
        state_copy = copy.deepcopy(state)
        state_copy.up.occupations = np.zeros(shape=state_copy.up.energies.shape)
        state_copy.up.occupations[:max_level] = occupation[0]
        state_copy.down.occupations = np.zeros(shape=state_copy.down.energies.shape)
        state_copy.down.occupations[:max_level] = occupation[1]
        E = iDEA.observables.single_particle_energy(s, state_copy)
        energies.append(E)
    
    # Choose the correct energy index.
    energy_index = np.argsort(energies)[k]

    # Construct correct occupations.
    state.up.occupations = np.zeros(shape=state_copy.up.energies.shape)
    state.up.occupations[:max_level] = occupations[energy_index][0]
    state.up.occupied = np.nonzero(state.up.occupations)[0]
    state.down.occupations = np.zeros(shape=state_copy.down.energies.shape)
    state.down.occupations[:max_level] = occupations[energy_index][1]
    state.down.occupied = np.nonzero(state.down.occupations)[0]

    return state


def sc_step(s: iDEA.system.System, state: iDEA.state.SingleBodyState, up_H: np.ndarray, down_H: np.ndarray):
    """
    Performs a single step of the self-consistent cycle.

    Args:
        s: iDEA.system.System, System object.
        state: iDEA.state.SingleBodyState, Previous state.
        up_H: np.ndarray, Hamiltonian for up electrons.
        down_H: np.ndarray, Hamiltonian for down electrons.

    Returns:
        state: iDEA.state.SingleBodyState, New state.
    """
    # Solve the non-interacting Schrodinger equation.
    state.up.energies, state.up.orbitals = spla.eigh(up_H)
    state.down.energies, state.down.orbitals = spla.eigh(down_H)

    # Normalise orbitals.
    state.up.orbitals = state.up.orbitals / np.sqrt(s.dx) 
    state.down.orbitals = state.down.orbitals / np.sqrt(s.dx) 

    return state


def solve(s: iDEA.system.System, hamiltonian_function: Callable = None, k: int = 0, restricted: bool = False, mixing: float = 0.5, tol: float = 1e-10) -> iDEA.state.SingleBodyState:
    """
    Solves the Schrodinger equation for the given system.

    Args:
        s: iDEA.system.System, System object.
        hamiltonian_function: Callable, Hamiltonian function [If None this will be the non_interacting function]. (default = None)
        k: int, Energy state to solve for. (default = 0, the ground-state)
        restricted: bool, Is the calculation restricted (r) on unrestricted (u). (default=False)
        mixing: float, Mixing parameter. (default = 0.5)
        tol: float, Tollerance of convergence. (default = 1e-10)

    Returns:
        state: iDEA.state.SingleBodyState, Solved state.
    """
    # Construct the single-body state.
    state = iDEA.state.SingleBodyState()
    state.up.occupations = np.zeros(shape=s.x.shape)
    state.up.occupations[:s.up_count] = 1.0
    state.down.occupations = np.zeros(shape=s.x.shape)
    state.down.occupations[:s.down_count] = 1.0

    # Determine the Hamiltonian function.
    if hamiltonian_function is None:
        hamiltonian_function = hamiltonian

    # Construct the initial values.
    n_old = np.zeros(shape=s.x.shape)
    up_n_old = np.zeros(shape=s.x.shape)
    down_n_old = np.zeros(shape=s.x.shape)
    p_old = np.zeros(shape=s.x.shape*2)
    up_p_old = np.zeros(shape=s.x.shape*2)
    down_p_old = np.zeros(shape=s.x.shape*2)

    # Construct the initial Hamiltonian.
    H, up_H, down_H = hamiltonian_function(s, up_n_old, down_n_old, up_p_old, down_p_old)

    # Apply restriction.
    if restricted:
        up_H = H
        down_H = H

    # Run self-consitent algorithm.
    convergence = 1.0
    while convergence > tol:
    
        # Perform single self-consistent step.
        state = sc_step(s, state, up_H, down_H)

        # Update values.
        n, up_n, down_n = iDEA.observables.density(s, state, return_spins=True)
        p, up_p, down_p = iDEA.observables.density_matrix(s, state, return_spins=True)

        # Construct the new Hamiltonian.
        H, up_H, down_H = hamiltonian_function(s, up_n, down_n, up_p, down_p)

        # Apply restriction.
        if restricted:
            up_H = H
            down_H = H

        # Test for convergence.
        convergence = np.sum(abs(n - n_old)) * s.dx

        # Update old values.
        n_old = n
        up_n_old = up_n
        down_n_old = down_n
        p_old = p
        up_p_old = up_p
        down_p_old = down_p

        # Update terminal.
        print(r"iDEA.methods.method.solve: convergence = {0:.5}, tollerance = {1:.5}".format(convergence, tol), end="\r")
    
    # Compute state occupations.
    state = add_occupations(s, state, k)
    print()

    return state


def propagate(s: iDEA.system.System, state: iDEA.state.SingleBodyState, v_ptrb: np.ndarray, t: np.ndarray, H: np.ndarray = None) -> iDEA.state.SingleBodyEvolution:
    """
    Propagate a set of orbitals forward in time due to a local pertubation.

    Args: 
        s: iDEA.system.System, System object.
        state: iDEA.state.SingleBodyState, State to be propigated.
        v_ptrb: np.ndarray, Local perturbing potential [static or dynamic].
        t: np.ndarray, Grid of time values.
        H: np.ndarray, Static Hamiltonian [If None this will be computed from s]. (default = None)

    Returns:
        evolution: iDEA.state.SingleBodyEvolution, Solved time-dependent evolution.
    """
    if len(v_ptrb.shape) == 1:
        return _propagate_static(s, state, v_ptrb, t, H)
    elif len(v_ptrb.shape) == 2:
        return _propagate_dynamic(s, state, v_ptrb, t, H)
    else:
        raise AttributeError(f"v_ptrb must have shape 1 or 2, got {v_ptrb.shape} instead.")


def _propagate_static(s: iDEA.system.System, state: iDEA.state.SingleBodyState, v_ptrb: np.ndarray, t: np.ndarray, H: np.ndarray = None) -> iDEA.state.SingleBodyEvolution:
    """
    Propagate a set of orbitals forward in time due to a static local pertubation.

    Args: 
        s: iDEA.system.System, System object.
        state: iDEA.state.SingleBodyState, State to be propigated.
        v_ptrb: np.ndarray, Local perturbing potential on the grid of x values, indexed as v_ptrb[space].
        t: np.ndarray, Grid of time values.
        H: np.ndarray, Static Hamiltonian [If None this will be computed from s]. (default = None)

    Returns:
        evolution: iDEA.state.SingleBodyEvolution, Solved time-dependent evolution.
    """
    # Construct the unperturbed Hamiltonian.
    if H is None:
        H = hamiltonian(s)

    # Construct the pertubation potential.
    Vptrb = np.diag(v_ptrb)

    # Compute timestep.
    dt = t[1] - t[0]

    # Construct time propigation operator.
    U = spla.expm(-1.0j * (H + Vptrb) * dt)

    # Initilise time-dependent orbitals.
    td_up_orbitals = np.zeros(shape=(t.shape[0], s.x.shape[0], state.up.occupied.shape[0]), dtype=np.complex)
    td_down_orbitals = np.zeros(shape=(t.shape[0], s.x.shape[0], state.down.occupied.shape[0]), dtype=np.complex)
    td_up_orbitals[0, :, :] = state.up.orbitals[:, state.up.occupied]
    td_down_orbitals[0, :, :] = state.down.orbitals[:, state.down.occupied]

    # Propagate up orbitals.
    for i in range(state.up.occupied.shape[0]):
        for j, ti in enumerate(t):
            if j != 0:
                print("iDEA.methods.non_interacting.propagate: propagating up orbital {0}/{1}, time = {2:.3f}/{3:.3f}".format(i + 1, s.up_count, ti, np.max(t)), end="\r")
                td_up_orbitals[j, :, i] = np.dot(U, td_up_orbitals[j - 1, :, i])
                norm = npla.norm(td_up_orbitals[j, :, i]) * np.sqrt(s.dx)
                td_up_orbitals[j, :, i] /= norm
        print()

    # Propagate down orbitals.
    for i in range(state.down.occupied.shape[0]):
        for j, ti in enumerate(t):
            if j != 0:
                print("iDEA.methods.non_interacting.propagate: propagating down orbital {0}/{1}, time = {2:.3f}/{3:.3f}".format(i + 1, s.down_count, ti, np.max(t)), end="\r")
                td_down_orbitals[j, :, i] = np.dot(U, td_down_orbitals[j - 1, :, i])
                norm = npla.norm(td_down_orbitals[j, :, i]) * np.sqrt(s.dx)
                td_down_orbitals[j, :, i] /= norm
        print()

    # Construct the single-body time-dependent evolution.
    evolution = iDEA.state.SingleBodyEvolution(state)
    evolution.up.td_orbitals = td_up_orbitals
    evolution.down.td_orbitals = td_down_orbitals
    evolution.v_ptrb = v_ptrb
    evolution.t = t

    return evolution


def _propagate_dynamic(s: iDEA.system.System, state: iDEA.state.SingleBodyState, v_ptrb: np.ndarray, t: np.ndarray, H: np.ndarray = None) -> iDEA.state.SingleBodyEvolution:
    """
    Propagate a set of orbitals forward in time due to a dynamic local pertubation.

    Args: 
        s: iDEA.system.System, System object.
        state: iDEA.state.SingleBodyState, State to be propigated.
        v_ptrb: np.ndarray, Local perturbing potential on the grid of t and x values, indexed as v_ptrb[time,space].
        t: np.ndarray, Grid of time values.
        H: np.ndarray, Static Hamiltonian [If None this will be computed from s]. (default = None)

    Returns:
        evolution: iDEA.state.SingleBodyEvolution, Solved time-dependent evolution.
    """
    # Construct the unperturbed Hamiltonian.
    if H is None:
        H = hamiltonian(s)
    H = sps.csc_matrix(H)

    # Compute timestep.
    dt = t[1] - t[0]

    # Initilise time-dependent orbitals.
    td_up_orbitals = np.zeros(shape=(t.shape[0], s.x.shape[0], state.up.occupied.shape[0]), dtype=np.complex)
    td_down_orbitals = np.zeros(shape=(t.shape[0], s.x.shape[0], state.down.occupied.shape[0]), dtype=np.complex)
    td_up_orbitals[0, :, :] = state.up.orbitals[:, state.up.occupied]
    td_down_orbitals[0, :, :] = state.down.orbitals[:, state.down.occupied]

    # Propagate up orbitals.
    for i in range(state.up.occupied.shape[0]):
        for j, ti in enumerate(t):
            if j != 0:
                print("iDEA.methods.non_interacting.propagate: propagating up orbital {0}/{1}, time = {2:.3f}/{3:.3f}".format(i + 1, s.up_count, ti, np.max(t)), end="\r")
                Vptrb = sps.diags(v_ptrb[j,:]).tocsc()
                td_up_orbitals[j, :, i] = spsla.expm_multiply(spla.expm(-1.0j * (H + Vptrb) * dt), td_up_orbitals[j - 1, :, i])
                norm = npla.norm(td_up_orbitals[j, :, i]) * np.sqrt(s.dx)
                td_up_orbitals[j, :, i] /= norm
        print()

    # Propagate down orbitals.
    for i in range(state.down.occupied.shape[0]):
        for j, ti in enumerate(t):
            if j != 0:
                print("iDEA.methods.non_interacting.propagate: propagating down orbital {0}/{1}, time = {2:.3f}/{3:.3f}".format(i + 1, s.down_count, ti, np.max(t)), end="\r")
                Vptrb = sps.diags(v_ptrb[j,:]).tocsc()
                td_down_orbitals[j, :, i] = spsla.expm_multiply(spla.expm(-1.0j * (H + Vptrb) * dt), td_down_orbitals[j - 1, :, i])
                norm = npla.norm(td_down_orbitals[j, :, i]) * np.sqrt(s.dx)
                td_down_orbitals[j, :, i] /= norm
        print()

    # Construct the single-body time-dependent evolution.
    evolution = iDEA.state.SingleBodyEvolution(state)
    evolution.up.td_orbitals = td_up_orbitals
    evolution.down.td_orbitals = td_down_orbitals
    evolution.v_ptrb = v_ptrb
    evolution.t = t

    return evolution