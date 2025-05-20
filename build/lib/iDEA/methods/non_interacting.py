"""Contains all non-interacting functionality and solvers."""

import copy
import itertools
from collections.abc import Callable

import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from tqdm import tqdm

import iDEA.observables
import iDEA.state
import iDEA.system

name = "non_interacting"


def kinetic_energy_operator(s: iDEA.system.System) -> np.ndarray:
    r"""
    Compute single-particle kinetic energy operator as a matrix.

    This is built using a given number of finite differences to represent the second derivative.
    The number of differences taken is defined in s.stencil.

    | Args:
    |     s: iDEA.system.System, System object.

    | Returns:
    |     K: np.ndarray, Kintetic energy operator.
    """
    if s.stencil == 3:
        sd = 1.0 * np.array([1, -2, 1], dtype=np.float) / s.dx**2
        sdi = (-1, 0, 1)
    elif s.stencil == 5:
        sd = 1.0 / 12.0 * np.array([-1, 16, -30, 16, -1], dtype=np.float) / s.dx**2
        sdi = (-2, -1, 0, 1, 2)
    elif s.stencil == 7:
        sd = 1.0 / 180.0 * np.array([2, -27, 270, -490, 270, -27, 2], dtype=np.float) / s.dx**2
        sdi = (-3, -2, -1, 0, 1, 2, 3)
    elif s.stencil == 9:
        sd = 1.0 / 5040.0 * np.array([-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9], dtype=np.float) / s.dx**2
        sdi = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
    elif s.stencil == 11:
        sd = (
            1.0
            / 25200.0
            * np.array(
                [8, -125, 1000, -6000, 42000, -73766, 42000, -6000, 1000, -125, 8],
                dtype=np.float,
            )
            / s.dx**2
        )
        sdi = (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
    elif s.stencil == 13:
        sd = (
            1.0
            / 831600.0
            * np.array(
                [
                    -50,
                    864,
                    -7425,
                    44000,
                    -222750,
                    1425600,
                    -2480478,
                    1425600,
                    -222750,
                    44000,
                    -7425,
                    864,
                    -50,
                ],
                dtype=np.float,
            )
            / s.dx**2
        )
        sdi = (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6)
    second_derivative = np.zeros((s.x.shape[0], s.x.shape[0]))
    for i in range(len(sdi)):
        second_derivative += np.diag(
            np.full(
                np.diag(np.zeros((s.x.shape[0], s.x.shape[0])), k=sdi[i]).shape[0],
                sd[i],
            ),
            k=sdi[i],
        )
    K = -0.5 * second_derivative
    return K


def external_potential_operator(s: iDEA.system.System) -> np.ndarray:
    r"""
    Compute the external potential operator.

    | Args:
    |     s: iDEA.system.System, System object.

    | Returns:
    |     Vext: np.ndarray, External potential energy operator.
    """
    Vext = np.diag(s.v_ext)
    return Vext


def hamiltonian(
    s: iDEA.system.System,
    up_n: np.ndarray = None,
    down_n: np.ndarray = None,
    up_p: np.ndarray = None,
    down_p: np.ndarray = None,
    K: np.ndarray = None,
    Vext: np.ndarray = None,
) -> np.ndarray:
    r"""
    Compute the Hamiltonian from the kinetic and potential terms.

    | Args:
    |     s: iDEA.system.System, System object.
    |     up_n: np.ndarray, Charge density of up electrons.
    |     down_n: np.ndarray, Charge density of down electrons.
    |     up_p: np.ndarray, Charge density matrix of up electrons.
    |     down_p: np.ndarray, Charge density matrix of down electrons.
    |     K: np.ndarray, Single-particle kinetic energy operator [If None this will be computed from s]. (default = None)
    |     Vext: np.ndarray, Potential energy operator [If None this will be computed from s]. (default = None)

    | Returns:
    |     H: np.ndarray, Hamiltonian, up Hamiltonian, down Hamiltonian.
    """
    if K is None:
        K = kinetic_energy_operator(s)
    if Vext is None:
        Vext = external_potential_operator(s)
    H = K + Vext
    return H, H, H


def total_energy(s: iDEA.system.System, state: iDEA.state.SingleBodyState) -> float:
    r"""
    Compute the total energy of a non_interacting state.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.SingleBodyState, State. (default = None)

    | Returns:
    |     E: float, Total energy.
    """
    return iDEA.observables.single_particle_energy(s, state)


def add_occupations(s: iDEA.system.System, state: iDEA.state.SingleBodyState, k: int) -> iDEA.state.SingleBodyState:
    r"""
    Calculate the occpuations of a state in a given energy excitation.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.SingleBodyState, State.
    |     k: int, Excitation state [k = 0 is the ground-state].

    | Returns:
    |     state: iDEA.state.SingleBodyState, State with occupations added.
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
        np.put(up_occupation, up_index, [1.0] * s.up_count)
        up_occupations.append(up_occupation)
    down_occupations = []
    for down_index in down_indices:
        down_occupation = np.zeros(shape=down_energies.shape, dtype=np.float)
        np.put(down_occupation, down_index, [1.0] * s.down_count)
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


def sc_step(
    s: iDEA.system.System,
    state: iDEA.state.SingleBodyState,
    up_H: np.ndarray,
    down_H: np.ndarray,
):
    r"""
    Performs a single step of the self-consistent cycle.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.SingleBodyState, Previous state.
    |     up_H: np.ndarray, Hamiltonian for up electrons.
    |     down_H: np.ndarray, Hamiltonian for down electrons.

    | Returns:
    |     state: iDEA.state.SingleBodyState, New state.
    """
    # Solve the non-interacting Schrodinger equation.
    state.up.energies, state.up.orbitals = spla.eigh(up_H)
    state.down.energies, state.down.orbitals = spla.eigh(down_H)

    # Normalise orbitals.
    state.up.orbitals = state.up.orbitals / np.sqrt(s.dx)
    state.down.orbitals = state.down.orbitals / np.sqrt(s.dx)

    return state


def solve(
    s: iDEA.system.System,
    hamiltonian_function: Callable = None,
    k: int = 0,
    restricted: bool = False,
    mixing: float = 0.5,
    tol: float = 1e-10,
    initial: tuple = None,
    name: str = "non_interacting",
    silent: bool = False,
    **kwargs,
) -> iDEA.state.SingleBodyState:
    r"""
    Solves the Schrodinger equation for the given system.

    | Args:
    |     s: iDEA.system.System, System object.
    |     hamiltonian_function: Callable, Hamiltonian function [If None this will be the non_interacting function]. (default = None)
    |     k: int, Energy state to solve for. (default = 0, the ground-state)
    |     restricted: bool, Is the calculation restricted (r) on unrestricted (u). (default=False)
    |     mixing: float, Mixing parameter. (default = 0.5)
    |     tol: float, Tollerance of convergence. (default = 1e-10)
    |     initial: tuple. Tuple of initial values used to begin the self-consistency (n, up_n, down_n, p, up_p, down_p). (default = None)
    |     name: str, Name of method. (default = "non_interacting")
    |     silent: bool, Set to true to prevent printing. (default = False)


    | Returns:
    |     state: iDEA.state.SingleBodyState, Solved state.
    """
    # Construct the single-body state.
    state = iDEA.state.SingleBodyState()
    state.up.occupations = np.zeros(shape=s.x.shape)
    state.up.occupations[: s.up_count] = 1.0
    state.down.occupations = np.zeros(shape=s.x.shape)
    state.down.occupations[: s.down_count] = 1.0

    # Determine the Hamiltonian function.
    if hamiltonian_function is None:
        hamiltonian_function = hamiltonian

    # Construct the initial values.
    if initial is None:
        n_old = np.zeros(shape=s.x.shape)
        up_n_old = np.zeros(shape=s.x.shape)
        down_n_old = np.zeros(shape=s.x.shape)
        p_old = np.zeros(shape=s.x.shape * 2)
        up_p_old = np.zeros(shape=s.x.shape * 2)
        down_p_old = np.zeros(shape=s.x.shape * 2)
    else:
        n_old = initial[0]
        up_n_old = initial[1]
        down_n_old = initial[2]
        p_old = initial[3]
        up_p_old = initial[4]
        down_p_old = initial[5]

    # Construct the initial Hamiltonian. (And break the symmetry.)
    H_old, up_H_old, down_H_old = hamiltonian_function(s, up_n_old, down_n_old, up_p_old, down_p_old, **kwargs)
    H, up_H, down_H = hamiltonian_function(s, up_n_old, down_n_old, up_p_old, down_p_old, **kwargs)
    down_H += sps.spdiags(1e-12 * s.x, np.array([0]), s.x.shape[0], s.x.shape[0]).toarray()

    # Apply restriction.
    if restricted:
        up_H_old = H_old
        down_H_old = H_old
        up_H = H
        down_H = H

    # Run self-consitent algorithm.
    convergence = 1.0
    count = 0
    while convergence > tol:
        # Perform single self-consistent step.
        state = sc_step(s, state, up_H, down_H)

        # Update values.
        n, up_n, down_n = iDEA.observables.density(s, state, return_spins=True)
        p, up_p, down_p = iDEA.observables.density_matrix(s, state, return_spins=True)

        # Perform mixing.
        if count != 0:
            n = mixing * n + (1.0 - mixing) * n_old
            up_n = mixing * up_n + (1.0 - mixing) * up_n_old
            down_n = mixing * down_n + (1.0 - mixing) * down_n_old
            p = mixing * p + (1.0 - mixing) * p_old
            up_p = mixing * up_p + (1.0 - mixing) * up_p_old
            down_p = mixing * down_p + (1.0 - mixing) * down_p_old

        # Construct the new Hamiltonian.
        H, up_H, down_H = hamiltonian_function(s, up_n, down_n, up_p, down_p, **kwargs)

        # Apply restriction.
        if restricted:
            up_H_old = H_old
            down_H_old = H_old
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
        count += 1
        if silent is False:
            print(
                rf"iDEA.methods.{name}.solve: convergence = {convergence:.5}, tolerance = {tol:.5}",
                end="\r",
            )

    # Compute state occupations.
    state = add_occupations(s, state, k)
    if silent is False:
        print()

    return state


def propagate_step(
    s: iDEA.system.System,
    evolution: iDEA.state.SingleBodyEvolution,
    j: int,
    hamiltonian_function: Callable,
    v_ptrb: np.ndarray,
    dt: float,
    restricted: bool,
    **kwargs,
):
    r"""
    Propagate a set of orbitals forward in time due to a dynamic local pertubation.

    | Args:
    |     s: iDEA.system.System, System object.
    |     evolution: iDEA.state.SingleBodyEvolution, Time-dependent evolution.
    |     j: int, Time index to step to.
    |     hamiltonian_function: Callable, Hamiltonian function [If None this will be the non_interacting function]. (default = None)
    |     v_ptrb: np.ndarray, Local perturbing potential on the grid of t and x values, indexed as v_ptrb[time,space].
    |     dt: float, Timestep.
    |     restricted: bool, Is the calculation restricted (r) on unrestricted (u). (default=False)

    | Returns:
    |     evolution: iDEA.state.SingleBodyEvolution, Time-dependent evolution solved at time index j from j-1.
    """
    n, up_n, down_n = iDEA.observables.density(
        s, evolution=evolution, time_indices=np.array([j - 1]), return_spins=True
    )
    p, up_p, down_p = iDEA.observables.density_matrix(
        s, evolution=evolution, time_indices=np.array([j - 1]), return_spins=True
    )
    H, up_H, down_H = hamiltonian_function(s, up_n[0, ...], down_n[0, ...], up_p[0, ...], down_p[0, ...], **kwargs)
    H = sps.csc_matrix(H)
    up_H = sps.csc_matrix(up_H)
    down_H = sps.csc_matrix(down_H)
    Vptrb = sps.diags(v_ptrb[j, :]).tocsc()

    # Apply restriction.
    if restricted:
        up_H = H
        down_H = H

    for i in range(evolution.up.occupied.shape[0]):
        up_O = -1.0j * (up_H + Vptrb) * dt
        evolution.up.td_orbitals[j, :, i] = spsla.expm_multiply(up_O, evolution.up.td_orbitals[j - 1, :, i])
        norm = npla.norm(evolution.up.td_orbitals[j, :, i]) * np.sqrt(s.dx)
        evolution.up.td_orbitals[j, :, i] /= norm
    for i in range(evolution.down.occupied.shape[0]):
        down_O = -1.0j * (down_H + Vptrb) * dt
        evolution.down.td_orbitals[j, :, i] = spsla.expm_multiply(down_O, evolution.down.td_orbitals[j - 1, :, i])
        norm = npla.norm(evolution.down.td_orbitals[j, :, i]) * np.sqrt(s.dx)
        evolution.down.td_orbitals[j, :, i] /= norm

    return evolution


def propagate(
    s: iDEA.system.System,
    state: iDEA.state.SingleBodyState,
    v_ptrb: np.ndarray,
    t: np.ndarray,
    hamiltonian_function: Callable = None,
    restricted: bool = False,
    name: str = "non_interacting",
    **kwargs,
) -> iDEA.state.SingleBodyEvolution:
    r"""
    Propagate a set of orbitals forward in time due to a dynamic local pertubation.

    | Args:
    |    s: iDEA.system.System, System object.
    |    state: iDEA.state.SingleBodyState, State to be propigated.
    |    v_ptrb: np.ndarray, Local perturbing potential on the grid of t and x values, indexed as v_ptrb[time,space].
    |    t: np.ndarray, Grid of time values. \n
    |    hamiltonian_function: Callable, Hamiltonian function [If None this will be the non_interacting function]. (default = None)
    |    restricted: bool, Is the calculation restricted (r) on unrestricted (u). (default=False)
    |    name: str, Name of method. (default = "non_interacting")

    | Returns:
    |    evolution: iDEA.state.SingleBodyEvolution, Solved time-dependent evolution.
    """
    # Determine the Hamiltonian function.
    if hamiltonian_function is None:
        hamiltonian_function = hamiltonian

    # Construct the unperturbed Hamiltonian.
    n, up_n, down_n = iDEA.observables.density(s, state=state, return_spins=True)
    p, up_p, down_p = iDEA.observables.density_matrix(s, state=state, return_spins=True)
    H, up_H, down_H = hamiltonian_function(s, up_n, down_n, up_p, down_p, **kwargs)
    H = sps.csc_matrix(H)
    up_H = sps.csc_matrix(up_H)
    down_H = sps.csc_matrix(down_H)
    down_H += sps.spdiags(1e-12 * s.x, np.array([0]), s.x.shape[0], s.x.shape[0])

    # Apply restriction.
    if restricted:
        up_H = H
        down_H = H

    # Compute timestep.
    dt = t[1] - t[0]

    # Initilise the single-body time-dependent evolution.
    evolution = iDEA.state.SingleBodyEvolution(state)
    evolution.up.td_orbitals = np.zeros(shape=(t.shape[0], s.x.shape[0], state.up.occupied.shape[0]), dtype=np.complex)
    evolution.down.td_orbitals = np.zeros(
        shape=(t.shape[0], s.x.shape[0], state.down.occupied.shape[0]), dtype=np.complex
    )
    evolution.up.td_orbitals[0, :, :] = state.up.orbitals[:, state.up.occupied]
    evolution.down.td_orbitals[0, :, :] = state.down.orbitals[:, state.down.occupied]
    evolution.v_ptrb = v_ptrb
    evolution.t = t

    # Propagate.
    for j, ti in enumerate(tqdm(t, desc=f"iDEA.methods.{name}.propagate: propagating state")):
        if j != 0:
            evolution = propagate_step(s, evolution, j, hamiltonian_function, v_ptrb, dt, restricted, **kwargs)

    return evolution
