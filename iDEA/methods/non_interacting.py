"""Contains all non-interacting functionality and solvers."""

import copy
import itertools
import warnings
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
import iDEA.utilities

name = "non_interacting"

# Delta SCF (k > 0) controls: iterations allowed per attempt, and attempts (each halving the mixing) before giving up.
max_excited_iterations = 5000
max_excited_attempts = 4


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
        sd = 1.0 * np.array([1, -2, 1], dtype=float) / s.dx**2
        sdi = (-1, 0, 1)
    elif s.stencil == 5:
        sd = 1.0 / 12.0 * np.array([-1, 16, -30, 16, -1], dtype=float) / s.dx**2
        sdi = (-2, -1, 0, 1, 2)
    elif s.stencil == 7:
        sd = 1.0 / 180.0 * np.array([2, -27, 270, -490, 270, -27, 2], dtype=float) / s.dx**2
        sdi = (-3, -2, -1, 0, 1, 2, 3)
    elif s.stencil == 9:
        sd = 1.0 / 5040.0 * np.array([-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9], dtype=float) / s.dx**2
        sdi = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
    elif s.stencil == 11:
        sd = (
            1.0
            / 25200.0
            * np.array(
                [8, -125, 1000, -6000, 42000, -73766, 42000, -6000, 1000, -125, 8],
                dtype=float,
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
                dtype=float,
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


def _rank_configurations(
    s: iDEA.system.System,
    state: iDEA.state.SingleBodyState,
    max_level: int,
    hamiltonian_function: Callable = None,
    energy_function: Callable = None,
    restricted: bool = False,
    **kwargs,
) -> list:
    r"""
    Enumerate every occupation configuration within the lowest max_level orbitals of each spin and rank them by energy.

    Configurations are enumerated in lexicographic order of (up indices, down indices) and sorted with a stable sort,
    so exactly degenerate configurations (e.g. the spin-flipped pairs ([1], [2]) and ([2], [1])) are always ordered
    the same way, independently of max_level. The ranking is therefore a total order, so distinct excitations k map
    to distinct configurations.

    If either hamiltonian_function or energy_function is None, each configuration is ranked by the sum of its
    occupied eigenvalues. This is the exact total energy for non-interacting electrons. Otherwise each configuration
    is ranked by the method's own total energy evaluated with the orbitals of state frozen: the Hamiltonian is built
    from the configuration's density (and density matrix), the orbital energies are replaced by the expectation values
    of that Hamiltonian, and energy_function is then called. This removes the double counting of the interaction
    terms that the eigenvalue sum contains.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.SingleBodyState, Reference state whose orbitals and energies are used.
    |     max_level: int, Number of orbitals of each spin to consider.
    |     hamiltonian_function: Callable, Hamiltonian function of the method. (default = None)
    |     energy_function: Callable, Total energy function of the method. (default = None)
    |     restricted: bool, Is the calculation restricted (r) on unrestricted (u). (default=False)

    | Returns:
    |     configurations: list, List of (up_indices, down_indices) tuples sorted by energy (lowest first).
    """
    configurations = list(
        itertools.product(
            itertools.combinations(range(max_level), s.up_count),
            itertools.combinations(range(max_level), s.down_count),
        )
    )
    energies = []
    for up_index, down_index in configurations:
        state_copy = copy.deepcopy(state)
        state_copy.up.occupations = np.zeros(shape=state.up.energies.shape)
        state_copy.up.occupations[list(up_index)] = 1.0
        state_copy.down.occupations = np.zeros(shape=state.down.energies.shape)
        state_copy.down.occupations[list(down_index)] = 1.0
        if hamiltonian_function is None or energy_function is None:
            E = iDEA.observables.single_particle_energy(s, state_copy)
        else:
            n, up_n, down_n = iDEA.observables.density(s, state_copy, return_spins=True)
            p, up_p, down_p = iDEA.observables.density_matrix(s, state_copy, return_spins=True)
            H, up_H, down_H = hamiltonian_function(s, up_n, down_n, up_p, down_p, **kwargs)
            if restricted:
                up_H = H
                down_H = H
            up_orbitals = state.up.orbitals[:, :max_level]
            down_orbitals = state.down.orbitals[:, :max_level]
            state_copy.up.energies = np.zeros(shape=state.up.energies.shape)
            state_copy.up.energies[:max_level] = np.sum(up_orbitals.conj() * (up_H @ up_orbitals), axis=0).real * s.dx
            state_copy.down.energies = np.zeros(shape=state.down.energies.shape)
            state_copy.down.energies[:max_level] = (
                np.sum(down_orbitals.conj() * (down_H @ down_orbitals), axis=0).real * s.dx
            )
            E = energy_function(s, state_copy, **kwargs)
        energies.append(E)
    order = np.argsort(energies, kind="stable")
    return [configurations[i] for i in order]


def add_occupations(
    s: iDEA.system.System,
    state: iDEA.state.SingleBodyState,
    k: int,
    hamiltonian_function: Callable = None,
    energy_function: Callable = None,
    restricted: bool = False,
    **kwargs,
) -> iDEA.state.SingleBodyState:
    r"""
    Calculate the occupations of a state in a given energy excitation.

    All configurations of s.up_count up and s.down_count down electrons in the orbitals of state are ranked by energy
    (see _rank_configurations) and the k-th lowest is chosen. Exactly degenerate configurations are ordered
    deterministically, so each k maps to a distinct configuration.

    If hamiltonian_function and energy_function are given, configurations are ranked by the method's total energy
    with the orbitals of state frozen. In this case the ground-state (aufbau) configuration is always taken to be
    k = 0, and the candidate orbital window is enlarged until the k + 1 lowest configurations no longer change.
    Otherwise configurations are ranked by the sum of occupied eigenvalues, which is exact for non-interacting
    electrons.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.SingleBodyState, State.
    |     k: int, Excitation state [k = 0 is the ground-state].
    |     hamiltonian_function: Callable, Hamiltonian function used to rank configurations. (default = None)
    |     energy_function: Callable, Total energy function used to rank configurations. (default = None)
    |     restricted: bool, Is the calculation restricted (r) on unrestricted (u). (default=False)

    | Returns:
    |     state: iDEA.state.SingleBodyState, State with occupations added.
    """
    # Calculate the max level or orbitals needed to achieve required state and only use these.
    # With eigenvalue-sum ranking, any configuration that occupies an orbital above this level has at least k + 1
    # configurations below it, so the k + 1 lowest configurations always lie within it.
    num_orbitals = state.up.energies.shape[0]
    max_level = min(max(s.up_count, s.down_count) + k, num_orbitals)

    if hamiltonian_function is None or energy_function is None:
        configurations = _rank_configurations(s, state, max_level)
    else:
        # Rank by the method's total energy, keeping the aufbau configuration as the ground-state.
        aufbau = (tuple(range(s.up_count)), tuple(range(s.down_count)))

        def rank(level):
            ranked = _rank_configurations(s, state, level, hamiltonian_function, energy_function, restricted, **kwargs)
            ranked.remove(aufbau)
            return [aufbau] + ranked

        # The total energy is not guaranteed to grow with orbital index, so enlarge the window until it is converged.
        configurations = rank(max_level)
        while max_level < num_orbitals:
            extended = rank(max_level + 1)
            if extended[: k + 1] == configurations[: k + 1]:
                break
            configurations = extended
            max_level += 1

    # Construct correct occupations.
    up_index, down_index = configurations[k]
    state.up.occupations = np.zeros(shape=state.up.energies.shape)
    state.up.occupations[list(up_index)] = 1.0
    state.up.occupied = np.nonzero(state.up.occupations)[0]
    state.down.occupations = np.zeros(shape=state.down.energies.shape)
    state.down.occupations[list(down_index)] = 1.0
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
    energy_function: Callable = None,
    **kwargs,
) -> iDEA.state.SingleBodyState:
    r"""
    Solves the Schrodinger equation for the given system.

    Excited states (k > 0) are computed with the Delta SCF approach: the ground-state is first converged, the k-th
    lowest occupation configuration is chosen from the ground-state orbitals (see add_occupations), and the
    self-consistent cycle is then re-run with that configuration held fixed, so the returned state is self-consistent
    with its own (excited) occupation. During this cycle the occupied orbitals are tracked by maximum overlap with
    those of the previous iteration, and the mixing is halved (up to max_excited_attempts times) if it fails to
    converge. Configurations are ranked by energy_function (the method's total energy, evaluated with the
    ground-state orbitals frozen) when it is given, and by the sum of occupied eigenvalues otherwise. As the ranking
    is done before the excited state is relaxed, the self-consistent total energies of the returned states are not
    guaranteed to be in increasing order of k. If a configuration has no distinct self-consistent solution it can
    relax into a lower one (variational collapse); a warning is issued when the occupied orbitals end up reordered.

    | Args:
    |     s: iDEA.system.System, System object.
    |     hamiltonian_function: Callable, Hamiltonian function [If None this will be the non_interacting function]. (default = None)
    |     k: int, Excitation to solve for, i.e. the k-th lowest-energy occupation configuration, solved self-consistently (Delta SCF). Degenerate configurations are counted separately. (default = 0, the ground-state)
    |     restricted: bool, Is the calculation restricted (r) on unrestricted (u). (default=False)
    |     mixing: float, Mixing parameter. (default = 0.5)
    |     tol: float, Tollerance of convergence. (default = 1e-10)
    |     initial: tuple. Tuple of initial values used to begin the self-consistency (n, up_n, down_n, p, up_p, down_p). (default = None)
    |     name: str, Name of method. (default = "non_interacting")
    |     silent: bool, Set to true to prevent printing. (default = False)
    |     energy_function: Callable, Total energy function used to rank excited configurations [If None the sum of occupied eigenvalues is used]. (default = None)


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
        up_H = H
        down_H = H

    # Run self-consitent algorithm for the ground-state.
    old = (n_old, up_n_old, down_n_old, p_old, up_p_old, down_p_old)
    state = _self_consistent_cycle(
        s, state, hamiltonian_function, up_H, down_H, old, restricted, mixing, tol, name, silent, **kwargs
    )

    # Compute state occupations.
    if k == 0:
        state = add_occupations(s, state, k)
    else:
        # Delta SCF: choose the excited configuration from the converged ground-state orbitals, then re-converge the
        # orbitals self-consistently with that configuration held fixed (tracked by maximum overlap).
        reference = add_occupations(s, state, k, hamiltonian_function, energy_function, restricted, **kwargs)
        up_chosen = reference.up.occupied.copy()
        down_chosen = reference.down.occupied.copy()
        n_old, up_n_old, down_n_old = iDEA.observables.density(s, reference, return_spins=True)
        p_old, up_p_old, down_p_old = iDEA.observables.density_matrix(s, reference, return_spins=True)
        old = (n_old, up_n_old, down_n_old, p_old, up_p_old, down_p_old)

        # Excited states typically need more damping than the ground-state, so halve the mixing until converged.
        excited_mixing = mixing
        for attempt in range(max_excited_attempts):
            H, up_H, down_H = hamiltonian_function(s, up_n_old, down_n_old, up_p_old, down_p_old, **kwargs)
            if restricted:
                up_H = H
                down_H = H
            if silent is False:
                print()
            try:
                state = _self_consistent_cycle(
                    s,
                    copy.deepcopy(reference),
                    hamiltonian_function,
                    up_H,
                    down_H,
                    old,
                    restricted,
                    excited_mixing,
                    tol,
                    name,
                    silent,
                    maximum_overlap=True,
                    max_iterations=max_excited_iterations,
                    **kwargs,
                )
                break
            except RuntimeError as error:
                if attempt == max_excited_attempts - 1:
                    raise RuntimeError(
                        f"iDEA.methods.{name}.solve: Delta SCF for k = {k} (up {up_chosen.tolist()}, down "
                        f"{down_chosen.tolist()}) did not converge, with mixing reduced down to {excited_mixing:.5}."
                    ) from error
                excited_mixing = 0.5 * excited_mixing

        # Warn if the excited configuration did not stay where it was placed (e.g. collapsed towards a lower state).
        if not (np.array_equal(state.up.occupied, up_chosen) and np.array_equal(state.down.occupied, down_chosen)):
            aufbau = np.array_equal(state.up.occupied, np.arange(s.up_count)) and np.array_equal(
                state.down.occupied, np.arange(s.down_count)
            )
            warnings.warn(
                f"iDEA.methods.{name}.solve: the k = {k} configuration (up {up_chosen.tolist()}, down "
                f"{down_chosen.tolist()}) reordered during self-consistency and converged occupying orbitals (up "
                f"{state.up.occupied.tolist()}, down {state.down.occupied.tolist()}). The returned state is "
                "self-consistent, but "
                + ("has likely collapsed to the ground-state." if aufbau else "may coincide with a lower excitation."),
                stacklevel=2,
            )
    if silent is False:
        print()

    return state


def _maximum_overlap_occupations(
    s: iDEA.system.System, spin_state: iDEA.utilities.Container, occupied_orbitals: np.ndarray
) -> None:
    r"""
    Occupy, in place, the orbitals of spin_state with the largest overlap with the given previously occupied orbitals.

    | Args:
    |     s: iDEA.system.System, System object.
    |     spin_state: iDEA.utilities.Container, The up or down part of a state (orbitals already updated).
    |     occupied_orbitals: np.ndarray, Previously occupied orbitals, indexed as occupied_orbitals[space, orbital].
    """
    count = occupied_orbitals.shape[1]
    if count == 0:
        return
    overlap = np.sum(abs(occupied_orbitals.conj().T @ spin_state.orbitals * s.dx) ** 2, axis=0)
    occupied = np.sort(np.argsort(-overlap, kind="stable")[:count])
    spin_state.occupations = np.zeros(shape=spin_state.energies.shape)
    spin_state.occupations[occupied] = 1.0
    spin_state.occupied = occupied


def _self_consistent_cycle(
    s: iDEA.system.System,
    state: iDEA.state.SingleBodyState,
    hamiltonian_function: Callable,
    up_H: np.ndarray,
    down_H: np.ndarray,
    old: tuple,
    restricted: bool,
    mixing: float,
    tol: float,
    name: str,
    silent: bool,
    maximum_overlap: bool = False,
    max_iterations: int = None,
    **kwargs,
) -> iDEA.state.SingleBodyState:
    r"""
    Run the self-consistent cycle with the occupations of state held fixed.

    If maximum_overlap is False the occupations are held fixed by orbital index (aufbau for the ground-state).
    If maximum_overlap is True the occupied orbitals are instead chosen at each iteration as those with the largest
    overlap with the previously occupied orbitals (the maximum overlap method). This keeps an excited configuration
    fixed even when the order of the orbitals changes during the cycle, which otherwise causes the cycle to oscillate.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.SingleBodyState, State whose occupations are used.
    |     hamiltonian_function: Callable, Hamiltonian function.
    |     up_H: np.ndarray, Initial Hamiltonian for up electrons.
    |     down_H: np.ndarray, Initial Hamiltonian for down electrons.
    |     old: tuple, Initial values (n, up_n, down_n, p, up_p, down_p).
    |     restricted: bool, Is the calculation restricted (r) on unrestricted (u).
    |     mixing: float, Mixing parameter.
    |     tol: float, Tollerance of convergence.
    |     name: str, Name of method.
    |     silent: bool, Set to true to prevent printing.
    |     maximum_overlap: bool, Choose occupied orbitals by maximum overlap. (default = False)
    |     max_iterations: int, Maximum number of iterations before raising a RuntimeError, which is also raised if the convergence has not improved for max_iterations // 5 iterations [If None there is no limit]. (default = None)

    | Returns:
    |     state: iDEA.state.SingleBodyState, Converged state.
    """
    n_old, up_n_old, down_n_old, p_old, up_p_old, down_p_old = old
    convergence = 1.0
    count = 0
    best_convergence = np.inf
    best_count = 0
    while convergence > tol:
        # Give up if the iteration limit is reached, or the convergence has stopped improving.
        if max_iterations is not None and (count >= max_iterations or count - best_count > max(max_iterations // 5, 1)):
            raise RuntimeError(
                f"iDEA.methods.{name}.solve: self-consistency not reached after {count} iterations "
                f"(convergence = {convergence:.5}, tolerance = {tol:.5}). Try reducing the mixing parameter."
            )

        # Perform single self-consistent step.
        if maximum_overlap:
            up_occupied_orbitals = state.up.orbitals[:, state.up.occupied]
            down_occupied_orbitals = state.down.orbitals[:, state.down.occupied]
        state = sc_step(s, state, up_H, down_H)
        if maximum_overlap:
            _maximum_overlap_occupations(s, state.up, up_occupied_orbitals)
            _maximum_overlap_occupations(s, state.down, down_occupied_orbitals)

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
            up_H = H
            down_H = H

        # Test for convergence.
        convergence = np.sum(abs(n - n_old)) * s.dx
        if convergence < best_convergence:
            best_convergence = convergence
            best_count = count

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
    evolution.up.td_orbitals = np.zeros(shape=(t.shape[0], s.x.shape[0], state.up.occupied.shape[0]), dtype=complex)
    evolution.down.td_orbitals = np.zeros(shape=(t.shape[0], s.x.shape[0], state.down.occupied.shape[0]), dtype=complex)
    evolution.up.td_orbitals[0, :, :] = state.up.orbitals[:, state.up.occupied]
    evolution.down.td_orbitals[0, :, :] = state.down.orbitals[:, state.down.occupied]
    evolution.v_ptrb = v_ptrb
    evolution.t = t

    # Propagate.
    for j, _ti in enumerate(tqdm(t, desc=f"iDEA.methods.{name}.propagate: propagating state")):
        if j != 0:
            evolution = propagate_step(s, evolution, j, hamiltonian_function, v_ptrb, dt, restricted, **kwargs)

    return evolution
