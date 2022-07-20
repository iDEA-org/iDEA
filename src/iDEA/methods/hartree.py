"""Contains all Hartree functionality and solvers."""


from collections.abc import Callable
import numpy as np
import iDEA.system
import iDEA.state
import iDEA.observables
import iDEA.methods.non_interacting


name = "hartree"


kinetic_energy_operator = iDEA.methods.non_interacting.kinetic_energy_operator
external_potential_operator = iDEA.methods.non_interacting.external_potential_operator


def hartree_potential_operator(s: iDEA.system.System, n: np.ndarray) -> np.ndarray:
    """
    Compute the Hartree potential operator.

    Args;
        s: iDEA.system.System, System object.
        n: np.ndarray, Charge density.

    Returns:
        Vh: np.ndarray, Hartree potential energy operator.
    """
    v_h = iDEA.observables.hartree_potential(s, n)
    Vh = np.diag(v_h)
    return Vh


def hamiltonian(s: iDEA.system.System, up_n: np.ndarray, down_n: np.ndarray, up_p: np.ndarray, down_p: np.ndarray, K: np.ndarray = None, V: np.ndarray = None) -> np.ndarray:
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
        Vext = external_potential_operator(s)
    Vh = hartree_potential_operator(s, up_n + down_n)
    H = K + Vext + Vh
    return H, H, H


def total_energy(s: iDEA.system.System, state: iDEA.state.SingleBodyState) -> float:
    """
    Compute the total energy.

    Args:
        s: iDEA.system.System, System object.
        state: iDEA.state.SingleBodyState, State. (default = None)

    Returns:
        E: float, Total energy.
    """
    E = iDEA.observables.single_particle_energy(s, state)
    n = iDEA.observables.density(s, state)
    v_h = iDEA.observables.hartree_potential(s, n)
    E -= 0.5 * iDEA.observables.hartree_energy(s, n, v_h)
    return E


def solve(s: iDEA.system.System, k: int = 0, restricted: bool = False, mixing: float = 0.5, tol: float = 1e-10, initial: tuple = None, silent: bool = False) -> iDEA.state.SingleBodyState:
    """
    Solves the Schrodinger equation for the given system.

    Args:
        s: iDEA.system.System, System object.
        k: int, Energy state to solve for. (default = 0, the ground-state)
        restricted: bool, Is the calculation restricted (r) on unrestricted (u). (default=False)
        mixing: float, Mixing parameter. (default = 0.5)
        tol: float, Tollerance of convergence. (default = 1e-10)
        initial: tuple. Tuple of initial values used to begin the self-consistency (n, up_n, down_n, p, up_p, down_p). (default = None)
        silent: bool, Set to true to prevent printing. (default = False)

    Returns:
        state: iDEA.state.SingleBodyState, Solved state.
    """
    return iDEA.methods.non_interacting.solve(s, hamiltonian, k, restricted, mixing, tol, initial, name, silent)


def propagate(s: iDEA.system.System, state: iDEA.state.SingleBodyState, v_ptrb: np.ndarray, t: np.ndarray, hamiltonian_function: Callable = None, restricted: bool = False) -> iDEA.state.SingleBodyEvolution:
    """
    Propagate a set of orbitals forward in time due to a dynamic local pertubation.

    Args: 
        s: iDEA.system.System, System object.
        state: iDEA.state.SingleBodyState, State to be propigated.
        v_ptrb: np.ndarray, Local perturbing potential on the grid of t and x values, indexed as v_ptrb[time,space].
        t: np.ndarray, Grid of time values.
        hamiltonian_function: Callable, Hamiltonian function [If None this will be the non_interacting function]. (default = None)
        restricted: bool, Is the calculation restricted (r) on unrestricted (u). (default=False)

    Returns:
        evolution: iDEA.state.SingleBodyEvolution, Solved time-dependent evolution.
    """
    return iDEA.methods.non_interacting.propagate(s, state, v_ptrb, t, hamiltonian, restricted, name) 