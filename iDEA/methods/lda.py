"""Contains all LDA functionality and solvers."""

from collections.abc import Callable
import numpy as np
import iDEA.system
import iDEA.state
import iDEA.observables
import iDEA.methods.non_interacting
import iDEA.methods.hartree


name = "lda"


kinetic_energy_operator = iDEA.methods.non_interacting.kinetic_energy_operator
external_potential_operator = iDEA.methods.non_interacting.external_potential_operator
hartree_potential_operator = iDEA.methods.hartree.hartree_potential_operator
propagate_step = iDEA.methods.non_interacting.propagate_step


class HEG:
    """Class to hold parameters fitted from 1D HEG."""

    ex_lda = {}
    ex_lda["spin_polerised"] = {
        "a": -1.1511,
        "b": 3.3440,
        "c": -9.7079,
        "d": 19.088,
        "e": -20.896,
        "f": 9.4861,
        "g": 0.73586,
    }
    ec_lda = {}
    ec_lda["spin_polerised"] = {
        "a": 0.0009415195,
        "b": 0.2601,
        "c": 0.06404,
        "d": 0.000248,
        "e": 0.00000261,
        "f": 1.254,
        "g": 28.8,
    }
    vx_lda = {}
    eps = ex_lda["spin_polerised"]
    a, b, c, d, e, f, g = (
        eps["a"],
        eps["b"],
        eps["c"],
        eps["d"],
        eps["e"],
        eps["f"],
        eps["g"],
    )
    vx_lda["spin_polerised"] = {
        "a": (g + 1) * a,
        "b": (g + 2) * b,
        "c": (g + 3) * c,
        "d": (g + 4) * d,
        "e": (g + 5) * e,
        "f": (g + 6) * f,
        "g": g,
    }


def exchange_correlation_potential(s: iDEA.system.System, n: np.ndarray, separate: bool = False) -> np.ndarray:
    r"""
    Compute the LDA exchange-correlation potential from a density.

    | Args:
    |     s: iDEA.system.System, System object.
    |     n: np.ndarray, Charge density of the system.
    |     seperate: bool, Set to True to return v_xc, v_x, v_c.

    | Returns:
    |     v_xc: np.ndarray, Exchange correlation potential, or evolution of exchange correlation potential.
    """
    if len(n.shape) == 1:
        p = HEG.vx_lda["spin_polerised"]
        q = HEG.ec_lda["spin_polerised"]
        v_x = np.zeros(shape=s.x.shape[0])
        v_c = np.copy(v_x)
        for j in range(s.x.shape[0]):
            if n[j] != 0.0:
                v_x[j] = (
                    p["a"]
                    + p["b"] * n[j]
                    + p["c"] * n[j] ** 2
                    + p["d"] * n[j] ** 3
                    + p["e"] * n[j] ** 4
                    + p["f"] * n[j] ** 5
                ) * n[j] ** p["g"]
                r_s = 0.5 / n[j]
                energy = (
                    -((q["a"] * r_s + q["e"] * r_s**2) / (1.0 + q["b"] * r_s + q["c"] * r_s**2 + q["d"] * r_s**3))
                    * np.log(1.0 + q["f"] * r_s + q["g"] * r_s**2)
                    / q["f"]
                )
                derivative = (
                    r_s
                    * (q["a"] + q["e"] * r_s)
                    * (q["b"] + r_s * (2.0 * q["c"] + 3.0 * q["d"] * r_s))
                    * np.log(1.0 + q["f"] * r_s + q["g"] * (r_s**2))
                    - (
                        r_s
                        * (q["a"] + q["e"] * r_s)
                        * (q["f"] + 2.0 * q["g"] * r_s)
                        * (q["b"] * r_s + q["c"] * (r_s**2) + q["d"] * (r_s**3) + 1.0)
                        / (q["f"] * r_s + q["g"] * (r_s**2) + 1.0)
                    )
                    - (
                        (q["a"] + 2.0 * q["e"] * r_s)
                        * (q["b"] * r_s + q["c"] * (r_s**2) + q["d"] * (r_s**3) + 1.0)
                        * np.log(1.0 + q["f"] * r_s + q["g"] * (r_s**2))
                    )
                ) / (q["f"] * (q["b"] * r_s + q["c"] * (r_s**2) + q["d"] * (r_s**3) + 1.0) ** 2)
                v_c[j] = energy - r_s * derivative
        v_xc = v_x + v_c
        if separate == True:
            return v_xc, v_x, v_c
        else:
            return v_xc

    elif len(n.shape) == 2:
        raise NotImplementedError()

    else:
        raise AttributeError(f"Expected array of shape 1 or 2, got {n.shape} instead.")


def exchange_correlation_potential_operator(s: iDEA.system.System, n: np.ndarray) -> np.ndarray:
    r"""
    Compute the exchange potential operator.

    | Args;
    |     s: iDEA.system.System, System object.
    |     n: np.ndarray, Charge density of the system.

    | Returns:
    |     Vxc: np.ndarray, Exchange correlation potential energy operator.
    """
    v_xc = exchange_correlation_potential(s, n)
    Vxc = np.diag(v_xc)
    return Vxc


def hamiltonian(
    s: iDEA.system.System,
    up_n: np.ndarray,
    down_n: np.ndarray,
    up_p: np.ndarray,
    down_p: np.ndarray,
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
    Vh = hartree_potential_operator(s, up_n + down_n)
    Vxc = exchange_correlation_potential_operator(s, up_n + down_n)
    H = K + Vext + Vh + Vxc
    return H, H, H


def exchange_correlation_energy(s: iDEA.system.System, n: np.ndarray, separate: bool = False) -> np.ndarray:
    r"""
    Compute the LDA exchange-correlation energy from a density.

    | Args:
    |     s: iDEA.system.System, System object.
    |     n: np.ndarray, Charge density of the system.
    |     seperate: bool, Set to True to return E_xc, E_x, E_c.

    | Returns:
    |     E_xc: np.ndarray, Exchange correlation energy, or evolution of exchange correlation energy.
    """
    p = HEG.ex_lda["spin_polerised"]
    q = HEG.ec_lda["spin_polerised"]
    e_x = np.zeros(shape=s.x.shape[0])
    e_c = np.copy(e_x)
    for j in range(s.x.shape[0]):
        if n[j] != 0.0:
            e_x[j] = (
                p["a"]
                + p["b"] * n[j]
                + p["c"] * n[j] ** 2
                + p["d"] * n[j] ** 3
                + p["e"] * n[j] ** 4
                + p["f"] * n[j] ** 5
            ) * n[j] ** p["g"]
            r_s = 0.5 / n[j]
            e_c[j] = (
                -((q["a"] * r_s + q["e"] * r_s**2) / (1.0 + q["b"] * r_s + q["c"] * r_s**2 + q["d"] * r_s**3))
                * np.log(1.0 + q["f"] * r_s + q["g"] * r_s**2)
                / q["f"]
            )
    e_xc = e_x + e_c
    E_xc = np.dot(e_xc, n) * s.dx
    E_x = np.dot(e_x, n) * s.dx
    E_c = np.dot(e_c, n) * s.dx
    if separate == True:
        return E_xc, E_x, E_c
    else:
        return E_xc


def total_energy(s: iDEA.system.System, state: iDEA.state.SingleBodyState) -> float:
    r"""
    Compute the total energy.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.SingleBodyState, State. (default = None)

    | Returns:
    |     E: float, Total energy.
    """
    E = iDEA.observables.single_particle_energy(s, state)
    n = iDEA.observables.density(s, state)
    v_h = iDEA.observables.hartree_potential(s, n)
    E -= iDEA.observables.hartree_energy(s, n, v_h)
    v_xc = exchange_correlation_potential(s, n)
    E -= np.dot(n, v_xc) * s.dx
    E += exchange_correlation_energy(s, n)
    return E.real


def solve(
    s: iDEA.system.System,
    k: int = 0,
    restricted: bool = False,
    mixing: float = 0.5,
    tol: float = 1e-10,
    initial: tuple = None,
    silent: bool = False,
) -> iDEA.state.SingleBodyState:
    r"""
    Solves the Schrodinger equation for the given system.

    | Args:
    |     s: iDEA.system.System, System object.
    |     k: int, Energy state to solve for. (default = 0, the ground-state)
    |     restricted: bool, Is the calculation restricted (r) on unrestricted (u). (default=False)
    |     mixing: float, Mixing parameter. (default = 0.5)
    |     tol: float, Tollerance of convergence. (default = 1e-10)
    |     initial: tuple. Tuple of initial values used to begin the self-consistency (n, up_n, down_n, p, up_p, down_p). (default = None)
    |     silent: bool, Set to true to prevent printing. (default = False)

    | Returns:
    |     state: iDEA.state.SingleBodyState, Solved state.
    """
    return iDEA.methods.non_interacting.solve(s, hamiltonian, k, restricted, mixing, tol, initial, name, silent)


def propagate(
    s: iDEA.system.System,
    state: iDEA.state.SingleBodyState,
    v_ptrb: np.ndarray,
    t: np.ndarray,
    hamiltonian_function: Callable = None,
    restricted: bool = False,
) -> iDEA.state.SingleBodyEvolution:
    r"""
    Propagate a set of orbitals forward in time due to a dynamic local pertubation.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.SingleBodyState, State to be propigated.
    |     v_ptrb: np.ndarray, Local perturbing potential on the grid of t and x values, indexed as v_ptrb[time,space].
    |     t: np.ndarray, Grid of time values.
    |     hamiltonian_function: Callable, Hamiltonian function [If None this will be the non_interacting function]. (default = None)
    |     restricted: bool, Is the calculation restricted (r) on unrestricted (u). (default=False)

    | Returns:
    |     evolution: iDEA.state.SingleBodyEvolution, Solved time-dependent evolution.
    """
    return iDEA.methods.non_interacting.propagate(s, state, v_ptrb, t, hamiltonian, restricted, name)
