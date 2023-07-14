"""Contains all Strictly Correlated Electrons functions for use in Kohn-Sham Strictly-Correlated Electrons.

References:
1. F. Malet and P. Gori-Giorgi, Strong correlation in Kohn-Sham density functional theory, Phys. Rev. Lett. 109, 246402 (2012). DOI: [10.1103/PhysRevLett.109.246402](https://dx.doi.org/10.1103/PhysRevLett.109.246402)
2. F. Malet, A. Mirtschink, J. C. Cremon, S. M. Reimann, and P. Gori-Giorgi. Kohn-Sham density functional theory for quantum wires in arbitrary correlation regimes. Phys. Rev. B 87, 115146 (2013) DOI: [10.1103/PhysRevB.87.115146](http://dx.doi.org/10.1103/PhysRevB.87.115146)
3. A. Mirtschink, M. Seidl, and P. Gori-Giorgi. Derivative discontinuity in the strong-interaction limit of density functional theory. Phys. Rev. Lett.  111, 126402 (2013) DOI: [10.1103/PhysRevLett.111.126402](http://dx.doi.org/10.1103/PhysRevLett.111.126402)
4. F. Malet, A. Mirtschink, K. J. H. Giesbertz, L. O. Wagner, and P. Gori-Giorgi. Exchange-correlation functionals from the strongly-interacting limit of DFT: Applications to model chemical systems. Phys. Chem. Chem. Phys. 16, 14551 (2014) DOI: [10.1039/c4cp00407h](http://dx.doi.org/10.1039/c4cp00407h)
5. A. Marie, D. P. Kooi, J. Grossi, M. Seidl, Z. Musslimani, K.J.H. Giesbertz, P. Gori-Giorgi. Real space Mott-Anderson electron localization with long-range interactions. Physical Review Research 4, 043192 (2022) DOI: [10.1103/PhysRevResearch.4.043192](https://dx.doi.org/10.1103/PhysRevResearch.4.043192)
"""


from collections.abc import Callable
import numpy as np
import scipy
from scipy.interpolate import PPoly
import iDEA.system
import iDEA.state
import iDEA.observables
import iDEA.methods.non_interacting
from typing import Callable, Dict
from functools import partial

name = "kssce"

default_params = {
    "interp_n": "cubic",
    "interp_invNe": "hermite_cubic",
    "interp_vsce": "cubic",
}

kinetic_energy_operator = iDEA.methods.non_interacting.kinetic_energy_operator
external_potential_operator = iDEA.methods.non_interacting.external_potential_operator
propagate_ni = iDEA.methods.non_interacting.propagate
solve_ni = iDEA.methods.non_interacting.solve


def interpolate_n(x: np.ndarray, n: np.ndarray, interp: str = "cubic") -> PPoly:
    """
    Obtain n(x) as an interpolatant

    Ne(x) needs to be obtained via integration, so we need a method that supports antiderivatives.
    We do not have the derivative of n(x), so we cannot use the Hermite cubic spline.

    | Args:
    |     x: np.ndarray, x-coordinates.
    |     n: np.ndarray, Charge density.
    |     interp: string, Interpolation method. (default = cubic)
    | Returns:
    |     n(x): PPoly, Interpolant of n(x).
    """

    if interp == "cubic":
        # Cubic spline interpolation
        return scipy.interpolate.CubicSpline(
            x,
            n,
            bc_type="clamped",  # First derivative on both ends is zero
        )
    elif interp == "akima":
        # Akima interpolation
        return scipy.interpolate.Akima1DInterpolator(x, n)
    elif interp == "pchip":
        # Piecewise cubic Hermite interpolation
        return scipy.interpolate.PchipInterpolator(x, n)
    else:
        raise ValueError("Interpolation method not recognized.")


def interpolate_invNe(
    n: np.ndarray, Ne: np.ndarray, x: np.ndarray, interp: str = "cubic"
) -> PPoly:
    """
    Obtain x(Ne)

    This is done by interpolation with Ne as the independent variable and x as the dependent variable.

    We have a dx/dNe(Ne) = 1/(n(Neinv(Ne))), so we can use the Hermite cubic spline.

    This would be nicer to do by an inversion of Ne(x), but scipy does not support doing that at multiple points at the same time.

    | Args:
    |     n: np.ndarray, Charge density.
    |     Ne: np.ndarray, Cumulant of the charge density.
    |     x: np.ndarray, x-coordinates.
    |     interp: string, Interpolation method. (default = cubic)
    | Returns:
    |     invNe: PPoly, gives x(Ne).
    """

    if interp == "hermite_cubic":
        # Hermite cubic spline interpolation with derivative
        return scipy.interpolate.CubicHermiteSpline(
            Ne,
            x,
            1 / n,  # Derivative of x(Ne) = 1/n(Neinv(Ne))
        )
    elif interp == "cubic":
        # Cubic spline interpolation
        return scipy.interpolate.CubicSpline(
            Ne,
            x,
            bc_type="not-a-knot",  # Derivative diverges at both ends
        )
    elif interp == "akima":
        # Akima interpolation
        return scipy.interpolate.Akima1DInterpolator(
            Ne,
            x,
        )
    elif interp == "pchip":
        # Piecewise cubic Hermite interpolation
        return scipy.interpolate.PchipInterpolator(
            Ne,
            x,
        )
    else:
        raise ValueError("Interpolation method not recognized.")


def compute_comotion_functions(
    n: np.ndarray, Ne: np.ndarray, x: np.ndarray, N: int, interp: str = "hermite_cubic"
) -> np.ndarray:
    """
    Compute all comotion functions at once.

    | Args:
    |     Ne: np.ndarray, Cumulant of the co-motion function.
    |     x: np.ndarray, Coordinates of the co-motion function.
    |     N: int, Number of particles.
    | Returns:
    |     f: np.ndarray, Co-motion functions.
    """

    # Interpolate inverse of the cumulant
    invNe_interp = interpolate_invNe(n, Ne, x, interp=interp)
    # Electron indices
    i = np.arange(1, N + 1)
    return invNe_interp(
        Ne[None, :]
        + i[:, None]
        - 1
        - np.heaviside(Ne[None, :] - N + i[:, None] - 1, 0.0) * N
    )


def sce_potential_operator(
    s: iDEA.system.System,
    n: np.ndarray,
    method_params: Dict[str, str] = default_params,
) -> np.ndarray:
    r"""
    Compute the SCE potential operator V_SCE, which is diag(v_sce(x)).

    Interpolate v'_sce(x) \sum_i w'(|x-f_i|) and integrate to get the potential v_sce(x).

    The derivative v''_sce(x) is \sum_i sign(x-f_i) w''(|x-f_i|) (1-f_i'(x)).

    f_i'(x) is given by n(x)/n(f(x)) from n(x) dx = n(f(x)) df(x)

    | Args:
    |     s: iDEA.system.System, System object.
    |     n: np.ndarray, Charge density.
    |     method_params: Dict[str, str], Dictionary of method parameters.
    | Returns:
    |     V_SCE: np.ndarray, SCE potential energy operator.
    """
    if np.isclose(np.sum(n), 0):
        return np.zeros(s.x.shape)
    else:
        # Interpolate the charge density
        n_interp = interpolate_n(s.x, n, interp=method_params["interp_n"])
        # Compute the cumulant of the charge density
        Ne = n_interp.antiderivative()(s.x)

        # Compute co-motion functions
        f = compute_comotion_functions(
            n, Ne, s.x, s.count, method_params["interp_invNe"]
        )

        # Compute the derivative of the SCE potential

        # Compute the interaction between the first electron and the rest
        w = s.interaction(s.x - f[1:])

        # Compute v_sce'(x) = \sum_i w'(|x-f_i|)
        v_scep = np.sum(
            s.dinteraction(s.x - f[1:]), axis=0
        )  # axis=0 sums over the electrons

        # Compute w''_i(x) = w''(|x-f_i|)
        wpp_i = s.ddinteraction(s.x - f[1:])

        # Compute f_i'(x) = n(x)/n(f(x))
        fp = n[None, :] / interpolate_n(s.x, n, interp=method_params["interp_n"])(f[1:])

        # Compute v_sce''(x) = \sum_i sign(x-f_i) w''(|x-f_i|) (1-f_i'(x))
        v_scepp = np.sum(wpp_i * np.sign(s.x - f[1:]) * (1 - fp), axis=0)
        # v_sce = -np.cumsum(np.sum(dw, axis=0))*s.dx

        if method_params["interp_vsce"] == "hermite_cubic":
            # Hermite cubic spline interpolation with derivative
            v_sce = scipy.interpolate.CubicHermiteSpline(
                s.x, v_scep, v_scepp
            ).antiderivative()(s.x)
        elif method_params["interp_vsce"] == "akima":
            # Akima interpolation
            v_sce = scipy.interpolate.Akima1DInterpolator(
                s.x, v_scepp
            ).antiderivative()(s.x)
        elif method_params["interp_vsce"] == "pchip":
            # Piecewise cubic Hermite interpolation
            v_sce = scipy.interpolate.PchipInterpolator(s.x, v_scepp).antiderivative()(
                s.x
            )
        elif method_params["interp_vsce"] == "cubic":
            # Cubic spline interpolation
            v_sce = scipy.interpolate.CubicSpline(
                s.x, v_scepp, bc_type="not-a-knot"
            ).antiderivative()(s.x)

        # Compute the SCE energy
        E_sce = (
            1 / 2 * interpolate_n(s.x, n * np.sum(w, axis=0)).integrate(s.x[0], s.x[-1])
        )

        # Fix the arbitrary potential in v_sce by demanding E_sce = int v_sce(x) n(x) dx
        v_sce = v_sce + E_sce / s.count * s.dx - np.sum(v_sce * n) / s.count * s.dx
        return np.diag(v_sce)


def sce_energy(
    s: iDEA.system.System,
    n: np.ndarray,
    method_params: Dict[str, str] = default_params,
) -> float:
    r"""
    Compute the SCE energy.

    | Args:
    |     s: iDEA.system.System, System object.
    |     n: np.ndarray, Charge density.
    |     method_params: Dict[str, str], Dictionary of method parameters.
    | Returns:
    |     E: float, SCE energy.
    """

    # Interpolate the charge density and integrate to get the cumulant Ne(x)
    Ne = interpolate_n(s.x, n, method_params["interp_n"]).antiderivative()(s.x)

    # Compute co-motion functions
    f = compute_comotion_functions(
        n, Ne, s.x, s.count, method_params["interp_invNe"]
    )  # +s.dx/2

    # Compute the interaction between the first electron and the rest
    w = s.interaction(np.abs(s.x - f[1:]))
    return (
        1
        / 2
        * interpolate_n(
            s.x, n * np.sum(w, axis=0), method_params["interp_n"]
        ).integrate(s.x[0], s.x[-1])
    )


def hamiltonian(
    s: iDEA.system.System,
    up_n: np.ndarray,
    down_n: np.ndarray,
    up_p: np.ndarray,
    down_p: np.ndarray,
    K: np.ndarray = None,
    Vext: np.ndarray = None,
    method_params: Dict[str, str] = default_params,
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
    |     default_params: Dict[str, str], Dictionary of method parameters.
    | Returns:
    |     H: np.ndarray, Hamiltonian, up Hamiltonian, down Hamiltonian.
    """
    if K is None:
        K = kinetic_energy_operator(s)
    if Vext is None:
        Vext = external_potential_operator(s)
    Vsce = sce_potential_operator(s, up_n + down_n, method_params)
    H = K + Vext + Vsce
    return H, H, H


def total_energy(
    s: iDEA.system.System,
    state: iDEA.state.SingleBodyState,
    method_params: Dict[str, str] = default_params,
) -> float:
    r"""
    Compute the total energy.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.SingleBodyState, State. (default = None)
    |     method_params: Dict[str, str], Dictionary of method parameters.
    | Returns:
    |     E: float, Total energy.
    """
    E = iDEA.observables.single_particle_energy(s, state)
    n = iDEA.observables.density(s, state)
    E += sce_energy(s, n, method_params)
    return E


def solve(
    s: iDEA.system.System,
    k: int = 0,
    restricted: bool = False,
    mixing: float = 0.5,
    tol: float = 1e-10,
    initial: tuple = None,
    silent: bool = False,
    method_params: Dict[str, str] = default_params,
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
    |     method_params: Dict[str, str], Dictionary of method parameters.
    | Returns:
    |     state: iDEA.state.SingleBodyState, Solved state.
    """
    return solve_ni(
        s,
        partial(hamiltonian, method_params=method_params),
        k,
        restricted,
        mixing,
        tol,
        initial,
        name,
        silent,
    )


def propagate(
    s: iDEA.system.System,
    state: iDEA.state.SingleBodyState,
    v_ptrb: np.ndarray,
    t: np.ndarray,
    restricted: bool = False,
    method_params: Dict[str, str] = default_params,
) -> iDEA.state.SingleBodyEvolution:
    r"""
    Propagate a set of orbitals forward in time due to a dynamic local pertubation.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.SingleBodyState, State to be propigated.
    |     v_ptrb: np.ndarray, Local perturbing potential on the grid of t and x values, indexed as v_ptrb[time,space].
    |     t: np.ndarray, Grid of time values.
    |     restricted: bool, Is the calculation restricted (r) on unrestricted (u). (default=False)
    |     method_params: Dict[str, str], Dictionary of method parameters.
    | Returns:
    |     evolution: iDEA.state.SingleBodyEvolution, Solved time-dependent evolution.
    """
    return propagate_ni(
        s,
        state,
        v_ptrb,
        t,
        partial(hamiltonian, method_params=method_params),
        restricted,
        name,
    )
