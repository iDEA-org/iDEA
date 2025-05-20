import copy
import itertools
import string
from typing import Union

import numpy as np

import iDEA.methods.interacting
import iDEA.methods.non_interacting
import iDEA.state
import iDEA.system


def observable(
    s: iDEA.system.System,
    observable_operator: np.ndarray,
    state: Union[iDEA.state.SingleBodyState, iDEA.state.ManyBodyState] = None,
    evolution: Union[iDEA.state.SingleBodyEvolution, iDEA.state.ManyBodyEvolution] = None,
    return_spins: bool = False,
) -> Union[float, np.ndarray]:
    r"""
    Compute an observable based on a given operator and state or evolution.

    | Args:
    |     s: iDEA.system.System, System object.
    |     observable_operator: np.ndarray, Obserbable operator.
    |     state: iDEA.state.SingleBodyState or iDEA.state.ManyBodyState, State. (default = None)
    |     evolution: iDEA.state.SingleBodyEvolution or iDEA.state.ManyBodyEvolution, Evolution. (default = None)
    |     return_spins: bool, True to also return the spin observables: total, up, down. (default = False)

    | Returns:
    |     observable: float or np.ndarray, Observable.
    """
    if state is not None and type(state) == iDEA.state.ManyBodyState:
        raise NotImplementedError()

    if state is not None and type(state) == iDEA.state.SingleBodyState:
        up_O = 0.0
        for i in range(state.up.orbitals.shape[1]):
            up_O += (
                np.vdot(
                    state.up.orbitals[:, i],
                    np.dot(observable_operator, state.up.orbitals[:, i]),
                )
                * state.up.occupations[i]
                * s.dx
            )
        down_O = 0.0
        for i in range(state.down.orbitals.shape[1]):
            down_O += (
                np.vdot(
                    state.down.orbitals[:, i],
                    np.dot(observable_operator, state.down.orbitals[:, i]),
                )
                * state.down.occupations[i]
                * s.dx
            )
        O = up_O + down_O
        if return_spins:
            return O, up_O, down_O
        else:
            return O

    if evolution is not None and type(evolution) == iDEA.state.ManyBodyEvolution:
        raise NotImplementedError()

    if evolution is not None and type(evolution) == iDEA.state.SingleBodyEvolution:
        up_O = np.zeros(shape=evolution.t.shape, dtype=complex)
        for i, I in enumerate(evolution.up.occupied):
            for j in evolution.t:
                up_O[j] += (
                    np.vdot(
                        evolution.up.td_orbitals[j, :, i],
                        np.dot(observable_operator, evolution.up.td_orbitals[j, :, i]),
                    )
                    * evolution.up.occupations[I]
                    * s.dx
                )
        down_O = np.zeros(shape=evolution.t.shape, dtype=complex)
        for i, I in enumerate(evolution.down.occupied):
            for j in evolution.t:
                down_O[j] += (
                    np.vdot(
                        evolution.down.td_orbitals[j, :, i],
                        np.dot(observable_operator, evolution.down.td_orbitals[j, :, i]),
                    )
                    * evolution.down.occupations[I]
                    * s.dx
                )
        O = up_O + down_O
        if return_spins:
            return O.real, up_O.real, down_O.real
        else:
            return O.real

    else:
        raise AttributeError("State or Evolution must be provided.")


def density(
    s: iDEA.system.System,
    state: Union[iDEA.state.SingleBodyState, iDEA.state.ManyBodyState] = None,
    evolution: Union[iDEA.state.SingleBodyEvolution, iDEA.state.ManyBodyEvolution] = None,
    time_indices: np.ndarray = None,
    return_spins: bool = False,
) -> np.ndarray:
    r"""
    Compute the charge density of a non_interacting state.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.SingleBodyState or iDEA.state.ManyBodyState, State. (default = None)
    |     evolution: iDEA.state.SingleBodyEvolution or iDEA.state.ManyBodyEvolution, Evolution. (default = None)
    |     time_indices: np.ndarray, Time indices to compute observable if given evolution. If None will perform for all time indices. (default = None)
    |     return_spins: bool, True to also return the spin densities: total, up, down. (default = False)

    | Returns:
    |     n: np.ndarray, Charge density, or evolution of charge density.
    """
    if state is not None and type(state) == iDEA.state.ManyBodyState:
        spin_densities = np.zeros(shape=(s.x.shape[0], 2))
        for i in range(s.x.shape[0]):
            for k in range(2):
                spin_densities[i, k] = np.sum(abs(state.full[i, k, ...]) ** 2) * s.dx ** (s.count - 1) * s.count
        up_n = spin_densities[:, 0]
        down_n = spin_densities[:, 1]
        n = up_n + down_n
        if return_spins:
            return n, up_n, down_n
        else:
            return n

    if state is not None and type(state) == iDEA.state.SingleBodyState:
        up_n = np.zeros(shape=s.x.shape[0])
        down_n = np.zeros(shape=s.x.shape[0])
        for i in range(state.up.orbitals.shape[1]):
            up_n += abs(state.up.orbitals[:, i]) ** 2 * state.up.occupations[i]
        for i in range(state.down.orbitals.shape[1]):
            down_n += abs(state.down.orbitals[:, i]) ** 2 * state.down.occupations[i]
        n = up_n + down_n
        if return_spins:
            return n, up_n, down_n
        else:
            return n

    if evolution is not None and type(evolution) == iDEA.state.ManyBodyEvolution:
        if time_indices is None:
            time_indices = np.array(range(evolution.t.shape[0]))
        spin_densities = np.zeros(shape=(time_indices.shape[0], s.x.shape[0], 2))
        for j, ti in enumerate(time_indices):
            l = string.ascii_lowercase[: s.count]
            L = string.ascii_uppercase[: s.count]
            st = l + "," + L + "->" + "".join([i for sub in list(zip(l, L)) for i in sub])
            full = np.einsum(st, evolution.td_space[ti, ...], evolution.spin)
            L = list(zip(list(range(0, s.count * 2, 2)), list(range(1, s.count * 2, 2))))
            perms = itertools.permutations(list(range(s.count)))
            full_copy = copy.deepcopy(full)
            full = np.zeros_like(full_copy)
            for p in perms:
                indices = list(itertools.chain(*[L[e] for e in p]))
                full += iDEA.methods.interacting._permutation_parity(p) * np.moveaxis(
                    full_copy, list(range(s.count * 2)), indices
                )
            full = full / np.sqrt(np.sum(abs(full) ** 2) * s.dx**s.count)
            for i in range(s.x.shape[0]):
                for k in range(2):
                    spin_densities[j, i, k] = np.sum(abs(full[i, k, ...]) ** 2) * s.dx ** (s.count - 1) * s.count
        up_n = spin_densities[:, :, 0]
        down_n = spin_densities[:, :, 1]
        n = up_n + down_n
        if return_spins:
            return n, up_n, down_n
        else:
            return n

    if evolution is not None and type(evolution) == iDEA.state.SingleBodyEvolution:
        if time_indices is None:
            time_indices = np.array(range(evolution.t.shape[0]))
        up_n = np.zeros(shape=(time_indices.shape[0], s.x.shape[0]))
        for i, I in enumerate(evolution.up.occupied):
            for j, ti in enumerate(time_indices):
                up_n[j, :] += abs(evolution.up.td_orbitals[ti, :, i]) ** 2 * evolution.up.occupations[I]
        down_n = np.zeros(shape=(time_indices.shape[0], s.x.shape[0]))
        for i, I in enumerate(evolution.down.occupied):
            for j, ti in enumerate(time_indices):
                down_n[j, :] += abs(evolution.down.td_orbitals[ti, :, i]) ** 2 * evolution.down.occupations[I]
        n = up_n + down_n
        if return_spins:
            return n, up_n, down_n
        else:
            return n

    else:
        raise AttributeError("State or Evolution must be provided.")


def density_matrix(
    s: iDEA.system.System,
    state: Union[iDEA.state.SingleBodyState, iDEA.state.ManyBodyState] = None,
    evolution: Union[iDEA.state.SingleBodyEvolution, iDEA.state.ManyBodyEvolution] = None,
    time_indices: np.ndarray = None,
    return_spins: bool = False,
) -> np.ndarray:
    r"""
    Compute the charge density matrix of a non_interacting state.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.SingleBodyState or iDEA.state.ManyBodyState, State. (default = None)
    |     evolution: iDEA.state.SingleBodyEvolution or iDEA.state.ManyBodyEvolution, Evolution. (default = None)
    |     time_indices: np.ndarray, Time indices to compute observable if given evolution. If None will perform for all time indices. (default = None)
    |     return_spins: bool, True to also return the spin density matrices: total, up, down. (default = False)

    | Returns:
    |     p: np.ndarray, Charge density matrix, or evolution of charge density matrix.
    """
    if state is not None and type(state) == iDEA.state.ManyBodyState:
        tosum = list(range(2, s.count * 2))
        spin_p = (
            np.tensordot(state.full, state.full.conj(), axes=(tosum, tosum)).diagonal(axis1=1, axis2=3)
            * s.dx ** (s.count - 1)
            * s.count
        )
        up_p = spin_p[:, :, 0]
        down_p = spin_p[:, :, 1]
        p = up_p + down_p
        if return_spins:
            return p, up_p, down_p
        else:
            return p

    if state is not None and type(state) == iDEA.state.SingleBodyState:
        up_p = np.zeros(shape=s.x.shape * 2)
        down_p = np.zeros(shape=s.x.shape * 2)
        for i in range(state.up.orbitals.shape[1]):
            up_p += (
                np.tensordot(state.up.orbitals[:, i], state.up.orbitals[:, i].conj(), axes=0) * state.up.occupations[i]
            )
        for i in range(state.down.orbitals.shape[1]):
            down_p += (
                np.tensordot(state.down.orbitals[:, i], state.down.orbitals[:, i].conj(), axes=0)
                * state.down.occupations[i]
            )
        p = up_p + down_p
        if return_spins:
            return p, up_p, down_p
        else:
            return p

    if evolution is not None and type(evolution) == iDEA.state.ManyBodyEvolution:
        if time_indices is None:
            time_indices = np.array(range(evolution.t.shape[0]))
        tosum = list(range(2, s.count * 2))
        spin_density_matrices = np.zeros(shape=(time_indices.shape[0], s.x.shape[0], s.x.shape[0], 2), dtype=complex)
        for j, ti in enumerate(time_indices):
            l = string.ascii_lowercase[: s.count]
            L = string.ascii_uppercase[: s.count]
            st = l + "," + L + "->" + "".join([i for sub in list(zip(l, L)) for i in sub])
            full = np.einsum(st, evolution.td_space[ti, ...], evolution.spin)
            L = list(zip(list(range(0, s.count * 2, 2)), list(range(1, s.count * 2, 2))))
            perms = itertools.permutations(list(range(s.count)))
            full_copy = copy.deepcopy(full)
            full = np.zeros_like(full_copy)
            for p in perms:
                indices = list(itertools.chain(*[L[e] for e in p]))
                full += iDEA.methods.interacting._permutation_parity(p) * np.moveaxis(
                    full_copy, list(range(s.count * 2)), indices
                )
            full = full / np.sqrt(np.sum(abs(full) ** 2) * s.dx**s.count)
            spin_density_matrices[j, :, :, :] = (
                np.tensordot(full, full.conj(), axes=(tosum, tosum)).diagonal(axis1=1, axis2=3)
                * s.dx ** (s.count - 1)
                * s.count
            )
        up_p = spin_density_matrices[:, :, :, 0]
        down_p = spin_density_matrices[:, :, :, 1]
        p = up_p + down_p
        if return_spins:
            return p, up_p, down_p
        else:
            return p

    if evolution is not None and type(evolution) == iDEA.state.SingleBodyEvolution:
        if time_indices is None:
            time_indices = np.array(range(evolution.t.shape[0]))
        up_p = np.zeros(shape=(time_indices.shape[0], s.x.shape[0], s.x.shape[0]), dtype=complex)
        for i, I in enumerate(evolution.up.occupied):
            for j, ti in enumerate(time_indices):
                up_p[j, :] += (
                    np.tensordot(
                        evolution.up.td_orbitals[ti, :, i],
                        evolution.up.td_orbitals[ti, :, i].conj(),
                        axes=0,
                    )
                    * evolution.up.occupations[I]
                )
        down_p = np.zeros(shape=(time_indices.shape[0], s.x.shape[0], s.x.shape[0]), dtype=complex)
        for i, I in enumerate(evolution.down.occupied):
            for j, ti in enumerate(time_indices):
                down_p[j, :] += (
                    np.tensordot(
                        evolution.down.td_orbitals[ti, :, i],
                        evolution.down.td_orbitals[ti, :, i].conj(),
                        axes=0,
                    )
                    * evolution.down.occupations[I]
                )
        p = up_p + down_p
        if return_spins:
            return p, up_p, down_p
        else:
            return p

    else:
        raise AttributeError("State or Evolution must be provided.")


def kinetic_energy(
    s: iDEA.system.System,
    state: iDEA.state.SingleBodyState = None,
    evolution: iDEA.state.SingleBodyEvolution = None,
) -> Union[float, np.ndarray]:
    r"""
    Compute the kinetic energy of a non_interacting state.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.SingleBodyState, State. (default = None)
    |     evolution: iDEA.state.SingleBodyEvolution, Evolution. (default = None)

    | Returns:
    |     E_k: float or np.ndarray, Kinetic energy, or evolution of kinetic energy.
    """
    if state is not None and type(state) == iDEA.state.ManyBodyState:
        K = iDEA.methods.interacting.kinetic_energy_operator(s)
        return observable(s, K, state=state)

    if state is not None and type(state) == iDEA.state.SingleBodyState:
        K = iDEA.methods.non_interacting.kinetic_energy_operator(s)
        return observable(s, K, state=state)

    if evolution is not None and type(evolution) == iDEA.state.ManyBodyEvolution:
        K = iDEA.methods.interacting.kinetic_energy_operator(s)
        return observable(s, K, evolution=evolution)

    if evolution is not None and type(evolution) == iDEA.state.SingleBodyEvolution:
        K = iDEA.methods.non_interacting.kinetic_energy_operator(s)
        return observable(s, K, evolution=evolution)

    else:
        raise AttributeError("State or Evolution must be provided.")


def external_potential(s: iDEA.system.System) -> np.ndarray:
    r"""
    Compute the external potential.

    | Args:
    |     s: iDEA.system.System, System object.

    | Returns:
    |     v_ext: np.ndarray, External potential of the system.
    """
    return s.v_ext


def external_energy(s: iDEA.system.System, n: np.ndarray, v_ext: np.ndarray) -> Union[float, np.ndarray]:
    r"""
    Compute the external energy from a density and external potential.

    Args:
    |     s: iDEA.system.System, System object.
    |     n: np.ndarray, Charge density of the system.
    |     v_ext: np.ndarray, External potential of the system.

    Returns:
    |     E_ext: float or np.ndarray, External energy, or evolution of external energy.
    """
    if len(n.shape) == 1:
        E_ext = np.dot(n, v_ext) * s.dx
        return E_ext

    elif len(n.shape) == 2:
        E_ext = np.zeros(shape=n.shape[0])
        for j in range(E_ext.shape[0]):
            E_ext[j] = np.dot(n[j, :], v_ext[:]) * s.dx
        return E_ext

    else:
        raise AttributeError(f"Expected array of shape 1 or 2, got {n.shape} instead.")


def hartree_potential(s: iDEA.system.System, n: np.ndarray) -> np.ndarray:
    r"""
    Compute the Hartree potential from a density.

    | Args:
    |     s: iDEA.system.System, System object.
    |     n: np.ndarray, Charge density of the system.

    | Returns:
    |     v_h: np.ndarray, Hartree potential, or evolution of Hartree potential.
    """
    if len(n.shape) == 1:
        v_h = np.dot(n, s.v_int) * s.dx
        return v_h

    elif len(n.shape) == 2:
        v_h = np.zeros_like(n)
        for j in range(v_h.shape[0]):
            v_h[j, :] = np.dot(n[j, :], s.v_int[:, :]) * s.dx
        return v_h

    else:
        raise AttributeError(f"Expected array of shape 1 or 2, got {n.shape} instead.")


def hartree_energy(s: iDEA.system.System, n: np.ndarray, v_h: np.ndarray) -> Union[float, np.ndarray]:
    r"""
    Compute the Hartree energy from a density and Hartree potential.

    | Args:
    |     s: iDEA.system.System, System object.
    |     n: np.ndarray, Charge density of the system.
    |     v_h: np.ndarray, Hartree potential of the system.


    | Returns:
    |     E_h: float or np.ndarray, Hartree energy, or evolution of Hartree energy.
    """
    if len(n.shape) == 1:
        E_h = 0.5 * np.dot(n, v_h) * s.dx
        return E_h

    elif len(n.shape) == 2:
        E_h = np.zeros(shape=n.shape[0])
        for j in range(E_h.shape[0]):
            E_h[j] = 0.5 * np.dot(n[j, :], v_h[j, :]) * s.dx
        return E_h

    else:
        raise AttributeError(f"Expected array of shape 1 or 2, got {n.shape} instead.")


def exchange_potential(s: iDEA.system.System, p: np.ndarray) -> np.ndarray:
    r"""
    Compute the exchange potential from a density matrix.

    | Args:
    |     s: iDEA.system.System, System object.
    |     p: np.ndarray, Density matrix of the system.

    | Returns:
    |     v_x: np.ndarray, Exchange potential, or evolution of exchange potential.
    """
    if len(p.shape) == 2:
        v_x = -p * s.v_int
        return v_x

    elif len(p.shape) == 3:
        v_x = np.zeros_like(p)
        for j in range(v_x.shape[0]):
            v_x[j, :, :] = -p[j, :, :] * s.v_int[:, :]
        return v_x

    else:
        raise AttributeError(f"Expected array of shape 1 or 2, got {p.shape} instead.")


def exchange_energy(s: iDEA.system.System, p: np.ndarray, v_x: np.ndarray) -> Union[float, np.ndarray]:
    r"""
    Compute the exchange energy from a density matrix and exchange potential.

    | Args:
    |     s: iDEA.system.System, System object.
    |     p: np.ndarray, Density matrix of the system.
    |     v_x: np.ndarray, Exchange potential of the system.

    | Returns:
    |     E_x: float or np.ndarray, Exchange energy, or evolution of exchange energy.
    """
    if len(p.shape) == 2:
        E_x = 0.5 * np.tensordot(p, v_x, axes=2) * s.dx * s.dx
        return E_x

    elif len(p.shape) == 3:
        E_x = np.zeros(shape=p.shape[0], dtype=complex)
        for j in range(E_x.shape[0]):
            E_x[j] = 0.5 * np.tensordot(p[j, :, :].T, v_x[j, :, :], axes=2) * s.dx * s.dx
        return E_x.real

    else:
        raise AttributeError(f"Expected array of shape 1 or 2, got {p.shape} instead.")


def single_particle_energy(s: iDEA.system.System, state: iDEA.state.SingleBodyState) -> float:
    r"""
    Compute the single particle energy of a single particle state.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.SingleBodyState, State.

    | Returns:
    |     E: float, Single particle energy.
    """
    return np.sum(state.up.energies[:] * state.up.occupations[:]) + np.sum(
        state.down.energies[:] * state.down.occupations[:]
    )


def _placeholder(
    s: iDEA.system.System,
    state: Union[iDEA.state.SingleBodyState, iDEA.state.ManyBodyState] = None,
    evolution: Union[iDEA.state.SingleBodyEvolution, iDEA.state.ManyBodyEvolution] = None,
    return_spins: bool = False,
) -> Union[float, np.ndarray]:
    r"""
    Placeholer function. Use this as a template when constructing observable methods.

    | Args:
    |     s: iDEA.system.System, System object.
    |     state: iDEA.state.SingleBodyState or iDEA.state.ManyBodyState, State. (default = None)
    |     evolution: iDEA.state.SingleBodyEvolution or iDEA.state.ManyBodyEvolution, Evolution. (default = None)
    |     return_spins: bool, True to also return the spin placeholer: total, up, down. (default = False)

    | Returns:
    |     observable: float or np.ndarray, Placeholer.
    """
    if state is not None and type(state) == iDEA.state.SingleBodyState:
        raise NotImplementedError()
    if state is not None and type(state) == iDEA.state.ManyBodyState:
        raise NotImplementedError()
    if evolution is not None and type(evolution) == iDEA.state.SingleBodyEvolution:
        raise NotImplementedError()
    if evolution is not None and type(evolution) == iDEA.state.ManyBodyEvolution:
        raise NotImplementedError()
    else:
        raise AttributeError("State or Evolution must be provided.")
