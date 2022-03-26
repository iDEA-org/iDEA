import copy
import string
import itertools
from typing import Union
import numpy as np
import iDEA.system
import iDEA.state
import iDEA.methods.non_interacting
import iDEA.methods.interacting


def observable(s: iDEA.system.System, observable_operator: np.ndarray, state: Union[iDEA.state.SingleBodyState, iDEA.state.ManyBodyState] = None, evolution: Union[iDEA.state.SingleBodyEvolution, iDEA.state.ManyBodyEvolution] = None, return_spins: bool = False) -> Union[float, np.ndarray]:

    """
    Placeholer function. Use this as a template when constructing observable methods.

    Args:
        s: iDEA.system.System, System object.
        observable_operator: np.ndarray, Obserbable operator.
        state: iDEA.state.SingleBodyState or iDEA.state.ManyBodyState, State. (default = None)
        evolution: iDEA.state.SingleBodyEvolution or iDEA.state.ManyBodyEvolution, Evolution. (default = None)
        return_spins: bool, True to also return the spin observables: total, up, down. (default = False)

    Returns:
        observable: float or np.ndarray, Observable.
    """
    if state is not None and type(state) == iDEA.state.SingleBodyState:
        up_O = 0.0
        for i in range(state.up.orbitals.shape[1]):
            up_O += np.vdot(state.up.orbitals[:,i], np.dot(observable_operator, state.up.orbitals[:,i])) * state.up.occupations[i] * s.dx
        down_O = 0.0
        for i in range(state.down.orbitals.shape[1]):
            down_O += np.vdot(state.down.orbitals[:,i], np.dot(observable_operator, state.down.orbitals[:,i])) * state.down.occupations[i] * s.dx
        O = up_O + down_O
        if return_spins:
            return  O, up_O, down_O
        else:
            return O

    if state is not None and type(state) == iDEA.state.ManyBodyState:
        raise NotImplementedError() 

    if evolution is not None and type(evolution) == iDEA.state.SingleBodyEvolution:
        up_O = np.zeros(shape=evolution.t.shape, dtype=complex)
        for i, I in enumerate(evolution.up.occupied):
            for j, ti in enumerate(evolution.t):
                up_O[j] += np.vdot(evolution.up.td_orbitals[j,:,i], np.dot(observable_operator, evolution.up.td_orbitals[j,:,i])) * evolution.up.occupations[I] * s.dx
        down_O = np.zeros(shape=evolution.t.shape, dtype=complex)
        for i, I in enumerate(evolution.down.occupied):
            for j, ti in enumerate(evolution.t):
                down_O[j] += np.vdot(evolution.down.td_orbitals[j,:,i], np.dot(observable_operator, evolution.down.td_orbitals[j,:,i])) * evolution.down.occupations[I] * s.dx
        O = up_O + down_O
        if return_spins:
            return  O.real, up_O.real, down_O.real
        else:
            return O.real

    if evolution is not None and type(evolution) == iDEA.state.ManyBodyEvolution:
        raise NotImplementedError()
        
    else:
        raise AttributeError(f"State or Evolution must be provided.")


def kinetic_energy(s: iDEA.system.System, state: iDEA.state.SingleBodyState = None, evolution: iDEA.state.SingleBodyEvolution = None) -> Union[float, np.ndarray]:
    """
    Compute the kinetic energy of a non_interacting state.

    Args:
        s: iDEA.system.System, System object.
        state: iDEA.state.SingleBodyState, State. (default = None)
        evolution: iDEA.state.SingleBodyEvolution, Evolution. (default = None)

    Returns:
        energy_kin: float or np.ndarray, Total energy, or evolution of total energy.
    """
    if state is not None:
        K = iDEA.methods.non_interacting.kinetic_energy_operator(s)
        return iDEA.observables.observable(s, K, state=state)
    elif evolution is not None:
        K = iDEA.methods.non_interacting.kinetic_energy_operator(s)
        return iDEA.observables.observable(s, K, evolution=evolution)
    else:
        raise AttributeError(f"State or Evolution must be provided.")


def density(s: iDEA.system.System, state: Union[iDEA.state.SingleBodyState, iDEA.state.ManyBodyState] = None, evolution: Union[iDEA.state.SingleBodyEvolution, iDEA.state.ManyBodyEvolution] = None, return_spins: bool = False) -> np.ndarray:
    """
    Compute the charge density of a non_interacting state.

    Args:
        s: iDEA.system.System, System object.
        state: iDEA.state.SingleBodyState or iDEA.state.ManyBodyState, State. (default = None)
        evolution: iDEA.state.SingleBodyEvolution or iDEA.state.ManyBodyEvolution, Evolution. (default = None)
        return_spins: bool, True to also return the spin densities: total, up, down. (default = False)

    Returns:
        n: float or np.ndarray, Charge density, or evolution of charge density.
    """
    if state is not None and type(state) == iDEA.state.SingleBodyState:
        up_n = np.zeros(shape=s.x.shape[0])
        down_n = np.zeros(shape=s.x.shape[0])
        for i in range(state.up.orbitals.shape[1]):
            up_n += abs(state.up.orbitals[:,i])**2 * state.up.occupations[i]
        for i in range(state.down.orbitals.shape[1]):
            down_n += abs(state.down.orbitals[:,i])**2 * state.down.occupations[i]
        n = up_n + down_n
        if return_spins:
            return n, up_n, down_n
        else:
            return n

    if state is not None and type(state) == iDEA.state.ManyBodyState:
        spin_densities = np.zeros(shape=(s.x.shape[0], 2))
        for i in range(s.x.shape[0]):
            for k in range(2):
                spin_densities[i,k] = np.sum(abs(state.full[i,k,...])**2)*s.dx**(s.count - 1) * s.count
        up_n = spin_densities[:,0]
        down_n = spin_densities[:,1]
        n = up_n + down_n
        if return_spins:
            return n, up_n, down_n
        else:
            return n

    if evolution is not None and type(evolution) == iDEA.state.SingleBodyEvolution:
        up_n = np.zeros(shape=(evolution.t.shape[0], s.x.shape[0]))
        for i, I in enumerate(evolution.up.occupied):
            for j, ti in enumerate(evolution.t):
                up_n[j,:] += abs(evolution.up.td_orbitals[j,:,i])**2*evolution.up.occupations[I]
        down_n = np.zeros(shape=(evolution.t.shape[0], s.x.shape[0]))
        for i, I in enumerate(evolution.down.occupied):
            for j, ti in enumerate(evolution.t):
                down_n[j,:] += abs(evolution.down.td_orbitals[j,:,i])**2*evolution.down.occupations[I]
        n = up_n + down_n
        if return_spins:
            return n, up_n, down_n
        else:
            return n

    if evolution is not None and type(evolution) == iDEA.state.ManyBodyEvolution:
        spin_densities = np.zeros(shape=(evolution.t.shape[0], s.x.shape[0], 2))
        for j, ti in enumerate(evolution.t):
            l = string.ascii_lowercase[:s.count]
            L = string.ascii_uppercase[:s.count]
            st = l + ',' + L + '->' + ''.join([i for sub in list(zip(l,L)) for i in sub])
            full = np.einsum(st, evolution.td_space[j,...], evolution.spin)
            L = list(zip(list(range(0, s.count*2, 2)), list(range(1, s.count*2, 2))))
            perms = itertools.permutations(list(range(s.count)))
            full_copy = copy.deepcopy(full)
            full = np.zeros_like(full_copy)
            for p in perms:
                indices = list(itertools.chain(*[L[e] for e in p]))
                full += iDEA.methods.interacting._permutation_parity(p) * np.moveaxis(full_copy, list(range(s.count*2)), indices)
            full = full / np.sqrt(np.sum(abs(full)**2)*s.dx**s.count)
            for i in range(s.x.shape[0]):
                for k in range(2):
                    spin_densities[j,i,k] = np.sum(abs(full[i,k,...])**2)*s.dx**(s.count - 1) * s.count
        up_n = spin_densities[:,:,0]
        down_n = spin_densities[:,:,1]
        n = up_n + down_n
        if return_spins:
            return n, up_n, down_n
        else:
            return n
    else:
        raise AttributeError(f"State or Evolution must be provided.")


def external_energy(s: iDEA.system.System, n: np.ndarray, state: Union[iDEA.state.SingleBodyState, iDEA.state.ManyBodyState] = None, evolution: Union[iDEA.state.SingleBodyEvolution, iDEA.state.ManyBodyEvolution] = None) -> Union[float, np.ndarray]:
    """
    Compute the charge density matrix of a non_interacting state.

    Args:
        s: iDEA.system.System, System object.
        n, Charge density of the system [If None this will be computed]. (default = None)
        state: iDEA.state.SingleBodyState or iDEA.state.ManyBodyState, State. (default = None)
        evolution: iDEA.state.SingleBodyEvolution or iDEA.state.ManyBodyEvolution, Evolution. (default = None)

    Returns:
        e_ext: float or np.ndarray, External energy, or evolution of external energy.
    """
    if n is not None:
        pass

    elif n is None and state is not None:
        n = density(state=state)

    elif n is None and evolution is not None:
        n = density(evolution=evoluation)

    else:
        raise AttributeError(f"Density, State or Evolution must be provided.")

    if len(n.shape) == 1:
        e_ext = np.dot(n, s.v_ext) * s.dx
        return e_ext
        
    elif len(n.shape) == 2:
        e_ext = np.zeros(shape=n.shape[0])
        for j in range(n.shape[0]):
            e_ext[j] = np.dot(n[j,:], s.v_ext) * s.dx
        return e_ext

    else:
        raise AttributeError(f"Expected array of shape 1 or 2, got {n.shape} instead.")


def density_matrix(s: iDEA.system.System, state: Union[iDEA.state.SingleBodyState, iDEA.state.ManyBodyState] = None, evolution: Union[iDEA.state.SingleBodyEvolution, iDEA.state.ManyBodyEvolution] = None, return_spins: bool = False) -> np.ndarray:
    """
    Compute the charge density matrix of a non_interacting state.

    Args:
        s: iDEA.system.System, System object.
        state: iDEA.state.SingleBodyState or iDEA.state.ManyBodyState, State. (default = None)
        evolution: iDEA.state.SingleBodyEvolution or iDEA.state.ManyBodyEvolution, Evolution. (default = None)
        return_spins: bool, True to also return the spin density matrices: total, up, down. (default = False)

    Returns:
        p: float or np.ndarray, Charge density matrix, or evolution of charge density matrix.
    """
    if state is not None and type(state) == iDEA.state.SingleBodyState:
        up_p = np.zeros(shape=s.x.shape*2)
        down_p = np.zeros(shape=s.x.shape*2)
        for i in range(state.up.orbitals.shape[1]):
            up_p += np.tensordot(state.up.orbitals[:,i].conj(), state.up.orbitals[:,i], axes=0) * state.up.occupations[i]
        for i in range(state.down.orbitals.shape[1]):
            down_p += np.tensordot(state.down.orbitals[:,i].conj(), state.down.orbitals[:,i], axes=0) * state.down.occupations[i]
        p = up_p + down_p
        if return_spins:
            return p, up_p, down_p
        else:
            return p

    if state is not None and type(state) == iDEA.state.ManyBodyState:
        tosum = [2,3,4,5]
        spin_p = np.tensordot(state.full, state.full.conj(), axes=(tosum,tosum)).diagonal(axis1=1, axis2=3)*s.dx**(s.count - 1) * s.count
        up_p = spin_p[:,:,0]
        down_p = spin_p[:,:,1]
        p = up_p + down_p
        if return_spins:
            return p, up_p, down_p
        else:
            return p

    if evolution is not None and type(evolution) == iDEA.state.SingleBodyEvolution:
        up_p = np.zeros(shape=(evolution.t.shape[0], s.x.shape[0], s.x.shape[0]), dtype=complex)
        for i, I in enumerate(evolution.up.occupied):
            for j, ti in enumerate(evolution.t):
                up_p[j,:] += np.tensordot(evolution.up.td_orbitals[j,:,i].conj(), evolution.up.td_orbitals[j,:,i], axes=0) * evolution.up.occupations[I]
        down_p = np.zeros(shape=(evolution.t.shape[0], s.x.shape[0], s.x.shape[0]), dtype=complex)
        for i, I in enumerate(evolution.down.occupied):
            for j, ti in enumerate(evolution.t):
                down_p[j,:] += np.tensordot(evolution.down.td_orbitals[j,:,i].conj(), evolution.down.td_orbitals[j,:,i], axes=0) * evolution.down.occupations[I]
        p = up_p + down_p
        if return_spins:
            return p, up_p, down_p
        else:
            return p

    if evolution is not None and type(evolution) == iDEA.state.ManyBodyEvolution:
        up_p = np.zeros(shape=(evolution.t.shape[0], s.x.shape[0], s.x.shape[0]), dtype=complex)
        for i, I in enumerate(evolution.up.occupied):
            for j, ti in enumerate(evolution.t):
                up_p[j,:] += np.tensordot(evolution.up.td_orbitals[j,:,i].conj(), evolution.up.td_orbitals[j,:,i], axes=0) * evolution.up.occupations[I]
        down_p = np.zeros(shape=(evolution.t.shape[0], s.x.shape[0], s.x.shape[0]), dtype=complex)
        for i, I in enumerate(evolution.down.occupied):
            for j, ti in enumerate(evolution.t):
                down_p[j,:] += np.tensordot(evolution.down.td_orbitals[j,:,i].conj(), evolution.down.td_orbitals[j,:,i], axes=0) * evolution.down.occupations[I]
        p = up_p + down_p
        if return_spins:
            return p, up_p, down_p
        else:
            return p

    else:
        raise AttributeError(f"State or Evolution must be provided.")


def _placeholder(s: iDEA.system.System, state: Union[iDEA.state.SingleBodyState, iDEA.state.ManyBodyState] = None, evolution: Union[iDEA.state.SingleBodyEvolution, iDEA.state.ManyBodyEvolution] = None, return_spins: bool = False) -> Union[float, np.ndarray]:

    """
    Placeholer function. Use this as a template when constructing observable methods.

    Args:
        s: iDEA.system.System, System object.
        state: iDEA.state.SingleBodyState or iDEA.state.ManyBodyState, State. (default = None)
        evolution: iDEA.state.SingleBodyEvolution or iDEA.state.ManyBodyEvolution, Evolution. (default = None)
        return_spins: bool, True to also return the spin placeholer: total, up, down. (default = False)

    Returns:
        observable: float or np.ndarray, Placeholer.
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
        raise AttributeError(f"State or Evolution must be provided.")


# def one_body_reduced_density_matrix(s, orbitals):
#     r"""Constructs the one-body reduced density matrix from single-particle orbitals.

#     .. math:: \rho(x,x') = \sum^N_{n=1}\phi_n(x)\phi^*_n(x')

#     s: System
#         System object.
#     orbitals: np.ndarray
#         Array of normalised orbitals, indexed as orbitals[space,orbital_number].

#     returns:
#     n: np.ndarray
#         One body reduced density matrix.
#     """
#     p = np.zeros(shape=(s.x.shape[0], s.x.shape[0]), dtype=np.complex)
#     for i in range(s.NE):
#         p += np.tensordot(orbitals[:, i].conj(), orbitals[:, i], axes=0)
#     return p


# def total_energy(s, energies):
#     """Calculates the total energy from single particle energies.

#     s: System
#         System object.
#     energies: np.ndarray
#         Array of single particle energies.

#     returns:
#     E: float
#         Total energy.
#     """
#     E = np.sum(energies[: s.NE])
#     return E


# def ionisation_potential(s, energies):
#     """Calculates the ionisation potential from single particle energies.

#     s: System
#         System object.
#     energies: np.ndarray
#         Array of single particle energies.

#     returns:
#     ip: float
#         Ionisation potential.
#     """
#     ip = -energies[s.NE - 1]
#     return ip


# def electron_affinity(s, energies):
#     """Calculates the electron affinity from single particle energies.

#     s: System
#         System object.
#     energies: np.ndarray
#         Array of single particle energies.

#     returns:
#     ea: float
#         Electron affinity.
#     """
#     ea = -energies[s.NE]
#     return ea


# def single_particle_gap(s, energies):
#     """Calculates the single particle gap from single particle energies.

#     s: System
#         System object.
#     energies: np.ndarray
#         Array of single particle energies.

#     returns:
#     gap: float
#         Single particle gap.
#     """
#     gap = energies[s.NE] - energies[s.NE - 1]
#     return gap
