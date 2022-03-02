from typing import Union
import numpy as np
import iDEA.system
import iDEA.state


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


def charge_density(s: iDEA.system.System, state: Union[iDEA.state.SingleBodyState, iDEA.state.ManyBodyState] = None, evolution: Union[iDEA.state.SingleBodyEvolution, iDEA.state.ManyBodyEvolution] = None, return_spins: bool = False) -> np.ndarray:
    """
    Compute the charge density of a non_interacting state.

    Args:
        s: iDEA.system.System, System object.
        state: iDEA.state.SingleBodyState or iDEA.state.ManyBodyState, State. (default = None)
        evolution: iDEA.state.SingleBodyEvolution or iDEA.state.ManyBodyEvolution, Evolution. (default = None)
        return_spins: bool, True to also return the spin densities: total, up, down. (default = False)

    Returns:
        density: float or np.ndarray, Charge density, or evolution of charge density.
    """
    if state is not None and type(state) == iDEA.state.SingleBodyState:
        up_density = np.zeros(shape=s.x.shape[0])
        down_density = np.zeros(shape=s.x.shape[0])
        for i in range(state.up.orbitals.shape[1]):
            up_density += abs(state.up.orbitals[:,i])**2*state.up.occupations[i]
        for i in range(state.down.orbitals.shape[1]):
            down_density += abs(state.down.orbitals[:,i])**2*state.down.occupations[i]
        density = up_density + down_density
        if return_spins:
            return density, up_density, down_density
        else:
            return density

    if state is not None and type(state) == iDEA.state.ManyBodyState:
        raise NotImplementedError() # TODO

    if evolution is not None and type(evolution) == iDEA.state.SingleBodyEvolution:
        up_density = np.zeros(shape=(evolution.t.shape[0], s.x.shape[0]), dtype=complex)
        for i, I in enumerate(evolution.up.occupied):
            for j, ti in enumerate(evolution.t):
                up_density[j,:] += abs(evolution.up.td_orbitals[j,:,i])**2*evolution.up.occupations[I]
        down_density = np.zeros(shape=(evolution.t.shape[0], s.x.shape[0]), dtype=complex)
        for i, I in enumerate(evolution.down.occupied):
            for j, ti in enumerate(evolution.t):
                down_density[j,:] += abs(evolution.down.td_orbitals[j,:,i])**2*evolution.down.occupations[I]
        density = up_density + down_density
        if return_spins:
            return density.real, up_density.real, down_density.real
        else:
            return density.real

    if evolution is not None and type(evolution) == iDEA.state.ManyBodyEvolution:
        raise NotImplementedError() # TODO

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
