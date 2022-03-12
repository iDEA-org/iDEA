"""Contains all interacting functionality and solvers."""


import copy
import string
import itertools
import functools
from typing import Union
import numpy as np
import scipy as sp
import scipy.sparse as sps
import numpy.linalg as npla
import scipy.linalg as spla
import scipy.sparse.linalg as spsla
import iDEA.system
import iDEA.state
import iDEA.observables
import iDEA.methods.non_interacting


def kinetic_energy_operator(s: iDEA.system.System) -> sps.dia_matrix:
    """
    Compute many-particle kinetic energy operator as a matrix.

    This is built using a given number of finite differences to represent the second derivative.
    The number of differences taken is defined in s.stencil.

    Args:
        s: iDEA.system.System, System object.

    Returns:
        K: sps.dia_matrix, Kintetic energy operator.
    """
    k = iDEA.methods.non_interacting.kinetic_energy_operator(s)
    k = sps.dia_matrix(k)
    I = sps.identity(s.x.shape[0], format='dia')
    partial_operators = lambda A, B, k, n: (A if i + k == n - 1 else B for i in range(n))
    fold_partial_operators = lambda f, po: functools.reduce(lambda acc, val: f(val, acc, format='dia'), po)
    generate_terms = lambda f, A, B, n: (fold_partial_operators(f, partial_operators(A, B, k, n)) for k in range(n))
    terms = generate_terms(sps.kron, h, I, s.count)
    K = sps.dia_matrix((s.x.shape[0]**s.count,)*2, dtype=float)
    for term in terms:
        K += term
    return K


def external_potential_operator(s: iDEA.system.System) -> sps.dia_matrix:
    """
    Compute many-particle external potential energy operator as a matrix.

    Args:
        s: iDEA.system.System, System object.

    Returns:
        Vext: sps.dia_matrix, External potential operator.
    """
    vext = iDEA.methods.non_interacting.external_potential_operator(s)
    vext = sps.dia_matrix(vext)
    I = sps.identity(s.x.shape[0], format='dia')
    partial_operators = lambda A, B, k, n: (A if i + k == n - 1 else B for i in range(n))
    fold_partial_operators = lambda f, po: functools.reduce(lambda acc, val: f(val, acc, format='dia'), po)
    generate_terms = lambda f, A, B, n: (fold_partial_operators(f, partial_operators(A, B, k, n)) for k in range(n))
    terms = generate_terms(sps.kron, h, I, s.count)
    Vext = sps.dia_matrix((s.x.shape[0]**s.count,)*2, dtype=float)
    for term in terms:
        Vext += term
    return Vext


def hamiltonian(s: iDEA.system.System) -> sps.dia_matrix:
    """
    Compute the many-body Hamiltonian.

    Args:
        s: iDEA.system.System, System object.

    Returns:
        H: sps.dia_matrix, Hamiltonian.
    """
    # Construct the non-interacting part of the many-body Hamiltonian
    h = iDEA.methods.non_interacting.hamiltonian(s)
    h = sps.dia_matrix(h)
    I = sps.identity(s.x.shape[0], format='dia')
    partial_operators = lambda A, B, k, n: (A if i + k == n - 1 else B for i in range(n))
    fold_partial_operators = lambda f, po: functools.reduce(lambda acc, val: f(val, acc, format='dia'), po)
    generate_terms = lambda f, A, B, n: (fold_partial_operators(f, partial_operators(A, B, k, n)) for k in range(n))
    terms = generate_terms(sps.kron, h, I, s.count)
    H0 = sps.dia_matrix((s.x.shape[0]**s.count,)*2, dtype=float)
    for term in terms:
        H0 += term

    # Add the interaction part of the many-body Hamiltonian
    symbols = string.ascii_lowercase + string.ascii_uppercase
    if s.count > 1:
        indices = ','.join([''.join(c) for c in itertools.combinations(symbols[:s.count], 2)])
        U = np.log(np.einsum(indices + '->' + symbols[:s.count], *(np.exp(s.v_int),)*int(s.count*(s.count-1)/2)))
        U = sps.diags(U.reshape((H0.shape[0])), format='dia')
    else:
        U = 0.0

    # Construct the total many-body Hamiltonian
    H = H0 + U

    return H


# def total_energy(s: iDEA.system.System, state: iDEA.state.SingleBodyState = None, evolution: iDEA.state.SingleBodyEvolution = None) -> Union[float, np.ndarray]:
#     """
#     Compute the total energy of a non_interacting state.

#     Args:
#         s: iDEA.system.System, System object.
#         state: iDEA.state.SingleBodyState, State. (default = None)
#         evolution: iDEA.state.SingleBodyEvolution, Evolution. (default = None)

#     Returns:
#         E: float or np.ndarray, Total energy, or evolution of total energy.
#     """
#     if state is not None:
#         return np.sum(state.up.energies[:] * state.up.occupations[:]) + np.sum(state.down.energies[:] * state.down.occupations[:])
#     elif evolution is not None:
#         H = hamiltonian(s)
#         return iDEA.observables.observable(s, H, evolution=evolution)
#     else:
#         raise AttributeError(f"State or Evolution must be provided.")


def _permutation_parity(perm):
    perm = list(perm)
    parity = 1
    for i in range(0,len(perm)-1):
        if perm[i] != i:
            parity *= -1
            mn = min(range(i,len(perm)), key=perm.__getitem__)
            perm[i],perm[mn] = perm[mn],perm[i]
    return parity


def antisymmetrize(s, wavefunctions, chis):
    # Hard coded method
    l = string.ascii_lowercase[:s.count]
    L = string.ascii_uppercase[:s.count]
    st = l + 'Y,' + L + 'Y->' + ''.join([i for sub in list(zip(l,L)) for i in sub]) + 'Y'
    wavefunctions = np.einsum(st, wavefunctions, chis)
    L = [(0,1), (2,3), (4,5)]
    perms = itertools.permutations([0,1,2])
    wavefunctions_copy = copy.deepcopy(wavefunctions)
    wavefunctions = np.zeros_like(wavefunctions)
    for p in perms:
        indices = list(itertools.chain(*[L[e] for e in p]))
        wavefunctions += permutation_parity(p) * np.moveaxis(wavefunctions_copy, [0,1,2,3,4,5], indices)

    # Filter out zeros
    allowed_wavefunctions = []
    allowed_energies = []
    for n in range(wavefunctions.shape[-1]):
        if np.allclose(wavefunctions[...,n], np.zeros(wavefunctions.shape[:-1])):
            pass
        else:
            allowed_wavefunctions.append(wavefunctions[...,n])
            allowed_energies.append(energies[n])
    wavefunctions = np.moveaxis(np.array(allowed_wavefunctions), 0, -1)
    energies = np.array(allowed_energies)

    # Normalise
    for k in range(wavefunctions.shape[-1]):
        wavefunctions[...,k] = wavefunctions[...,k] / np.sqrt(np.sum(abs(wavefunctions[...,k])**2)*s.dx**s.count)

    # Filter out duplicates
    allowed_wavefunctions = []
    allowed_energies = []
    for n in range(wavefunctions.shape[-1] - 1):
        if np.allclose(abs(wavefunctions[...,n]), abs(wavefunctions[...,n+1])):
            pass
        else:
            allowed_wavefunctions.append(wavefunctions[...,n])
            allowed_energies.append(energies[n])
    allowed_wavefunctions.append(wavefunctions[...,-1])
    allowed_energies.append(energies[-1])
    wavefunctions = np.moveaxis(np.array(allowed_wavefunctions), 0, -1)
    energies = np.array(allowed_energies)
    return wavefunctions, chis


def solve(s: iDEA.system.System, H: np.ndarray = None, k: int = 0, initial_k = None) -> iDEA.state.ManyBodyState:
    """
    Solves the interacting Schrodinger equation of the given system.

    Args:
        s: iDEA.system.System, System object.
        H: np.ndarray, Hamiltonian [If None this will be computed from s]. (default = None)
        k: int, Energy state to solve for. (default = 0, the ground-state)
        initial_k: int. TODO

    Returns:
        state: iDEA.state.ManyBodyState, Solved state.
    """
    # Construct the Hamiltonian.
    if H is None:
        H = hamiltonian(s)

    # Solve the many-body Schrodinger equation.
    energies, wavefunctions = spsla.eigsh(H.tocsr(), k=5, which='SA') # TODO

    # Reshape and normalise the solutions.
    wavefunctions = wavefunctions.reshape((s.x.shape[0],)*s.count + (wavefunctions.shape[-1],))
    for j in range(wavefunctions.shape[-1]):
        wavefunctions[...,j] = wavefunctions[...,j] / np.sqrt(np.sum(abs(wavefunctions[...,j])**2)*s.dx**s.count)

    # Construct the spin part.
    symbols = string.ascii_lowercase + string.ascii_uppercase
    u = np.array([1,0])
    d = np.array([0,1])
    spin_state = tuple([u if spin == 'u' else d for spin in s.electrons])
    chi = np.einsum(','.join(symbols[:s.count]) + '->' + ''.join(symbols[:s.count]), *spin_state)
    chis = np.zeros(shape=((2,)*s.count + (wavefunctions.shape[-1],)))
    for i in range(wavefunctions.shape[-1]):
        chis[...,i] = chi

    # Antisymmetrize.
    wavefunctions, chis = antisymmetrize(s, wavefunctions, chis)

    # Construct the many-body state.
    state = iDEA.state.ManyBodyState()
    state.space = wavefunctions
    state.spin = chis

    return state


# def propagate(s: iDEA.system.System, state: iDEA.state.SingleBodyState, v_ptrb: np.ndarray, t: np.ndarray, H: np.ndarray = None) -> iDEA.state.Evolution():
#     """
#     propagate a set of orbitals forward in time due to a local pertubation.

#     Args: 
#         s: iDEA.system.System, System object.
#         state: np.ndarray, Array of normalised orbitals, indexed as orbitals[space,orbital_number].
#         v_ptrb: np.ndarray, Local perturbing potential [static or dynamic].
#         t: np.ndarray, Grid of time values.
#         H: np.ndarray, Static Hamiltonian [If None this will be computed from s]. (default = None)

#     Returns:
#         evolution: iDEA.state.TDSingleBodyState, Solved time-dependent state.
#     """
#     if len(v_ptrb.shape) == 1:
#         return _propagate_static(s, state, v_ptrb, t, H=None)
#     elif len(v_ptrb.shape) == 2:
#         return _propagate_dynamic(s, state, v_ptrb, t, H=None)
#     else:
#         raise AttributeError(f"v_ptrb must have shape 1 or 2, got {v_ptrb.shape} instead.")


# def _propagate_static(s: iDEA.system.System, state: iDEA.state.SingleBodyState, v_ptrb: np.ndarray, t: np.ndarray, H: np.ndarray = None) -> iDEA.state.Evolution():
#     """
#     Propagate a set of orbitals forward in time due to a static local pertubation.

#     Args: 
#         s: iDEA.system.System, System object.
#         state: np.ndarray, Array of normalised orbitals, indexed as orbitals[space,orbital_number].
#         v_ptrb: np.ndarray, Local perturbing potential on the grid of x values, indexed as v_ptrb[space].
#         t: np.ndarray, Grid of time values.
#         H: np.ndarray, Static Hamiltonian [If None this will be computed from s]. (default = None)

#     Returns:
#         evolution: iDEA.state.TDSingleBodyState, Solved time-dependent state.
#     """
#     # Construct the unperturbed Hamiltonian.
#     if H is None:
#         H = hamiltonian(s)

#     # Construct the pertubation potential.
#     Vptrb = np.diag(v_ptrb)

#     # Compute timestep.
#     dt = t[1] - t[0]

#     # Construct time propigation operator.
#     U = spla.expm(-1.0j * (H + Vptrb) * dt)

#     # Initilise time-dependent orbitals.
#     td_up_orbitals = np.zeros(shape=(t.shape[0], s.x.shape[0], state.up.occupied.shape[0]), dtype=np.complex)
#     td_down_orbitals = np.zeros(shape=(t.shape[0], s.x.shape[0], state.down.occupied.shape[0]), dtype=np.complex)
#     td_up_orbitals[0, :, :] = state.up.orbitals[:, state.up.occupied]
#     td_down_orbitals[0, :, :] = state.down.orbitals[:, state.down.occupied]

#     # Propagate up orbitals.
#     for i in range(state.up.occupied.shape[0]):
#         for j, ti in enumerate(t):
#             if j != 0:
#                 print("iDEA.methods.non_interacting.propagate: propagating up orbital {0}/{1}, time = {2:.3f}/{3:.3f}".format(i + 1, s.up_count, ti, np.max(t)), end="\r")
#                 td_up_orbitals[j, :, i] = np.dot(U, td_up_orbitals[j - 1, :, i])
#                 norm = npla.norm(td_up_orbitals[j, :, i]) * np.sqrt(s.dx)
#                 td_up_orbitals[j, :, i] /= norm
#         print()

#     # Propagate down orbitals.
#     for i in range(state.down.occupied.shape[0]):
#         for j, ti in enumerate(t):
#             if j != 0:
#                 print("iDEA.methods.non_interacting.propagate: propagating down orbital {0}/{1}, time = {2:.3f}/{3:.3f}".format(i + 1, s.down_count, ti, np.max(t)), end="\r")
#                 td_down_orbitals[j, :, i] = np.dot(U, td_down_orbitals[j - 1, :, i])
#                 norm = npla.norm(td_down_orbitals[j, :, i]) * np.sqrt(s.dx)
#                 td_down_orbitals[j, :, i] /= norm
#         print()

#     # Construct the single-body time-dependent evolution.
#     evolution = iDEA.state.SingleBodyEvolution(state)
#     evolution.up.td_orbitals = td_up_orbitals
#     evolution.down.td_orbitals = td_down_orbitals
#     evolution.v_ptrb = v_ptrb
#     evolution.t = t

#     return evolution


# def _propagate_dynamic(s: iDEA.system.System, state: iDEA.state.SingleBodyState, v_ptrb: np.ndarray, t: np.ndarray, H: np.ndarray = None) -> iDEA.state.Evolution():
#     """
#     Propagate a set of orbitals forward in time due to a dynamic local pertubation.

#     Args: 
#         s: iDEA.system.System, System object.
#         state: np.ndarray, Array of normalised orbitals, indexed as orbitals[space,orbital_number].
#         v_ptrb: np.ndarray, Local perturbing potential on the grid of t and x values, indexed as v_ptrb[time,space].
#         t: np.ndarray, Grid of time values.
#         H: np.ndarray, Static Hamiltonian [If None this will be computed from s]. (default = None)

#     Returns:
#         evolution: iDEA.state.TDSingleBodyState, Solved time-dependent state.
#     """
#     # Construct the unperturbed Hamiltonian.
#     if H is None:
#         H = hamiltonian(s)
#     H = sps.csc_matrix(H)

#     # Compute timestep.
#     dt = t[1] - t[0]

#     # Initilise time-dependent orbitals.
#     td_up_orbitals = np.zeros(shape=(t.shape[0], s.x.shape[0], state.up.occupied.shape[0]), dtype=np.complex)
#     td_down_orbitals = np.zeros(shape=(t.shape[0], s.x.shape[0], state.down.occupied.shape[0]), dtype=np.complex)
#     td_up_orbitals[0, :, :] = state.up.orbitals[:, state.up.occupied]
#     td_down_orbitals[0, :, :] = state.down.orbitals[:, state.down.occupied]

#     # Propagate up orbitals.
#     for i in range(state.up.occupied.shape[0]):
#         for j, ti in enumerate(t):
#             if j != 0:
#                 print("iDEA.methods.non_interacting.propagate: propagating up orbital {0}/{1}, time = {2:.3f}/{3:.3f}".format(i + 1, s.up_count, ti, np.max(t)), end="\r")
#                 Vptrb = sps.diags(v_ptrb[j,:]).tocsc()
#                 td_up_orbitals[j, :, i] = spsla.expm_multiply(spla.expm(-1.0j * (H + Vptrb) * dt), td_up_orbitals[j - 1, :, i])
#                 norm = npla.norm(td_up_orbitals[j, :, i]) * np.sqrt(s.dx)
#                 td_up_orbitals[j, :, i] /= norm
#         print()

#     # Propagate down orbitals.
#     for i in range(state.down.occupied.shape[0]):
#         for j, ti in enumerate(t):
#             if j != 0:
#                 print("iDEA.methods.non_interacting.propagate: propagating down orbital {0}/{1}, time = {2:.3f}/{3:.3f}".format(i + 1, s.down_count, ti, np.max(t)), end="\r")
#                 Vptrb = sps.diags(v_ptrb[j,:]).tocsc()
#                 td_down_orbitals[j, :, i] = spsla.expm_multiply(spla.expm(-1.0j * (H + Vptrb) * dt), td_down_orbitals[j - 1, :, i])
#                 norm = npla.norm(td_down_orbitals[j, :, i]) * np.sqrt(s.dx)
#                 td_down_orbitals[j, :, i] /= norm
#         print()

#     # Construct the single-body time-dependent evolution.
#     evolution = iDEA.state.SingleBodyEvolution(state)
#     evolution.up.td_orbitals = td_up_orbitals
#     evolution.down.td_orbitals = td_down_orbitals
#     evolution.v_ptrb = v_ptrb
#     evolution.t = t

#     return evolution