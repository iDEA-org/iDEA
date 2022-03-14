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
    H = H0 + U*1e-10

    return H


def total_energy(s: iDEA.system.System, state: iDEA.state.ManyBodyState = None, evolution: iDEA.state.ManyBodyEvolution = None) -> Union[float, np.ndarray]:
    """
    Compute the total energy of a interacting state.

    Args:
        s: iDEA.system.System, System object.
        state: iDEA.state.ManyBodyState, State. (default = None)
        evolution: iDEA.state.ManyBodyEvolution, Evolution. (default = None)

    Returns:
        E: float or np.ndarray, Total energy, or evolution of total energy.
    """
    if state is not None:
        return state.energy
    elif evolution is not None:
        raise NotImplementedError() # TODO
    else:
        raise AttributeError(f"State or Evolution must be provided.")


def _permutation_parity(p):
    """
    Compute the permulation paritiy of a given permutation.

    Args:
        p: tuple, Permutation.

    Returns:
        parity: float, Permutation parity.
    """
    p = list(p)
    parity = 1
    for i in range(0,len(p)-1):
        if p[i] != i:
            parity *= -1
            mn = min(range(i,len(p)), key=p.__getitem__)
            p[i], p[mn] = p[mn], p[i]
    return parity


def antisymmetrize(s, spaces, spins, energies):
    """
    Antisymmetrize the solution to the Schrodinger equation.

    Args:
        s: iDEA.system.System, System object.
        spaces: np.ndarray, Spatial parts of the wavefunction.
        spins: np.ndarray, Spin parts of the wavefunction.
        energies: np.ndarray, Energies.

    Returns:
        fulls: np.ndarray, Full anantisymmetrized wavefunction.
        spaces: np.ndarray, Spatial parts of the wavefunction.
        spins: np.ndarray, Spin parts of the wavefunction.
        energies: np.ndarray, Energies.

    """
    # Perform antisymmetrization.
    l = string.ascii_lowercase[:s.count]
    L = string.ascii_uppercase[:s.count]
    st = l + 'Y,' + L + 'Y->' + ''.join([i for sub in list(zip(l,L)) for i in sub]) + 'Y'
    fulls = np.einsum(st, spaces, spins)
    L = list(zip(list(range(0, s.count*2, 2)), list(range(1, s.count*2, 2))))
    perms = itertools.permutations(list(range(s.count)))
    fulls_copy = copy.deepcopy(fulls)
    fulls = np.zeros_like(fulls)
    for p in perms:
        indices = list(itertools.chain(*[L[e] for e in p]))
        fulls += _permutation_parity(p) * np.moveaxis(fulls_copy, list(range(s.count*2)), indices)

    # Filter out zeros.
    allowed_fulls = []
    allowed_energies = []
    for n in range(fulls.shape[-1]):
        if np.allclose(fulls[...,n], np.zeros(fulls.shape[:-1])):
            pass
        else:
            allowed_fulls.append(fulls[...,n])
            allowed_energies.append(energies[n])
    fulls = np.moveaxis(np.array(allowed_fulls), 0, -1)
    energies = np.array(allowed_energies)

    # Normalise.
    for k in range(fulls.shape[-1]):
        fulls[...,k] = fulls[...,k] / np.sqrt(np.sum(abs(fulls[...,k])**2)*s.dx**s.count)

    # Filter out duplicates.
    allowed_fulls = []
    allowed_energies = []
    for n in range(fulls.shape[-1] - 1):
        if np.allclose(abs(fulls[...,n]), abs(fulls[...,n+1])):
            pass
        else:
            allowed_fulls.append(fulls[...,n])
            allowed_energies.append(energies[n])
    allowed_fulls.append(fulls[...,-1])
    allowed_energies.append(energies[-1])
    fulls = np.moveaxis(np.array(allowed_fulls), 0, -1)
    spaces = spaces[...,:fulls.shape[-1]]
    spins = spins[...,:fulls.shape[-1]]
    energies = np.array(allowed_energies)

    return fulls, spaces, spins, energies


def _estimate_kp(s, k):
    """
    Estimate the solution to the Schrodinger Equation needed to eachive given energy state.

    Args:
        s: iDEA.system.System, System object.
        k: int, Target energy state.

    Returns:
        kp: int, Extimate of kp.
    """
    return (abs(s.up_count - s.down_count) + 1)**2 * s.count * (k + 1)


def solve(s: iDEA.system.System, H: np.ndarray = None, k: int = 0, kp = None) -> iDEA.state.ManyBodyState:
    """
    Solves the interacting Schrodinger equation of the given system.

    Args:
        s: iDEA.system.System, System object.
        H: np.ndarray, Hamiltonian [If None this will be computed from s]. (default = None)
        k: int, Energy state to solve for. (default = 0, the ground-state)
        kp: int. TODO

    Returns:
        state: iDEA.state.ManyBodyState, Solved state.
    """
    # Construct the Hamiltonian.
    if H is None:
        H = hamiltonian(s)

    # Estimate the level of excitation. 
    if kp is None:
        kp = _estimate_kp(s, k)
        print(kp)

    # Solve the many-body Schrodinger equation.
    energies, spaces = spsla.eigsh(H.tocsr(), k=kp, which='SA')

    # Reshape and normalise the solutions.
    spaces = spaces.reshape((s.x.shape[0],)*s.count + (spaces.shape[-1],))
    for j in range(spaces.shape[-1]):
        spaces[...,j] = spaces[...,j] / np.sqrt(np.sum(abs(spaces[...,j])**2)*s.dx**s.count)

    # Construct the spin part.
    symbols = string.ascii_lowercase + string.ascii_uppercase
    u = np.array([1,0])
    d = np.array([0,1])
    spin_state = tuple([u if spin == 'u' else d for spin in s.electrons])
    spin = np.einsum(','.join(symbols[:s.count]) + '->' + ''.join(symbols[:s.count]), *spin_state)
    spins = np.zeros(shape=((2,)*s.count + (spaces.shape[-1],)))
    for i in range(spaces.shape[-1]):
        spins[...,i] = spin

    # Antisymmetrize.
    fulls, spaces, spins, energies = antisymmetrize(s, spaces, spins, energies)

    # Construct the many-body state.
    state = iDEA.state.ManyBodyState()
    state.space = spaces[...,k]
    state.spin = spins[...,k]
    state.full = fulls[...,k]
    state.energy = energies[k]

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