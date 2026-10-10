"""Defines the structures to describe the system states"""

import copy
import pickle
from abc import ABC as Interface

import numpy as np

import iDEA.utilities

__all__ = [
    "State",
    "States",
    "ManyBodyState",
    "ManyBodyStates",
    "SingleBodyState",
    "Evolution",
    "ManyBodyEvolution",
    "SingleBodyEvolution",
]


class State(Interface):
    """Interface class representing a static state."""


class States(Interface):
    """Interface class representing a group of static states."""


class Evolution(Interface):
    """Interface class representing a time-dependent evolution of a state."""


class ManyBodyState(State):
    """State of interacting particles."""

    def __init__(
        self,
        space: np.ndarray = None,
        spin: np.ndarray = None,
        full=None,
        energy=None,
    ):
        r"""
        State of particles in a many-body state.

        This is described by a spatial part
        .. math:: \psi(x_1,x_2,\dots,x_N)
        on the spatial grid, and a spin
        part on the spin grid
        .. math:: \chi(\sigma_1,\sigma_2,\dots,\sigma_N).
        These are NOT necessarily antisymmetric states,
        they can be combined using the antisymmetrisation operaration to produce the full
        wavefunction
        .. math:: \Psi(x_1,\sigma_1,x_2,\sigma_2,\dots,x_N,\sigma_N).

        Note: for states produced by iDEA.methods.interacting.solve, spin is the seed spin product state
        built from s.electrons (e.g. |up down> for "ud") and is identical for every state k. It carries no
        per-state spin information: the actual spin structure of a state (e.g. singlet vs triplet) lives
        only in full. Use iDEA.observables.total_spin_z and iDEA.observables.total_spin_squared to measure it.

        | Args:
        |     space: np.ndarray, Spatial part of the wavefunction on the spatial grid \psi(x_1,x_2,\dots,x_N). (default = None)
        |     spin: np.ndarray, Seed spin product state on the spin grid \chi(\sigma_1,\sigma_2,\dots,\sigma_N), not antisymmetrised and not state-specific (see note above). (default = None)
        |     full: np.ndarray, Total antisymmetrised wavefunction \Psi(x_1,\sigma_1,x_2,\sigma_2,\dots,x_N,\sigma_N). (default = None)
        |     energy: float, Total energy of the state.
        """
        if space is None:
            self.space = iDEA.utilities.ArrayPlaceholder()
        else:
            self.space = space
        if spin is None:
            self.spin = iDEA.utilities.ArrayPlaceholder()
        else:
            self.spin = spin
        if full is None:
            self.full = iDEA.utilities.ArrayPlaceholder()
        else:
            self.full = full
        if energy is None:
            self.energy = 0.0
        else:
            self.energy = energy


class ManyBodyStates(States):
    """A collection of many-body states over a range of energy levels from a single system of interacting particles."""

    def __init__(self, spaces: np.ndarray = None, spins: np.ndarray = None, fulls=None, energies=None):
        r"""
        Collection of many-body states indexed by energy level n.

        Each state is described by a spatial part
        .. math:: \psi(x_1,x_2,\dots,x_N,n)
        on the spatial grid, and a spin part on the spin grid
        .. math:: \chi(\sigma_1,\sigma_2,\dots,\sigma_N,n).
        These are NOT necessarily antisymmetric states;
        they can be combined using the antisymmetrisation operation to produce the full
        wavefunction
        .. math:: \Psi(x_1,\sigma_1,x_2,\sigma_2,\dots,x_N,\sigma_N,n).

        | Args:
        |     spaces: np.ndarray, Spatial part of each wavefunction on the spatial grid \psi(x_1,x_2,\dots,x_N,n). (default = None)
        |     spins: np.ndarray, Spin part of each wavefunction on the spin grid \chi(\sigma_1,\sigma_2,\dots,\sigma_N,n). (default = None)
        |     fulls: np.ndarray, Total antisymmetrised wavefunction of each state \Psi(x_1,\sigma_1,x_2,\sigma_2,\dots,x_N,\sigma_N,n). (default = None)
        |     energies: np.ndarray, Total energy of each state, indexed as energies[n]. (default = None)
        """
        if spaces is None:
            self.spaces = iDEA.utilities.ArrayPlaceholder()
        else:
            self.spaces = spaces
        if spins is None:
            self.spins = iDEA.utilities.ArrayPlaceholder()
        else:
            self.spins = spins
        if fulls is None:
            self.fulls = iDEA.utilities.ArrayPlaceholder()
        else:
            self.fulls = fulls
        if energies is None:
            self.energies = iDEA.utilities.ArrayPlaceholder()
        else:
            self.energies = energies


class SingleBodyState(State):
    r"""
    State of particles in a single-body state.

    This is described by three arrays for each spin channel:

    | up.energies: np.ndarray, Array of single-body energies, indexed as energies[orbital_number].
    | up.orbitals: np.ndarray, Array of single-body orbitals, indexed as orbitals[space,orbital_number].
    | up.occupations: np.ndarray, Array of single-body occupations, indexed as occupations[orbital_number].
    | up.occupied: np.ndarray, Indices of up.occupations that are non-zero, to indicate occupied orbitals.

    | down.energies: np.ndarray, Array of single-body energies, indexed as energies[orbital_number].
    | down.orbitals: np.ndarray, Array of single-body orbitals, indexed as orbitals[space,orbital_number].
    | down.occupations: np.ndarray, Array of single-body occupations, indexed as occupations[orbital_number].
    | down.occupied: np.ndarray, Indices of down.occupations that are non-zero, to indicate occupied orbitals.
    """

    def __init__(self):
        self.up = iDEA.utilities.Container()
        self.down = iDEA.utilities.Container()

        self.up.energies = iDEA.utilities.ArrayPlaceholder()
        self.up.orbitals = iDEA.utilities.ArrayPlaceholder()
        self.up.occupations = iDEA.utilities.ArrayPlaceholder()
        self.up.occupied = iDEA.utilities.ArrayPlaceholder()

        self.down.energies = iDEA.utilities.ArrayPlaceholder()
        self.down.orbitals = iDEA.utilities.ArrayPlaceholder()
        self.down.occupations = iDEA.utilities.ArrayPlaceholder()
        self.down.occupied = iDEA.utilities.ArrayPlaceholder()


class ManyBodyEvolution(Evolution):
    r"""
    Time-dependent evolution of particles in a many-body state.

    In addition to the arrays defined within the initial ManyBodyState, this state is described by three additional arrays:

    | td_space: np.ndarray, Spatial part of the wavefunction on the spatial grid \psi(t,x_1,x_2,\dots,x_N).
    | v_ptrb: np.ndarray, Perturbation potential that this time-dependence was driven by. indexed as v_ptrb[space] if static, and v_ptrb[time,space] if dynamic.
    | t: np.ndarray, Time grid used during evolution.
    """

    def __init__(self, initial_state: ManyBodyState):
        self.space = copy.deepcopy(initial_state.space)
        self.spin = copy.deepcopy(initial_state.spin)
        self.full = copy.deepcopy(initial_state.full)
        self.td_space = iDEA.utilities.ArrayPlaceholder()
        self.v_ptrb = iDEA.utilities.ArrayPlaceholder()
        self.t = iDEA.utilities.ArrayPlaceholder()


class SingleBodyEvolution(Evolution):
    r"""
    Time-dependent evolution of particles in a single-body state.

    In addition to the arrays defined within the initial SingleBodyState, this state is described by four additional arrays:

    | up.td_orbitals: np.ndarray, Array of single-body time-dependend orbitals, indexed as orbitals[time,space,orbital_number].
    | down.td_orbital: np.ndarray, Array of single-body time-dependend orbitals, indexed as orbitals[time,space,orbital_number].
    | v_ptrb: np.ndarray, Perturbation potential that this time-dependence was driven by. indexed as v_ptrb[space] if static, and v_ptrb[time,space] if dynamic.
    | t: np.ndarray, Time grid used during evolution.

    In this case, only the occupied time-dependent orbitals are stored, as only these are propigated.
    """

    def __init__(self, initial_state: SingleBodyState):
        self.up = copy.deepcopy(initial_state.up)
        self.down = copy.deepcopy(initial_state.down)
        self.up.td_orbitals = iDEA.utilities.ArrayPlaceholder()
        self.down.td_orbitals = iDEA.utilities.ArrayPlaceholder()
        self.v_ptrb = iDEA.utilities.ArrayPlaceholder()
        self.t = iDEA.utilities.ArrayPlaceholder()


def save_many_body_state(state: ManyBodyState, file_name: str) -> None:
    r"""
    Save a many body state to a system file.

    | Args:
    |     state: iDEA.state.ManyBodyState, State object to save.
    |     file_name: str, file name.
    """
    with open(file_name, "wb") as f:
        pickle.dump(state, f)


def save_single_body_state(state: SingleBodyState, file_name: str) -> None:
    r"""
    Save a single body state to a system file.

    | Args:
    |     state: iDEA.state.SingleBodyState, State object to save.
    |     file_name: str, file name.
    """
    with open(file_name, "wb") as f:
        pickle.dump(state, f)


def load_many_body_state(file_name: str) -> ManyBodyState:
    r"""
    Load a many body state from an system file.

    | Args:
    |     file_name: str, file name.

    | Returns
    |     system: iDEA.state.ManyBodyState, Loaded State object.
    """
    return pickle.load(open(file_name, "rb"))


def load_single_body_state(file_name: str) -> SingleBodyState:
    r"""
    Load a single body state from an system file.

    | Args:
    |     file_name: str, file name.

    | Returns
    |     system: iDEA.state.SingleBodyState, Loaded State object.
    """
    return pickle.load(open(file_name, "rb"))
