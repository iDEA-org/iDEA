"""Defines the structures to describe the system states"""


from abc import ABC as Interface
import copy
import numpy as np
import iDEA.utilities
import pickle


__all__ = [
    "State",
    "ManyBodyState",
    "SingleBodyState",
    "Evolution",
    "ManyBodyEvolution",
    "SingleBodyEvolution",
]


class State(Interface):
    """Interface class representing a static state."""


class Evolution(Interface):
    """Interface class representing a time-dependent evolution of a state."""


class ManyBodyState(State):
    """State of interacting particles."""

    def __init__(
        self, space: np.ndarray = None, spin: np.ndarray = None, full=None, energy=None, allspace: np.ndarray = None, allspin: np.ndarray= None, allfull=None, allenergy=None
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

        | Args:
        |     space: np.ndarray, Spatial part of the wavefunction on the spatial grid \psi(x_1,x_2,\dots,x_N). (default = None)
        |     spin: np.ndarray, Spin part of the wavefunction on the spin grid \chi(\sigma_1,\sigma_2,\dots,\sigma_N). (default = None)
        |     full: np.ndarray, Total antisymmetrised wavefunction \Psi(x_1,\sigma_1,x_2,\sigma_2,\dots,x_N,\sigma_N). (default = None)
        |     energy: float, Total energy of the state.
        |     allspace: np.ndarray, spatial part of all wavefunctions generated during solving
        |     allspin: np.ndarray, spin part of all wavefunctions generated during solving
        |     allfull: np.ndarray, full wavefunctions for all generated during solving
        |     allenergies: np.ndarray, all energies generated during solving
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
            self.energy = float()
        else:
            self.energy = energy


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
    pickle.dump(state, open(file_name, "wb"))

def save_single_body_state(state: SingleBodyState, file_name: str) -> None:
    r"""
    Save a single body state to a system file.

    | Args:
    |     state: iDEA.state.SingleBodyState, State object to save.
    |     file_name: str, file name.
    """
    pickle.dump(state, open(file_name, "wb"))

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