"""Defines the structures to describe the system states"""


from abc import ABC as Interface
import numpy as np
import iDEA.utilities


__all__ = ["State", "ManyBodyState", "SingleBodyState"]

class State(Interface):
    """Interface class representing a state."""
    pass


class ManyBodyState(State):
    """State of interacting particles."""

    def __init__(self, space: np.ndarray = None, spin: np.ndarray = None):
        r"""
        State of particles in a many-body state.

        This is described by a spatial part \psi(x_1,x_2,\dots,x_N) on the spatial grid, and a spin
        part on the spin grid \chi(\sigma_1,\sigma_2,\dots,\sigma_N). These are NOT necessarily antisymmetric states,
        they can be combined using the antisymmetrisation operaration to produce the full 
        wavefunction \Psi(x_1,\sigma_1,x_2,\sigma_2,\dots,x_N,\sigma_N) using iDEA.interacting functionality.
        This full array is not initiliased in this object due to is large redundant size, it is calculated and 
        destroyed when needed.

        Args:
            space: np.ndarray, Spatial part of the wavefunction on the spatial grid \psi(x_1,x_2,\dots,x_N). (default = None)
            spin: np.ndarray, Spin part of the wavefunction on the spin grid \chi(\sigma_1,\sigma_2,\dots,\sigma_N). (default = None)
        """
        if space is None:
            self.space = iDEA.utilities.ArrayPlaceholder()
        else:
            self.space = space
        if spin is None:
            self.spin = iDEA.utilities.ArrayPlaceholder()
        else:
            self.spin = spin
        self.full = iDEA.utilities.ArrayPlaceholder()


class SingleBodyState(State):
    r"""
    State of particles in a single-body state.
    
    This is described by three arrays for each spin channel: 
    
    up.energies: np.ndarray, Array of single-body energies, indexed as energies[orbital_number].
    up.orbitals: np.ndarray, Array of single-body orbitals, indexed as orbitals[space,orbital_number].
    up.occupations: np.ndarray, Array of single-body occupations, indexed as occupations[orbital_number].

    down.energies: np.ndarray, Array of single-body energies, indexed as energies[orbital_number].
    down.orbitals: np.ndarray, Array of single-body orbitals, indexed as orbitals[space,orbital_number].
    down.occupations: np.ndarray, Array of single-body occupations, indexed as occupations[orbital_number].
    """

    def __init__(self):
        self.up = iDEA.utilities.Container()
        self.down = iDEA.utilities.Container()
        
        self.up.energies = iDEA.utilities.ArrayPlaceholder()
        self.up.orbitals = iDEA.utilities.ArrayPlaceholder()
        self.up.occupations = iDEA.utilities.ArrayPlaceholder()

        self.down.energies = iDEA.utilities.ArrayPlaceholder()
        self.down.orbitals = iDEA.utilities.ArrayPlaceholder()
        self.down.occupations = iDEA.utilities.ArrayPlaceholder()