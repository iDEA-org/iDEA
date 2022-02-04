import numpy as np


# def charge_density(s, orbitals):
#     r"""Compute charge density from given non-interacting orbitals.

#     .. math:: n(x) = \sum_j^\mathrm{occ} = \phi_j^*(x)\phi_j(x)

#     s: System
#         System object.
#     orbitals: np.ndarray
#         Array of normalised orbitals, indexed as orbitals[space,orbital_number] or [time,space,orbital_number].

#     returns:
#     n: np.ndarray
#         Charge density. indexed as [space] or [time,space]
#     """
#     if len(orbitals.shape) == 3:
#         n = np.zeros(shape=(orbitals.shape[0], orbitals.shape[1]))
#         for j in range(orbitals.shape[0]):
#             n[j, :] = charge_density(s, orbitals[j, :, :])
#         return n
#     elif len(orbitals.shape) == 2:
#         occupied = orbitals[:, : s.NE]
#         n = np.sum(occupied.conj() * occupied, axis=1).real
#         return n
#     else:
#         pass  # TODO 


# def probability_density(s, orbitals):
#     r"""Compute probability density from given non-interacting orbitals.

#     .. math:: n(x) =\frac{1}{N} \sum_j^\mathrm{occ} \phi_j^*(x)\phi_j(x)

#     s: System
#         System object.
#     orbitals: np.ndarray
#         Array of normalised orbitals, indexed as orbitals[space,orbital_number].

#     returns:
#     n: np.ndarray
#         Charge density.
#     """
#     occupied = orbitals[:, : s.NE]
#     n = np.sum(occupied.conj() * occupied, axis=1).real / s.NE
#     return n


# def orbital_density(s, orbital):
#     r"""Compute charge density for a given non-interacting orbital.

#     .. math:: n(x) = \phi^*(x)\phi(x)

#     s: System
#         System object.
#     orbital: np.ndarray
#         Normalised orbital.

#     returns:
#     n: np.ndarray
#         Orbital density.
#     """
#     n = (orbital.conj() * orbital).real
#     return n


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
