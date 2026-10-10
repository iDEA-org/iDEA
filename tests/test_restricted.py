"""
iDEA Restricted Single-Body Methods Testing Platform

Regression tests for restricted Hartree-Fock and hybrid calculations (GitHub issue #40).

For a closed-shell system whose unrestricted solution is not spin-broken, the restricted and unrestricted
calculations must give the same ground state. Likewise for a fully spin-polarised system, where the down
channel is empty. Previously the restricted exchange operator was built from the total density matrix,
double counting exchange and making closed-shell restricted Hartree-Fock equivalent to non-interacting electrons.
"""

# Dependencies
import numpy as np
import pytest

import iDEA

# Small grid so the tests run quickly.
x = np.linspace(-10, 10, 101)
v_ext = -2.0 / (abs(x) + 1.0)
v_int = iDEA.interactions.softened_interaction(x)


def build_system(spin):
    return iDEA.system.System(x, v_ext, v_int, electrons=spin)


@pytest.mark.parametrize("spin", ["ud", "uu"])
def test_hartree_fock_restricted_matches_unrestricted(spin):
    s = build_system(spin)
    unrestricted = iDEA.methods.hartree_fock.solve(s, restricted=False, silent=True)
    restricted = iDEA.methods.hartree_fock.solve(s, restricted=True, silent=True)
    E_u = iDEA.methods.hartree_fock.total_energy(s, unrestricted)
    E_r = iDEA.methods.hartree_fock.total_energy(s, restricted)
    assert E_r == pytest.approx(E_u, abs=1e-6)
    assert restricted.up.energies[0] == pytest.approx(unrestricted.up.energies[0], abs=1e-6)
    n_u = iDEA.observables.density(s, unrestricted)
    n_r = iDEA.observables.density(s, restricted)
    assert np.sum(abs(n_r - n_u)) * s.dx < 1e-6


@pytest.mark.parametrize("spin", ["ud", "uu"])
def test_hybrid_restricted_matches_unrestricted(spin):
    s = build_system(spin)
    unrestricted = iDEA.methods.hybrid.solve(s, restricted=False, silent=True)
    restricted = iDEA.methods.hybrid.solve(s, restricted=True, silent=True)
    E_u = iDEA.methods.hybrid.total_energy(s, unrestricted)
    E_r = iDEA.methods.hybrid.total_energy(s, restricted)
    assert E_r == pytest.approx(E_u, abs=1e-6)


def test_restricted_hartree_fock_closed_shell_is_interacting():
    # The two electrons must repel: the restricted eigenvalue must lie well above the one-electron eigenvalue.
    s = build_system("ud")
    s1 = build_system("u")
    restricted = iDEA.methods.hartree_fock.solve(s, restricted=True, silent=True)
    one_electron = iDEA.methods.hartree_fock.solve(s1, silent=True)
    assert restricted.up.energies[0] > one_electron.up.energies[0] + 0.1
