"""
Tests for one-body observables (and the interaction energy) of exact many-body states.

The expectation value of a one-body operator is evaluated from the one-body reduced density matrix. These tests check
it against the brute-force many-body operator, check that the exact energy decomposition (kinetic + external +
interaction) reproduces the many-body eigenvalue, and check the non-interacting limit against non_interacting.solve.
"""

import numpy as np
import pytest

import iDEA


def _system(spin, num_points=41, length=8.0, interacting=True):
    x = np.linspace(-length, length, num_points)
    v_ext = -2.0 / (abs(x) + 1.0)
    v_int = iDEA.interactions.softened_interaction(x)
    if not interacting:
        v_int = 0.0 * v_int
    return iDEA.system.System(x, v_ext, v_int, spin)


def _brute_force(s, operator, space):
    psi = space.reshape(-1)
    return np.vdot(psi, operator @ psi).real / np.vdot(psi, psi).real


@pytest.mark.parametrize("spin, num_points", [("ud", 41), ("uu", 41), ("uud", 21)])
def test_energy_decomposition(spin, num_points):
    s = _system(spin, num_points=num_points)
    state = iDEA.methods.interacting.solve(s, k=0)

    E_k = iDEA.observables.kinetic_energy(s, state=state)
    n = iDEA.observables.density(s, state=state)
    E_ext = iDEA.observables.external_energy(s, n, s.v_ext)
    E_int = iDEA.observables.interaction_energy(s, state=state)

    assert np.isclose(E_k + E_ext + E_int, state.energy, rtol=0.0, atol=1e-10)
    assert E_int > 0.0


def test_kinetic_energy_matches_many_body_operator():
    s = _system("ud")
    state = iDEA.methods.interacting.solve(s, k=0)
    E_k = iDEA.observables.kinetic_energy(s, state=state)
    K = iDEA.methods.interacting.kinetic_energy_operator(s)
    assert np.isclose(E_k, _brute_force(s, K, state.space), rtol=0.0, atol=1e-10)


def test_observable_return_spins_and_external_operator():
    s = _system("ud")
    state = iDEA.methods.interacting.solve(s, k=0)
    V = iDEA.methods.non_interacting.external_potential_operator(s)
    O, up_O, down_O = iDEA.observables.observable(s, V, state=state, return_spins=True)
    n = iDEA.observables.density(s, state=state)
    assert np.isclose(O, up_O + down_O)
    assert np.isclose(up_O, down_O)
    assert np.isclose(O, iDEA.observables.external_energy(s, n, s.v_ext))


def test_non_interacting_limit():
    s = _system("ud", interacting=False)
    state = iDEA.methods.interacting.solve(s, k=0)
    ni_state = iDEA.methods.non_interacting.solve(s, k=0)
    assert np.isclose(
        iDEA.observables.kinetic_energy(s, state=state),
        iDEA.observables.kinetic_energy(s, state=ni_state),
        rtol=0.0,
        atol=1e-8,
    )
    assert np.isclose(iDEA.observables.interaction_energy(s, state=state), 0.0)


def test_kinetic_energy_evolution():
    s = _system("ud", num_points=31)
    state = iDEA.methods.interacting.solve(s, k=0)
    t = np.linspace(0.0, 0.5, 3)
    v_ptrb = np.tile(-0.1 * s.x, (t.shape[0], 1))
    evolution = iDEA.methods.interacting.propagate(s, state, v_ptrb, t)
    E_k = iDEA.observables.kinetic_energy(s, evolution=evolution)
    K = iDEA.methods.interacting.kinetic_energy_operator(s)
    assert E_k.shape == t.shape
    assert np.isclose(E_k[0], iDEA.observables.kinetic_energy(s, state=state))
    for j in range(t.shape[0]):
        assert np.isclose(E_k[j], _brute_force(s, K, evolution.td_space[j, ...]), rtol=0.0, atol=1e-10)


def test_many_body_operator_raises():
    s = _system("ud", num_points=21)
    state = iDEA.methods.interacting.solve(s, k=0)
    K = iDEA.methods.interacting.kinetic_energy_operator(s)
    with pytest.raises(NotImplementedError, match="one-body"):
        iDEA.observables.observable(s, K, state=state)


def test_sparse_one_body_operator_single_electron():
    # For one electron the interacting kinetic operator is a sparse (N_x, N_x) matrix. It must be contracted
    # elementwise with the density matrix, not multiplied as a matrix product.
    s = _system("u")
    state = iDEA.methods.interacting.solve(s, k=0)
    K_sparse = iDEA.methods.interacting.kinetic_energy_operator(s)
    K_dense = iDEA.methods.non_interacting.kinetic_energy_operator(s)
    assert K_sparse.shape == K_dense.shape
    E_k_sparse = iDEA.observables.observable(s, K_sparse, state=state)
    E_k_dense = iDEA.observables.observable(s, K_dense, state=state)
    assert np.isclose(E_k_sparse, E_k_dense, rtol=0.0, atol=1e-12)
    assert np.isclose(E_k_dense, _brute_force(s, K_sparse, state.space), rtol=0.0, atol=1e-10)
    ni_state = iDEA.methods.non_interacting.solve(s, k=0)
    assert np.isclose(E_k_dense, iDEA.observables.kinetic_energy(s, state=ni_state), rtol=0.0, atol=1e-8)
