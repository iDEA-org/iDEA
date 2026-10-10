"""
Tests for the many-body spin observables iDEA.observables.total_spin_z and iDEA.observables.total_spin_squared.

Small Hooke's-atom-like systems (harmonic external potential, softened Coulomb interaction) are solved with
iDEA.methods.interacting.solve and the spin character of each state is checked against the known result:
for "ud" the states alternate singlet / triplet, for "uu" every state is a triplet with S_z = 1, and for
"uud" the states have S_z = 1/2.
"""

import numpy as np
import pytest

import iDEA


def _system(electrons, num_points):
    x = np.linspace(-6, 6, num_points)
    omega = 0.5
    v_ext = 0.5 * omega**2 * x**2
    v_int = iDEA.interactions.softened_interaction(x)
    return iDEA.system.System(x, v_ext, v_int, electrons=electrons)


def _state(states, k):
    return iDEA.state.ManyBodyState(full=states.fulls[..., k], energy=states.energies[k])


@pytest.fixture(scope="module")
def ud():
    s = _system("ud", 25)
    return s, iDEA.methods.interacting.solve(s, k=-1, level=10)


@pytest.fixture(scope="module")
def uu():
    s = _system("uu", 25)
    return s, iDEA.methods.interacting.solve(s, k=-1, level=10)


def test_ud_singlet_triplet_interleaving(ud):
    s, states = ud
    expected_S2 = [0.0, 2.0, 0.0, 2.0]
    for k, S2 in enumerate(expected_S2):
        state = _state(states, k)
        assert iDEA.observables.total_spin_z(s, state) == pytest.approx(0.0, abs=1e-8)
        assert iDEA.observables.total_spin_squared(s, state) == pytest.approx(S2, abs=1e-6)


def test_ud_solve_k(ud):
    s, _ = ud
    ground = iDEA.methods.interacting.solve(s, k=0)
    first = iDEA.methods.interacting.solve(s, k=1)
    assert iDEA.observables.total_spin_squared(s, ground) == pytest.approx(0.0, abs=1e-6)
    assert iDEA.observables.total_spin_squared(s, first) == pytest.approx(2.0, abs=1e-6)
    # The spin attribute is the seed product state and does not distinguish the two.
    assert np.array_equal(ground.spin, first.spin)


def test_uu_triplet(uu):
    s, states = uu
    for k in range(states.fulls.shape[-1]):
        state = _state(states, k)
        assert iDEA.observables.total_spin_z(s, state) == pytest.approx(1.0, abs=1e-8)
        assert iDEA.observables.total_spin_squared(s, state) == pytest.approx(2.0, abs=1e-6)


def test_uud_spin_z():
    s = _system("uud", 12)
    state = iDEA.methods.interacting.solve(s, k=0)
    assert iDEA.observables.total_spin_z(s, state) == pytest.approx(0.5, abs=1e-8)
    S2 = iDEA.observables.total_spin_squared(s, state)
    # Three spin-1/2 particles: S = 1/2 (S^2 = 3/4) or S = 3/2 (S^2 = 15/4).
    assert 0.75 - 1e-6 <= S2 <= 3.75 + 1e-6


@pytest.mark.parametrize(
    "spinor, S_z, S2",
    [
        (np.array([[0.0, 1.0], [-1.0, 0.0]]) / np.sqrt(2), 0.0, 0.0),  # singlet
        (np.array([[0.0, 1.0], [1.0, 0.0]]) / np.sqrt(2), 0.0, 2.0),  # triplet m = 0
        (np.array([[1.0, 0.0], [0.0, 0.0]]), 1.0, 2.0),  # triplet m = +1
        (np.array([[0.0, 0.0], [0.0, 1.0]]), -1.0, 2.0),  # triplet m = -1
        (np.array([[0.0, 1.0], [0.0, 0.0]]), 0.0, 1.0),  # unsymmetrised |ud>: equal singlet/triplet mix
    ],
)
def test_two_spin_reference_states(spinor, S_z, S2):
    s = _system("ud", 6)
    phi = np.exp(-(s.x**2))
    space = np.einsum("a,b->ab", phi, phi)
    full = np.einsum("ab,AB->aAbB", space, spinor)
    full = full / np.sqrt(np.sum(abs(full) ** 2) * s.dx**2)
    state = iDEA.state.ManyBodyState(full=full)
    assert iDEA.observables.total_spin_z(s, state) == pytest.approx(S_z, abs=1e-12)
    assert iDEA.observables.total_spin_squared(s, state) == pytest.approx(S2, abs=1e-12)


def test_wrong_shape_raises():
    s = _system("ud", 6)
    state = iDEA.state.ManyBodyState(full=np.zeros((6, 2, 6)))
    with pytest.raises(ValueError):
        iDEA.observables.total_spin_z(s, state)
