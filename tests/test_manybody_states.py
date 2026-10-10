"""
Tests for indexing iDEA.state.ManyBodyStates into individual iDEA.state.ManyBodyState objects.

Uses a small two-electron harmonic system so that observables of the n-th state of a collection
(solve with k=-1) can be compared with the state obtained directly with solve(s, k=n).
"""

# Dependencies
import numpy as np
import pytest

import iDEA

LEVEL = 12


@pytest.fixture(scope="module")
def system():
    x = np.linspace(-6, 6, 41)
    v_ext = 0.5 * 0.25**2 * x**2
    v_int = iDEA.interactions.softened_interaction(x)
    return iDEA.system.System(x, v_ext, v_int, electrons="ud")


@pytest.fixture(scope="module")
def states(system):
    return iDEA.methods.interacting.solve(system, k=-1, level=LEVEL)


def test_len(states):
    assert len(states) == states.energies.shape[0] == states.fulls.shape[-1]
    assert len(iDEA.state.ManyBodyStates()) == 0


def test_getitem_returns_manybodystate(states):
    for n in range(len(states)):
        state = states[n]
        assert isinstance(state, iDEA.state.ManyBodyState)
        assert state.energy == states.energies[n]
        assert np.array_equal(state.space, states.spaces[..., n])
        assert np.array_equal(state.spin, states.spins[..., n])
        assert np.array_equal(state.full, states.fulls[..., n])


def test_getitem_negative_and_numpy_index(states):
    assert states[-1].energy == states.energies[-1]
    assert states[np.int64(1)].energy == states.energies[1]


def test_getitem_out_of_range(states):
    with pytest.raises(IndexError):
        states[len(states)]
    with pytest.raises(IndexError):
        states[-len(states) - 1]
    with pytest.raises(TypeError):
        states[1.0]


def test_getitem_returns_copy(states):
    state = states[0]
    state.full[...] = 0.0
    assert not np.allclose(states.fulls[..., 0], 0.0)


def test_iteration(states):
    energies = [state.energy for state in states]
    assert np.array_equal(energies, states.energies)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_observables_match_single_solve(system, states, n):
    reference = iDEA.methods.interacting.solve(system, k=n)
    state = states[n]
    assert state.energy == pytest.approx(reference.energy, abs=1e-8)
    assert iDEA.observables.density(system, state=state) == pytest.approx(
        iDEA.observables.density(system, state=reference), abs=1e-8
    )
    assert iDEA.observables.density_matrix(system, state=state) == pytest.approx(
        iDEA.observables.density_matrix(system, state=reference), abs=1e-8
    )
    assert iDEA.methods.interacting.total_energy(system, state) == pytest.approx(
        iDEA.methods.interacting.total_energy(system, reference), abs=1e-8
    )
