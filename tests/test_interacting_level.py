"""
Tests for the choice of level (number of spatial eigenstates) in iDEA.methods.interacting.solve.

Regression tests for:
    #41: solve failed non-deterministically for closed-shell systems with 4+ electrons (_estimate_level too small).
    #50: _estimate_level overestimated the level needed, making solve needlessly slow.

All systems use small grids so the tests run quickly.
"""

# Dependencies
import numpy as np
import pytest

import iDEA


def make_system(electrons, num_points, length=10.0, omega=0.25, interacting=True, stencil=13):
    r"""
    Construct a harmonic system with a softened Coulomb interaction.

    | Args:
    |     electrons: string, Electrons contained in the system.
    |     num_points: int, Number of grid points.
    |     length: float, Length of the grid.
    |     omega: float, Harmonic confinement frequency.
    |     interacting: bool, If False the interaction is switched off.
    |     stencil: int, Finite difference stencil.

    | Returns:
    |     s: iDEA.system.System, System object.
    """
    x = np.linspace(-length / 2, length / 2, num_points)
    v_ext = 0.5 * omega**2 * x**2
    v_int = iDEA.interactions.softened_interaction(x) * (1.0 if interacting else 0.0)
    return iDEA.system.System(x, v_ext, v_int, electrons, stencil=stencil)


class TestClosedShell:
    r"""Regression tests for #41: closed-shell systems with 4 electrons."""

    def test_uudd_ground_state_is_deterministic(self):
        r"""Previously this failed with an IndexError on roughly half of all attempts."""
        s = make_system("uudd", 9)
        H = iDEA.methods.interacting.hamiltonian(s)
        for _ in range(10):
            assert np.isclose(iDEA.methods.interacting.solve(s, H=H, k=0).energy, 2.99622847)


class TestEstimateLevel:
    r"""Regression tests for #50: the level estimate should be tight."""

    @pytest.mark.parametrize(
        "electrons, num_points, expected",
        [
            ("ud", 41, [1, 2, 3]),
            ("uu", 41, [2, 5, 7]),
            ("uud", 15, [2, 5, 8]),
            ("udd", 15, [2, 5, 8]),
        ],
    )
    def test_estimate(self, electrons, num_points, expected):
        s = make_system(electrons, num_points)
        assert [iDEA.methods.interacting._estimate_level(s, k) for k in range(3)] == expected

    @pytest.mark.parametrize("electrons", ["ud", "uu", "uud", "udd", "uuu", "uudd"])
    def test_estimate_much_smaller_than_dimension(self, electrons):
        s = make_system(electrons, 9)
        assert 1 <= iDEA.methods.interacting._estimate_level(s, 0) < 9**s.count // 10

    @pytest.mark.parametrize("electrons", ["ud", "uu", "uud"])
    def test_estimate_sufficient_non_interacting(self, electrons, monkeypatch):
        r"""For a non-interacting system the estimate is exactly enough, so no retry is ever needed."""
        s = make_system(electrons, 15, interacting=False)
        levels = []
        solve_antisymmetric = iDEA.methods.interacting._solve_antisymmetric

        def record(s, H, level, GPU=False):
            levels.append(level)
            return solve_antisymmetric(s, H, level, GPU=GPU)

        monkeypatch.setattr(iDEA.methods.interacting, "_solve_antisymmetric", record)
        for k in range(3):
            levels.clear()
            iDEA.methods.interacting.solve(s, k=k)
            assert levels == [iDEA.methods.interacting._estimate_level(s, k)]


class TestSolveLevel:
    r"""Tests of how solve chooses, grows and respects the level."""

    @pytest.mark.parametrize(
        "electrons, num_points",
        [("ud", 21), ("uu", 21), ("uud", 11), ("udd", 11), ("uuu", 11)],
    )
    @pytest.mark.parametrize("k", [0, 1, 2])
    def test_energy_matches_large_level(self, electrons, num_points, k):
        s = make_system(electrons, num_points)
        H = iDEA.methods.interacting.hamiltonian(s)
        reference = iDEA.methods.interacting.solve(s, H=H, k=k, level=80).energy
        assert np.isclose(iDEA.methods.interacting.solve(s, H=H, k=k).energy, reference)

    def test_retries_when_estimate_too_small(self, monkeypatch, capsys):
        s = make_system("uu", 21)
        H = iDEA.methods.interacting.hamiltonian(s)
        reference = iDEA.methods.interacting.solve(s, H=H, k=2, level=40).energy
        monkeypatch.setattr(iDEA.methods.interacting, "_estimate_level", lambda s, k: 1)
        state = iDEA.methods.interacting.solve(s, H=H, k=2)
        assert np.isclose(state.energy, reference)
        assert "retrying with level=" in capsys.readouterr().out

    def test_explicit_level_too_small_raises(self):
        s = make_system("uu", 21)
        with pytest.raises(ValueError, match="increase level"):
            iDEA.methods.interacting.solve(s, k=0, level=1)

    def test_explicit_level_is_used(self, monkeypatch):
        s = make_system("uu", 21)
        levels = []
        solve_antisymmetric = iDEA.methods.interacting._solve_antisymmetric

        def record(s, H, level, GPU=False):
            levels.append(level)
            return solve_antisymmetric(s, H, level, GPU=GPU)

        monkeypatch.setattr(iDEA.methods.interacting, "_solve_antisymmetric", record)
        iDEA.methods.interacting.solve(s, k=0, level=7)
        assert levels == [7]

    def test_all_states_with_explicit_level(self):
        s = make_system("uu", 21)
        states = iDEA.methods.interacting.solve(s, k=-1, level=10)
        assert states.energies.shape[0] >= 1
        assert np.all(np.diff(states.energies) >= -1e-10)

    def test_all_states_requires_level(self):
        s = make_system("uu", 21)
        with pytest.raises(ValueError, match="level must be provided"):
            iDEA.methods.interacting.solve(s, k=-1)

    def test_k_beyond_grid_raises(self):
        r"""Two same-spin electrons on 4 grid points have only 6 antisymmetric states."""
        s = make_system("uu", 4, length=4.0, stencil=3)
        assert np.isclose(
            iDEA.methods.interacting.solve(s, k=5).energy, iDEA.methods.interacting.solve(s, k=-1, level=15).energies[5]
        )
        with pytest.raises(ValueError, match="cannot be found on this grid"):
            iDEA.methods.interacting.solve(s, k=6)


def test_antisymmetrize_no_surviving_states():
    r"""A symmetric spatial state of two same-spin electrons is annihilated: return empty arrays, not an IndexError."""
    s = make_system("uu", 11)
    energies, spaces = np.linalg.eigh(iDEA.methods.interacting.hamiltonian(s).toarray())
    space = spaces[:, :1].reshape((11, 11, 1))
    spin = np.zeros((2, 2, 1))
    spin[0, 0, 0] = 1.0
    fulls, spaces, spins, energies = iDEA.methods.interacting.antisymmetrize(s, space, spin, energies[:1])
    assert fulls.shape[-1] == 0
    assert spaces.shape[-1] == 0
    assert spins.shape[-1] == 0
    assert energies.shape == (0,)
