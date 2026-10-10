"""
Tests of single-particle excited states (k > 0).

Checks that each excitation k maps to a distinct occupation configuration, with degenerate configurations enumerated
properly (issue #22), and that excited states are solved self-consistently with their own occupation (Delta SCF,
issue #43).
"""

# Dependencies
import numpy as np
import pytest

import iDEA

# Mixing for the interacting methods: the default of 0.5 does not converge the LDA ground-state of the test atom.
MIXING = 0.2


def max_residual(s, method, state):
    r"""
    Largest residual |H phi - e phi| over the occupied orbitals, with H rebuilt from the state itself.

    | Args:
    |     s: iDEA.system.System, System object.
    |     method: module, iDEA method used to compute the state.
    |     state: iDEA.state.SingleBodyState, State.

    | Returns:
    |     residual: float, Largest residual.
    """
    n, up_n, down_n = iDEA.observables.density(s, state, return_spins=True)
    p, up_p, down_p = iDEA.observables.density_matrix(s, state, return_spins=True)
    kwargs = {"alpha": 0.8} if method is iDEA.methods.hybrid else {}
    H, up_H, down_H = method.hamiltonian(s, up_n, down_n, up_p, down_p, **kwargs)
    residual = 0.0
    for spin, spin_H in ((state.up, up_H), (state.down, down_H)):
        for i in spin.occupied:
            residual = max(residual, np.max(abs(spin_H @ spin.orbitals[:, i] - spin.energies[i] * spin.orbitals[:, i])))
    return residual


def configuration(state):
    r"""Occupied orbital indices of a state as a hashable tuple."""
    return (tuple(state.up.occupied.tolist()), tuple(state.down.occupied.tolist()))


@pytest.fixture(scope="module")
def particle_in_a_box():
    x = np.linspace(0, 5, 50)
    return iDEA.system.System(x, np.zeros(x.shape), np.zeros((x.shape[0], x.shape[0])), electrons="ud")


@pytest.fixture(scope="module")
def atom():
    x = np.linspace(-10, 10, 51)
    return iDEA.system.System(x, -2.0 / (abs(x) + 1.0), iDEA.interactions.softened_interaction(x), electrons="ud")


class TestNonInteractingOccupations:
    def test_distinct_configurations(self, particle_in_a_box):
        configurations = [
            configuration(iDEA.methods.non_interacting.solve(particle_in_a_box, k=k, silent=True)) for k in range(20)
        ]
        assert len(set(configurations)) == len(configurations)

    def test_degenerate_pair(self, particle_in_a_box):
        # Issue #22: states 6 and 7 are the degenerate spin-swapped pair ([1], [2]) and ([2], [1]).
        state_6 = iDEA.methods.non_interacting.solve(particle_in_a_box, k=6, silent=True)
        state_7 = iDEA.methods.non_interacting.solve(particle_in_a_box, k=7, silent=True)
        assert {configuration(state_6), configuration(state_7)} == {((1,), (2,)), ((2,), (1,))}
        assert iDEA.methods.non_interacting.total_energy(particle_in_a_box, state_6) == pytest.approx(
            iDEA.methods.non_interacting.total_energy(particle_in_a_box, state_7)
        )

    def test_energies_ordered(self, particle_in_a_box):
        energies = [
            iDEA.methods.non_interacting.total_energy(
                particle_in_a_box, iDEA.methods.non_interacting.solve(particle_in_a_box, k=k, silent=True)
            )
            for k in range(20)
        ]
        assert np.all(np.diff(energies) >= -1e-12)

    @pytest.mark.parametrize("electrons", ["uu", "uud", "uudd"])
    def test_distinct_configurations_many_electrons(self, electrons):
        x = np.linspace(0, 5, 30)
        s = iDEA.system.System(x, np.zeros(x.shape), np.zeros((x.shape[0], x.shape[0])), electrons=electrons)
        configurations = [configuration(iDEA.methods.non_interacting.solve(s, k=k, silent=True)) for k in range(12)]
        assert len(set(configurations)) == len(configurations)


class TestDeltaSCF:
    @pytest.mark.parametrize("method_name", ["hartree", "hartree_fock", "lda", "hybrid"])
    def test_ground_state_aufbau(self, atom, method_name):
        method = getattr(iDEA.methods, method_name)
        state = method.solve(atom, k=0, silent=True, mixing=MIXING)
        assert configuration(state) == ((0,), (0,))
        assert max_residual(atom, method, state) < 1e-8

    @pytest.mark.parametrize("method_name", ["hartree", "hartree_fock", "lda", "hybrid"])
    def test_first_excitation_self_consistent(self, atom, method_name):
        # Issue #43: the returned excited state must be a stationary point of its own Hamiltonian.
        method = getattr(iDEA.methods, method_name)
        ground_state = method.solve(atom, k=0, silent=True, mixing=MIXING)
        excited_state = method.solve(atom, k=1, silent=True, mixing=MIXING)
        assert configuration(excited_state) in {((0,), (1,)), ((1,), (0,))}
        assert max_residual(atom, method, excited_state) < 1e-8
        assert method.total_energy(atom, excited_state) > method.total_energy(atom, ground_state)

    def test_spin_swapped_pair(self, atom):
        state_1 = iDEA.methods.hartree_fock.solve(atom, k=1, silent=True, mixing=MIXING)
        state_2 = iDEA.methods.hartree_fock.solve(atom, k=2, silent=True, mixing=MIXING)
        assert {configuration(state_1), configuration(state_2)} == {((0,), (1,)), ((1,), (0,))}
        assert iDEA.methods.hartree_fock.total_energy(atom, state_1) == pytest.approx(
            iDEA.methods.hartree_fock.total_energy(atom, state_2), abs=1e-8
        )

    def test_ranking_total_order(self, atom):
        # Ranking by the frozen-orbital total energy keeps the ground-state first and enumerates every configuration once.
        state = iDEA.methods.hartree_fock.solve(atom, k=0, silent=True, mixing=MIXING)
        ranked = iDEA.methods.non_interacting._rank_configurations(
            atom, state, 3, iDEA.methods.hartree_fock.hamiltonian, iDEA.methods.hartree_fock.total_energy
        )
        assert ranked[0] == ((0,), (0,))
        assert len(set(ranked)) == len(ranked) == 9

    def test_collapse_warns(self):
        # The second same-spin Hartree-Fock orbital has no bound excited partner, so the k = 1 configuration of
        # two up electrons relaxes back to the ground-state. This must be reported, not returned silently.
        x = np.linspace(-10, 10, 51)
        s = iDEA.system.System(x, -2.0 / (abs(x) + 1.0), iDEA.interactions.softened_interaction(x), electrons="uu")
        with pytest.warns(UserWarning, match="reordered during self-consistency"):
            state = iDEA.methods.hartree_fock.solve(s, k=1, silent=True, mixing=MIXING)
        assert max_residual(s, iDEA.methods.hartree_fock, state) < 1e-8
