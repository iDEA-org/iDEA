"""
Regression tests for iDEA.methods.interacting.antisymmetrize.

Checks that after antisymmetrization (including the zero and duplicate filters) every returned
state.space[..., n] / state.spin[..., n] pair still antisymmetrizes to state.full[..., n] (issue #42).
"""

# Dependencies
import copy
import itertools
import string

import numpy as np
import pytest

import iDEA


def _antisymmetrize_single(s, space, spin):
    r"""
    Antisymmetrize a single space x spin product and normalise it.

    | Args:
    |     s: iDEA.system.System, System object.
    |     space: np.ndarray, Spatial part of the wavefunction.
    |     spin: np.ndarray, Spin part of the wavefunction.

    | Returns:
    |     full: np.ndarray, Normalised antisymmetrized wavefunction.
    """
    l = string.ascii_lowercase[: s.count]
    L = string.ascii_uppercase[: s.count]
    st = l + "," + L + "->" + "".join([i for sub in list(zip(l, L)) for i in sub])
    pairs = list(zip(range(0, s.count * 2, 2), range(1, s.count * 2, 2)))
    full_copy = copy.deepcopy(np.einsum(st, space, spin))
    full = np.zeros_like(full_copy)
    for p in itertools.permutations(range(s.count)):
        indices = list(itertools.chain(*[pairs[e] for e in p]))
        full += iDEA.methods.interacting._permutation_parity(p) * np.moveaxis(
            full_copy, list(range(s.count * 2)), indices
        )
    return full / np.sqrt(np.sum(abs(full) ** 2) * s.dx**s.count)


@pytest.fixture(scope="module")
def uud_system():
    x = np.linspace(-6, 6, 16)
    v_ext = 0.5 * (0.25**2) * x**2
    return iDEA.system.System(x, v_ext, iDEA.interactions.softened_interaction(x), "uud")


@pytest.fixture(scope="module")
def uud_states(uud_system):
    # level=30 makes the duplicate filter drop non-tail entries for this system.
    return iDEA.methods.interacting.solve(uud_system, k=-1, level=30)


def test_space_spin_aligned_with_full(uud_system, uud_states):
    r"""
    Tests that each stored space x spin antisymmetrizes to the stored full wavefunction.
    """
    tolerance = 1.0e-8
    assert uud_states.fulls.shape[-1] > 1
    assert uud_states.spaces.shape[-1] == uud_states.fulls.shape[-1]
    assert uud_states.spins.shape[-1] == uud_states.fulls.shape[-1]
    assert uud_states.energies.shape[0] == uud_states.fulls.shape[-1]
    for n in range(uud_states.fulls.shape[-1]):
        full = _antisymmetrize_single(uud_system, uud_states.spaces[..., n], uud_states.spins[..., n])
        overlap = abs(np.sum(full * uud_states.fulls[..., n].conj()) * uud_system.dx**uud_system.count)
        assert overlap == pytest.approx(1.0, abs=tolerance)


def test_duplicate_filter_selects_kept_indices(uud_system, uud_states):
    r"""
    Tests that antisymmetrize co-selects spaces, spins and energies when duplicates are dropped mid-array.
    """
    n = uud_states.fulls.shape[-1]
    spaces = np.concatenate([uud_states.spaces[..., :1], uud_states.spaces], axis=-1)
    spins = np.concatenate([uud_states.spins[..., :1], uud_states.spins], axis=-1)
    energies = np.concatenate([uud_states.energies[:1], uud_states.energies])
    fulls, spaces_out, spins_out, energies_out = iDEA.methods.interacting.antisymmetrize(
        uud_system, spaces, spins, energies
    )
    assert fulls.shape[-1] == n
    assert np.allclose(energies_out, uud_states.energies)
    assert np.allclose(spaces_out, uud_states.spaces)
    assert np.allclose(spins_out, uud_states.spins)
