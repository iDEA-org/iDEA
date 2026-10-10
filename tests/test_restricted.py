"""
Tests that `restricted` has no effect for methods whose Hamiltonian is spin-independent.

non_interacting, hartree and lda return the same Hamiltonian for both spin channels, so a
restricted and an unrestricted solve must give identical results (see issue #44).
"""

import numpy as np
import pytest

import iDEA


@pytest.mark.parametrize(
    "method",
    [iDEA.methods.non_interacting, iDEA.methods.hartree, iDEA.methods.lda],
    ids=lambda m: m.name,
)
def test_restricted_is_noop_for_spin_independent_methods(method):
    x = np.linspace(-10, 10, 101)
    v_ext = -2.0 / (abs(x) + 1.0)
    s = iDEA.system.System(x, v_ext, iDEA.interactions.softened_interaction(x), "ud")

    unrestricted = method.solve(s, restricted=False, tol=1e-6, silent=True)
    restricted = method.solve(s, restricted=True, tol=1e-6, silent=True)

    assert np.array_equal(
        iDEA.observables.density(s, unrestricted),
        iDEA.observables.density(s, restricted),
    )
    assert method.total_energy(s, unrestricted) == method.total_energy(s, restricted)
