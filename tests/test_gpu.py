"""
iDEA GPU/CPU agreement testing platform.

The only parts of iDEA that run on the GPU are the exact interacting solver
(iDEA.methods.interacting.solve) and the exact interacting time propagation
(iDEA.methods.interacting.propagate), together with the many-body Hamiltonian
they are built on (iDEA.methods.interacting.hamiltonian). These tests assert
that each of them produces the same answer on the GPU as it does on the CPU.

The approximate methods (non-interacting, Hartree, Hartree-Fock, LDA, hybrids)
have no GPU path and are therefore not covered here.

The whole module is skipped when cupy is not installed, so it is a no-op on
machines without an NVIDIA GPU (including CI). To run it:

    pip install -e ".[gpu]"
    pytest tests/test_gpu.py -v

Systems are kept deliberately small (a few thousand basis states) so the whole
module runs in seconds -- these tests check correctness, not performance. See
benchmarking/ for the scaling measurements.
"""

import numpy as np
import pytest

import iDEA.interactions
import iDEA.methods.interacting
import iDEA.observables
import iDEA.system

cp = pytest.importorskip("cupy", reason="cupy is required for the GPU tests: pip install -e '.[gpu]'")


# Tolerances.
#
# Ground state: the CPU and GPU paths call two different iterative eigensolvers
# (scipy's ARPACK vs cupy's Lanczos), so they converge to slightly different
# points. A loose tolerance here still catches every structural bug -- a wrong
# Hamiltonian, reshape, or antisymmetrisation gives an O(1) difference, not a
# 1e-6 one.
GS_RTOL = 1e-6
GS_ATOL = 1e-8

# Propagation: both paths start from the *same* state, so the only difference is
# the matrix exponential itself (scipy's expm_multiply vs the scaled truncated
# Taylor series in _expm_multiply_gpu). Those agree to ~1e-14 per step, so this
# can be held much tighter.
TD_RTOL = 1e-8
TD_ATOL = 1e-10


def make_system(N: int = 40, spin: str = "uu") -> iDEA.system.System:
    """Build a small softened-interaction QHO system on an N-point grid."""
    x = np.linspace(-10, 10, N)
    v_ext = 0.5 * (0.25**2) * x**2
    v_int = iDEA.interactions.softened_interaction(x)
    return iDEA.system.System(x, v_ext, v_int, electrons=spin)


def norms(s: iDEA.system.System, td_space: np.ndarray) -> np.ndarray:
    """Norm of the many-body wavefunction at each time index."""
    return np.array([np.sum(abs(td_space[j, ...]) ** 2) * s.dx**s.count for j in range(td_space.shape[0])])


class TestHamiltonian:
    """The many-body Hamiltonian must be identical on both backends."""

    @pytest.mark.parametrize("spin", ["uu", "ud"])
    def test_two_electron(self, spin):
        s = make_system(N=40, spin=spin)
        H_cpu = iDEA.methods.interacting.hamiltonian(s, GPU=False)
        H_gpu = iDEA.methods.interacting.hamiltonian(s, GPU=True)
        assert np.allclose(H_cpu.toarray(), cp.asnumpy(H_gpu.toarray()), rtol=1e-12, atol=1e-14)

    def test_three_electron(self):
        """Three electrons exercises a deeper kron chain and the two-body interaction sum."""
        s = make_system(N=12, spin="uud")
        H_cpu = iDEA.methods.interacting.hamiltonian(s, GPU=False)
        H_gpu = iDEA.methods.interacting.hamiltonian(s, GPU=True)
        assert np.allclose(H_cpu.toarray(), cp.asnumpy(H_gpu.toarray()), rtol=1e-12, atol=1e-14)


class TestGroundState:
    """iDEA.methods.interacting.solve must agree on CPU and GPU."""

    @pytest.mark.parametrize("spin", ["uu", "ud"])
    def test_energy(self, spin):
        s = make_system(N=40, spin=spin)
        cpu = iDEA.methods.interacting.solve(s, k=0, GPU=False)
        gpu = iDEA.methods.interacting.solve(s, k=0, GPU=True)
        assert np.isclose(cpu.energy, gpu.energy, rtol=GS_RTOL, atol=GS_ATOL)

    @pytest.mark.parametrize("spin", ["uu", "ud"])
    def test_density(self, spin):
        """Density is gauge invariant, so it is immune to the arbitrary eigenvector sign."""
        s = make_system(N=40, spin=spin)
        cpu = iDEA.methods.interacting.solve(s, k=0, GPU=False)
        gpu = iDEA.methods.interacting.solve(s, k=0, GPU=True)
        n_cpu = iDEA.observables.density(s, state=cpu)
        n_gpu = iDEA.observables.density(s, state=gpu)
        assert np.allclose(n_cpu, n_gpu, rtol=GS_RTOL, atol=GS_ATOL)

    @pytest.mark.parametrize("spin", ["uu", "ud"])
    def test_wavefunction(self, spin):
        """Compared as magnitudes: eigensolvers may return opposite overall signs."""
        s = make_system(N=40, spin=spin)
        cpu = iDEA.methods.interacting.solve(s, k=0, GPU=False)
        gpu = iDEA.methods.interacting.solve(s, k=0, GPU=True)
        assert np.allclose(abs(cpu.space), abs(gpu.space), rtol=GS_RTOL, atol=GS_ATOL)

    def test_excited_state_energy(self):
        """k=1 covers the _estimate_level path.

        Only the energy is compared: excited states of this system can be
        degenerate, and degenerate eigenvectors are free to mix differently
        between the two solvers, so their densities need not match.
        """
        s = make_system(N=40, spin="ud")
        cpu = iDEA.methods.interacting.solve(s, k=1, GPU=False)
        gpu = iDEA.methods.interacting.solve(s, k=1, GPU=True)
        assert np.isclose(cpu.energy, gpu.energy, rtol=GS_RTOL, atol=GS_ATOL)


class TestPropagate:
    """iDEA.methods.interacting.propagate must agree on CPU and GPU.

    Every test here solves the ground state once on the CPU and hands that same
    state to both propagations. This isolates the propagator from the
    eigensolver, so a failure points at the GPU matrix exponential rather than
    at eigenvector sign or degeneracy differences.
    """

    N = 40
    SPIN = "uu"
    # dt = 0.1 gives dt * ||H||_inf ~ 3.3, so _expm_multiply_gpu splits each
    # step into 4 substeps. Anything that makes this 1 would leave the scaling
    # loop untested -- test_substepping_is_exercised guards against that.
    T = np.linspace(0, 1.0, 11)

    @pytest.fixture(scope="class")
    def setup(self):
        s = make_system(N=self.N, spin=self.SPIN)
        state = iDEA.methods.interacting.solve(s, k=0, GPU=False)
        v_ptrb = np.tile(-0.1 * s.x, (self.T.shape[0], 1))
        return s, state, v_ptrb

    @pytest.fixture(scope="class")
    def evolutions(self, setup):
        s, state, v_ptrb = setup
        cpu = iDEA.methods.interacting.propagate(s, state, v_ptrb, self.T, GPU=False)
        gpu = iDEA.methods.interacting.propagate(s, state, v_ptrb, self.T, GPU=True)
        return s, cpu, gpu

    def test_substepping_is_exercised(self, setup):
        """Guard that these parameters actually drive the scaling loop.

        _expm_multiply_gpu splits the step into m substeps so that ||A/m||_inf
        <= 1. If m were 1 the loop would be a single pass and the scaling logic
        would go untested, so assert the chosen dt genuinely requires m > 1.
        """
        s, _state, _v_ptrb = setup
        H = iDEA.methods.interacting.hamiltonian(s, GPU=False)
        dt = self.T[1] - self.T[0]
        inf_norm = float(abs(H).sum(axis=1).max()) * dt
        assert int(inf_norm) + 1 > 1, f"dt too small to exercise substepping (||A||_inf = {inf_norm})"

    def test_wavefunction_all_timesteps(self, evolutions):
        """The whole complex wavefunction must match at every time index."""
        _s, cpu, gpu = evolutions
        assert cpu.td_space.shape == gpu.td_space.shape
        assert np.allclose(cpu.td_space, gpu.td_space, rtol=TD_RTOL, atol=TD_ATOL)

    def test_density_evolution(self, evolutions):
        s, cpu, gpu = evolutions
        n_cpu = iDEA.observables.density(s, evolution=cpu)
        n_gpu = iDEA.observables.density(s, evolution=gpu)
        assert np.allclose(n_cpu, n_gpu, rtol=TD_RTOL, atol=TD_ATOL)

    def test_gpu_conserves_norm(self, evolutions):
        """Independent of the CPU path: the GPU propagation must stay unitary."""
        s, _cpu, gpu = evolutions
        assert np.allclose(norms(s, gpu.td_space), 1.0, rtol=1e-9, atol=1e-9)

    def test_accepts_cpu_hamiltonian(self, setup):
        """propagate(GPU=True) must accept a scipy Hamiltonian and move it to the device."""
        s, state, v_ptrb = setup
        H = iDEA.methods.interacting.hamiltonian(s, GPU=False)
        gpu = iDEA.methods.interacting.propagate(s, state, v_ptrb, self.T, H=H, GPU=True)
        cpu = iDEA.methods.interacting.propagate(s, state, v_ptrb, self.T, H=H, GPU=False)
        assert np.allclose(cpu.td_space, gpu.td_space, rtol=TD_RTOL, atol=TD_ATOL)
