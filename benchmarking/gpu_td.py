"""Benchmark GPU vs CPU scaling of the iDEA interacting time-dependent solver.

For a range of grid sizes, solves the QHO ground state and then propagates it
under a suddenly applied uniform electric field, recording wall-clock time and
memory usage of the propagation for both the CPU and GPU paths.

The ground state is solved once per grid size and handed to both backends, so
only the propagation is timed -- the eigensolver is setup here, not the
measurement.

Prints the detected hardware and an estimated runtime before starting. The
estimate is calibrated against this machine with one small solve.

Note on what this measures: cupy provides no sparse expm_multiply, so the GPU
path uses the scaled truncated Taylor series in
iDEA.methods.interacting._expm_multiply_gpu. That does roughly 2.6x the sparse
mat-vecs of the Al-Mohy-Higham scheme scipy uses on the CPU, and synchronises
with the host twice per Taylor term, so this is a comparison of two different
algorithms rather than of the two pieces of hardware alone. At these grid
sizes the GPU is expected to lose.

Usage (from repo root):
    python benchmarking/gpu_td.py

Output:
    benchmarking/gpu_td_scaling.png
"""

import gc
import os
import time

import _bench
import matplotlib.pyplot as plt
import numpy as np

import iDEA.methods.interacting

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


# Capped at 300: extending to 450 would spend roughly ten minutes in the
# ground-state eigensolver before a single propagation step is measured.
GRID_SIZES = list(range(50, 301, 50))
TIME_LENGTH = 0.2
TIME_STEPS = 21

t = np.linspace(0, TIME_LENGTH, TIME_STEPS)
steps = TIME_STEPS - 1

hw = _bench.Hardware()

print("Calibrating runtime estimate...", flush=True)
cpu_scale = _bench.calibrate(gpu=False)
gpu_scale = _bench.calibrate(gpu=True) if HAS_CUPY else None
estimate = _bench.estimate_time_dependent(GRID_SIZES, steps, cpu_scale, gpu_scale)

_bench.print_preamble(
    title="interacting solver, time-dependent propagation",
    hw=hw,
    sweep=GRID_SIZES,
    detail=f"QHO, 2 electrons (uu), {steps} timesteps of dt={t[1] - t[0]:.3g}, sudden uniform field",
    estimate_s=estimate,
)

cpu_times, cpu_mems = [], []
gpu_times, gpu_mems = [], []

for N in GRID_SIZES:
    print(f"  N={N}  (Hamiltonian size {N**2}x{N**2}, {steps} timesteps)", flush=True)
    s = _bench.make_qho(N)
    v_ptrb = _bench.make_perturbation(s, t)
    gc.collect()

    try:
        state = iDEA.methods.interacting.solve(s, k=0, GPU=False)
    except Exception as exc:
        print(f"    ground state FAILED: {exc}")
        cpu_times.append(float("nan"))
        cpu_mems.append(float("nan"))
        gpu_times.append(float("nan"))
        gpu_mems.append(float("nan"))
        continue
    gc.collect()

    try:
        t0 = time.perf_counter()
        evolution, peak_gb = _bench.peak_rss_gb(
            lambda system=s, st=state, vp=v_ptrb: iDEA.methods.interacting.propagate(system, st, vp, t, GPU=False)
        )
        t1 = time.perf_counter()
        cpu_times.append(t1 - t0)
        cpu_mems.append(peak_gb)
        del evolution
    except Exception as exc:
        print(f"    CPU FAILED: {exc}")
        cpu_times.append(float("nan"))
        cpu_mems.append(float("nan"))
    gc.collect()

    if not HAS_CUPY:
        gpu_times.append(float("nan"))
        gpu_mems.append(float("nan"))
        del state
        continue

    cp.get_default_memory_pool().free_all_blocks()
    try:
        t0 = time.perf_counter()
        evolution = iDEA.methods.interacting.propagate(s, state, v_ptrb, t, GPU=True)
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()
        gpu_times.append(t1 - t0)
        gpu_mems.append(cp.get_default_memory_pool().total_bytes() / 1024**3)
        del evolution
    except Exception as exc:
        print(f"    GPU FAILED: {exc}")
        gpu_times.append(float("nan"))
        gpu_mems.append(float("nan"))
    del state
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


_bench.summarise(GRID_SIZES, cpu_times, gpu_times)

_bench.plot_scaling(
    sweep=GRID_SIZES,
    cpu_times=cpu_times,
    gpu_times=gpu_times,
    cpu_mems=cpu_mems,
    gpu_mems=gpu_mems,
    hw=hw,
    suptitle=f"iDEA interacting time-dependent solver - scaling with grid size (QHO, 2 electrons, {steps} timesteps)",
    time_title="Propagation time",
    out_path=os.path.join(os.path.dirname(__file__), "gpu_td_scaling.png"),
)

plt.show()
