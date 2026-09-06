"""Benchmark GPU vs CPU scaling of the iDEA interacting ground-state solver.

Solves the QHO ground state for a range of grid sizes and records wall-clock
time and memory usage for both the CPU and GPU paths.

Prints the detected hardware and an estimated runtime before starting, so you
know what you are committing to. The estimate is calibrated against this
machine with one small solve.

Usage (from repo root):
    python benchmarking/gpu.py

Output:
    benchmarking/gpu_scaling.png
"""

import gc
import os
import time

import _bench
import matplotlib.pyplot as plt

import iDEA.methods.interacting

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


GRID_SIZES = list(range(50, 451, 50))

hw = _bench.Hardware()

print("Calibrating runtime estimate...", flush=True)
cpu_scale = _bench.calibrate(gpu=False)
gpu_scale = _bench.calibrate(gpu=True) if HAS_CUPY else None
estimate = _bench.estimate_ground_state(GRID_SIZES, cpu_scale, gpu_scale)

_bench.print_preamble(
    title="interacting solver, ground state",
    hw=hw,
    sweep=GRID_SIZES,
    detail="QHO, 2 electrons (uu), ground state",
    estimate_s=estimate,
)

cpu_times, cpu_mems = [], []
gpu_times, gpu_mems = [], []

for N in GRID_SIZES:
    print(f"  N={N}  (Hamiltonian size {N**2}x{N**2})", flush=True)
    s = _bench.make_qho(N)
    gc.collect()

    try:
        t0 = time.perf_counter()
        state, peak_gb = _bench.peak_rss_gb(lambda system=s: iDEA.methods.interacting.solve(system, k=0, GPU=False))
        t1 = time.perf_counter()
        cpu_times.append(t1 - t0)
        cpu_mems.append(peak_gb)
        del state
    except Exception as exc:
        print(f"    CPU FAILED: {exc}")
        cpu_times.append(float("nan"))
        cpu_mems.append(float("nan"))
    gc.collect()

    if not HAS_CUPY:
        gpu_times.append(float("nan"))
        gpu_mems.append(float("nan"))
        continue

    cp.get_default_memory_pool().free_all_blocks()
    try:
        t0 = time.perf_counter()
        state = iDEA.methods.interacting.solve(s, k=0, GPU=True)
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()
        gpu_times.append(t1 - t0)
        gpu_mems.append(cp.get_default_memory_pool().total_bytes() / 1024**3)
        del state
    except Exception as exc:
        print(f"    GPU FAILED: {exc}")
        gpu_times.append(float("nan"))
        gpu_mems.append(float("nan"))
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
    suptitle="iDEA interacting solver - scaling with grid size (QHO, 2 electrons, ground state)",
    time_title="Solve time",
    out_path=os.path.join(os.path.dirname(__file__), "gpu_scaling.png"),
)

plt.show()
