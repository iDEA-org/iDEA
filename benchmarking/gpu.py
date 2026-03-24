"""Benchmark GPU vs CPU scaling of the iDEA interacting solver.

Solves the QHO ground state for a range of grid sizes and records
wall-clock time and memory usage for both the CPU and GPU paths.

Usage (from repo root):
    python benchmarking/gpu.py

Output:
    benchmarking/gpu_scaling.png
"""

import gc
import os
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import psutil

import iDEA.interactions
import iDEA.methods.interacting
import iDEA.system


def make_qho(N: int) -> iDEA.system.System:
    x = np.linspace(-10, 10, N)
    v_ext = 0.5 * (0.25**2) * x**2
    v_int = iDEA.interactions.softened_interaction(x)
    return iDEA.system.System(x, v_ext, v_int, "uu")


GRID_SIZES = list(range(50, 451, 50))

cpu_times = []
cpu_mems = []
gpu_times = []
gpu_mems = []


proc = psutil.Process(os.getpid())


def _peak_rss_gb(fn):
    baseline = proc.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def _poll():
        while not stop.is_set():
            rss = proc.memory_info().rss
            if rss > peak[0]:
                peak[0] = rss
            time.sleep(0.05)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    result = fn()
    stop.set()
    t.join()
    return result, (peak[0] - baseline) / 1024**3


print("=== CPU benchmarks ===")
for N in GRID_SIZES:
    print(f"  N={N}  (Hamiltonian size {N**2}×{N**2})")
    s = make_qho(N)
    gc.collect()

    t0 = time.perf_counter()
    try:
        state, peak_gb = _peak_rss_gb(lambda: iDEA.methods.interacting.solve(s, k=0, GPU=False))
        t1 = time.perf_counter()
        cpu_times.append(t1 - t0)
        cpu_mems.append(peak_gb)
        del state
    except Exception as exc:
        t1 = time.perf_counter()
        print(f"    FAILED: {exc}")
        cpu_times.append(float("nan"))
        cpu_mems.append(float("nan"))

    gc.collect()


print("\n=== GPU benchmarks ===")
try:
    import cupy as cp

    _has_cupy = True
except ImportError:
    print("  cupy not available – skipping GPU benchmarks")
    _has_cupy = False

for N in GRID_SIZES:
    if not _has_cupy:
        gpu_times.append(float("nan"))
        gpu_mems.append(float("nan"))
        continue

    print(f"  N={N}  (Hamiltonian size {N**2}×{N**2})")
    s = make_qho(N)
    cp.get_default_memory_pool().free_all_blocks()

    t0 = time.perf_counter()
    try:
        state = iDEA.methods.interacting.solve(s, k=0, GPU=True)
        t1 = time.perf_counter()
        del state
        gc.collect()
        peak_vram_gb = cp.get_default_memory_pool().total_bytes() / 1024**3
        gpu_times.append(t1 - t0)
        gpu_mems.append(peak_vram_gb)
    except Exception as exc:
        t1 = time.perf_counter()
        print(f"    FAILED: {exc}")
        gpu_times.append(float("nan"))
        gpu_mems.append(float("nan"))

    cp.get_default_memory_pool().free_all_blocks()


COLOR_CPU = "steelblue"
COLOR_GPU = "green"

x_cpu = GRID_SIZES[: len(cpu_times)]
x_gpu = GRID_SIZES[: len(gpu_times)]

fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(12, 5))

ax_time.plot(x_cpu, cpu_times, "o-", color=COLOR_CPU, label="CPU (Intel i9)")
ax_time.plot(x_gpu, gpu_times, "o-", color=COLOR_GPU, label="GPU (RTX 4090)")
ax_time.set_title("Solve time")
ax_time.set_xlabel("Number of grid points")
ax_time.set_ylabel("Time (s)")
ax_time.legend()

ax_mem.plot(x_cpu, cpu_mems, "o-", color=COLOR_CPU, label="CPU RAM (Intel i9)")
ax_mem.plot(x_gpu, gpu_mems, "o-", color=COLOR_GPU, label="GPU VRAM (RTX 4090)")
ax_mem.set_title("Peak memory usage")
ax_mem.set_xlabel("Number of grid points")
ax_mem.set_ylabel("Memory (GB)")
ax_mem.legend()

fig.suptitle("iDEA interacting solver – scaling with grid size (QHO, 2 electrons, ground state)")
plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "gpu_scaling.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved to {out_path}")
plt.show()
