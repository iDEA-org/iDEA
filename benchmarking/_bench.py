"""Shared helpers for the iDEA CPU/GPU scaling benchmarks.

Used by gpu.py (ground state) and gpu_td.py (time dependent). Holds the system
builder, the memory probe, hardware detection, the runtime estimator that backs
the preamble each script prints before it starts work, and the plotting.
"""

import os
import platform
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import psutil

import iDEA.interactions
import iDEA.system

# Reference timings measured on an Intel i9 / RTX 4090. These are only used to
# estimate how long a run will take; each script calibrates them against the
# machine it is actually running on with a single small solve, so they do not
# need to be accurate on other hardware.
GS_CPU_REF = {50: 0.03, 100: 0.46, 150: 2.0, 200: 6.2, 250: 18.0, 300: 50.0, 350: 102.0, 400: 175.0, 450: 238.0}
GS_GPU_REF = {50: 0.3, 100: 0.5, 150: 1.0, 200: 2.0, 250: 3.5, 300: 5.0, 350: 8.0, 400: 12.0, 450: 17.0}

# Seconds per propagation timestep.
TD_CPU_STEP_REF = {50: 0.005, 100: 0.015, 150: 0.032, 200: 0.060, 250: 0.100, 300: 0.190}
# The GPU exponential is a scaled truncated Taylor series that does roughly
# 2.6x the sparse mat-vecs of scipy's algorithm and synchronises twice per
# term, so at these sizes it is expected to be slower than the CPU.
TD_GPU_STEP_REF = {N: 2.5 * v for N, v in TD_CPU_STEP_REF.items()}

CALIBRATION_N = 100

COLOR_CPU = "steelblue"
COLOR_GPU = "green"


def make_qho(N: int, spin: str = "uu") -> iDEA.system.System:
    """Build a softened-interaction quantum harmonic oscillator on an N-point grid."""
    x = np.linspace(-10, 10, N)
    v_ext = 0.5 * (0.25**2) * x**2
    v_int = iDEA.interactions.softened_interaction(x)
    return iDEA.system.System(x, v_ext, v_int, spin)


def make_perturbation(s: iDEA.system.System, t: np.ndarray, field: float = 0.1) -> np.ndarray:
    """Suddenly applied uniform electric field, constant for all time."""
    return np.tile(-field * s.x, (t.shape[0], 1))


def peak_rss_gb(fn):
    """Run fn, returning (result, peak resident memory above baseline in GB)."""
    proc = psutil.Process(os.getpid())
    baseline = proc.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def _poll():
        while not stop.is_set():
            rss = proc.memory_info().rss
            if rss > peak[0]:
                peak[0] = rss
            time.sleep(0.05)

    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()
    try:
        result = fn()
    finally:
        stop.set()
        thread.join()
    return result, (peak[0] - baseline) / 1024**3


def _cpu_model() -> str:
    """Human-readable CPU model, tidied of vendor decoration."""
    name = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    if not name:
        name = platform.processor() or platform.machine() or "unknown CPU"
    for junk in ("(R)", "(TM)", "(r)", "(tm)", "CPU"):
        name = name.replace(junk, "")
    return " ".join(name.split())


def _gpu_info():
    """(name, VRAM GB) for the active device, or (None, None) if cupy is unavailable."""
    try:
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        return props["name"].decode(), props["totalGlobalMem"] / 1024**3
    except Exception:
        return None, None


class Hardware:
    """Detected CPU and GPU description, used for the preamble and plot labels."""

    def __init__(self):
        self.cpu = _cpu_model()
        self.threads = psutil.cpu_count(logical=True)
        self.ram_gb = psutil.virtual_memory().total / 1024**3
        self.gpu, self.vram_gb = _gpu_info()

    @property
    def has_gpu(self) -> bool:
        return self.gpu is not None

    def cpu_label(self) -> str:
        """Short label for plot legends, e.g. 'Intel i9-13900K'."""
        parts = self.cpu.split()
        for p in parts:
            if "-" in p and any(c.isdigit() for c in p):
                vendor = "Intel" if "Intel" in self.cpu else ("AMD" if "AMD" in self.cpu else parts[0])
                return f"{vendor} {p}"
        return " ".join(parts[:3])

    def gpu_label(self) -> str:
        return self.gpu.replace("NVIDIA ", "") if self.gpu else "no GPU"


def _interp_ref(table: dict, N: int) -> float:
    """Log-log interpolate a reference timing table at grid size N."""
    xs = np.array(sorted(table))
    ys = np.array([table[x] for x in xs])
    return float(np.exp(np.interp(np.log(N), np.log(xs), np.log(ys))))


def calibrate(gpu: bool = False) -> float:
    """Time one small solve and return this machine's speed ratio to the reference.

    A ratio above 1.0 means this machine is slower than the i9/4090 the
    reference numbers were taken on, and the estimate is scaled up accordingly.
    """
    import iDEA.methods.interacting

    s = make_qho(CALIBRATION_N)
    t0 = time.perf_counter()
    try:
        iDEA.methods.interacting.solve(s, k=0, GPU=gpu)
    except Exception:
        return 1.0
    elapsed = time.perf_counter() - t0
    reference = GS_GPU_REF[CALIBRATION_N] if gpu else GS_CPU_REF[CALIBRATION_N]
    return elapsed / reference


def estimate_ground_state(sweep, cpu_scale: float, gpu_scale: float = None) -> float:
    """Estimated total seconds for a ground-state sweep."""
    total = sum(_interp_ref(GS_CPU_REF, N) for N in sweep) * cpu_scale
    if gpu_scale is not None:
        total += sum(_interp_ref(GS_GPU_REF, N) for N in sweep) * gpu_scale
    return total


def estimate_time_dependent(sweep, steps: int, cpu_scale: float, gpu_scale: float = None) -> float:
    """Estimated total seconds for a time-dependent sweep.

    Each grid size costs one ground-state solve (shared by both backends) plus
    the propagation itself on each backend.
    """
    total = sum(_interp_ref(GS_CPU_REF, N) for N in sweep) * cpu_scale
    total += sum(_interp_ref(TD_CPU_STEP_REF, N) for N in sweep) * steps * cpu_scale
    if gpu_scale is not None:
        total += sum(_interp_ref(TD_GPU_STEP_REF, N) for N in sweep) * steps * gpu_scale
    return total


def format_duration(seconds: float) -> str:
    if seconds < 10:
        return f"~{seconds:.1f} sec"
    if seconds < 90:
        return f"~{seconds:.0f} sec"
    if seconds < 3600:
        return f"~{seconds / 60:.0f} min"
    return f"~{seconds / 3600:.1f} hr"


def print_preamble(title: str, hw: Hardware, sweep, detail: str, estimate_s: float) -> None:
    """Print detected hardware, the planned sweep and the runtime estimate."""
    rule = "=" * 74
    print(rule)
    print(f"iDEA benchmark - {title}")
    print(rule)
    print(f"  CPU : {hw.cpu}  ({hw.threads} threads, {hw.ram_gb:.1f} GB RAM)")
    if hw.has_gpu:
        print(f"  GPU : {hw.gpu}  ({hw.vram_gb:.1f} GB VRAM)")
    else:
        print("  GPU : none detected (cupy not installed) - GPU points will be skipped")
    print()
    print(f"  Sweep    : N = {', '.join(str(N) for N in sweep)}  ({len(sweep)} points)")
    print(f"  Detail   : {detail}")
    print(f"  Largest  : {max(sweep) ** 2:,} x {max(sweep) ** 2:,} Hamiltonian")
    print()
    print(f"  Estimated runtime: {format_duration(estimate_s)} (calibrated on this machine)")
    print("  Press Ctrl-C to abort.")
    print(rule)
    print(flush=True)


def plot_scaling(sweep, cpu_times, gpu_times, cpu_mems, gpu_mems, hw, suptitle, time_title, out_path):
    """Two-panel time and memory scaling plot, labelled with the detected hardware."""
    cpu_label = hw.cpu_label()
    gpu_label = hw.gpu_label()

    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(12, 5))

    ax_time.plot(sweep, cpu_times, "o-", color=COLOR_CPU, label=f"CPU ({cpu_label})")
    ax_time.plot(sweep, gpu_times, "o-", color=COLOR_GPU, label=f"GPU ({gpu_label})")
    ax_time.set_title(time_title)
    ax_time.set_xlabel("Number of grid points")
    ax_time.set_ylabel("Time (s)")
    ax_time.legend()

    ax_mem.plot(sweep, cpu_mems, "o-", color=COLOR_CPU, label=f"CPU RAM ({cpu_label})")
    ax_mem.plot(sweep, gpu_mems, "o-", color=COLOR_GPU, label=f"GPU VRAM ({gpu_label})")
    ax_mem.set_title("Peak memory usage")
    ax_mem.set_xlabel("Number of grid points")
    ax_mem.set_ylabel("Memory (GB)")
    ax_mem.legend()

    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {out_path}")
    return fig


def summarise(sweep, cpu_times, gpu_times) -> None:
    """Print a small CPU/GPU comparison table once the sweep is done."""
    print()
    print(f"  {'N':>5} {'CPU (s)':>10} {'GPU (s)':>10} {'speedup':>9}")
    for N, c, g in zip(sweep, cpu_times, gpu_times):
        speedup = "-" if np.isnan(c) or np.isnan(g) or g == 0 else f"{c / g:.1f}x"
        cs = "failed" if np.isnan(c) else f"{c:.2f}"
        gs = "skipped" if np.isnan(g) else f"{g:.2f}"
        print(f"  {N:5d} {cs:>10} {gs:>10} {speedup:>9}")
    print(flush=True)
