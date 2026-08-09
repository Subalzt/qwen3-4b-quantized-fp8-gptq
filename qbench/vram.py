"""Out-of-process VRAM sampler.

Why a separate process: torch.cuda.max_memory_allocated() only sees torch's own
caching allocator. It cannot see llama.cpp, it cannot see vLLM's preallocated
KV pool the way the driver sees it, and it cannot see CUDA context overhead.
The only number that is comparable across every runtime in the grid is what the
driver reports, sampled from outside the process under test.

Run standalone:
    python -m qbench.vram --out samples.jsonl --hz 20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass


def _nvml():
    try:
        import pynvml  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "pynvml is required for VRAM sampling: pip install nvidia-ml-py"
        ) from e
    return pynvml


def sample_loop(out_path: str, gpu_index: int, hz: float, stop_after: float) -> None:
    pynvml = _nvml()
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    period = 1.0 / hz
    t_end = time.time() + stop_after
    with open(out_path, "w", encoding="utf-8") as fh:
        while time.time() < t_end:
            t = time.time()
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle)
            except Exception:
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                except Exception:
                    procs = []
            rec = {
                "t": t,
                "used_mb": mem.used / 2**20,
                "total_mb": mem.total / 2**20,
                "procs": [
                    {
                        "pid": p.pid,
                        "mb": (p.usedGpuMemory or 0) / 2**20,
                    }
                    for p in procs
                ],
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            dt = period - (time.time() - t)
            if dt > 0:
                time.sleep(dt)
    pynvml.nvmlShutdown()


@dataclass
class VramReport:
    peak_mb: float
    baseline_mb: float
    delta_mb: float
    n_samples: int
    hz: float
    total_mb: float
    gpu_name: str = ""


class VramSampler:
    """Context manager. Spawns the sampler, collects a baseline, then yields.

    Usage:
        with VramSampler(hz=20, baseline_s=1.5) as s:
            model = load_the_thing()   # everything under test happens here
            run_the_benchmark()
        report = s.report()
    """

    def __init__(
        self,
        hz: float = 20.0,
        gpu_index: int = 0,
        baseline_s: float = 1.5,
        max_s: float = 60 * 60,
        out_path: str | None = None,
    ) -> None:
        self.hz = hz
        self.gpu_index = gpu_index
        self.baseline_s = baseline_s
        self.max_s = max_s
        self.out_path = out_path or os.path.join(
            tempfile.gettempdir(), f"qbench_vram_{os.getpid()}.jsonl"
        )
        self.proc: subprocess.Popen | None = None
        self._t_baseline_end = 0.0

    def __enter__(self) -> "VramSampler":
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "qbench.vram",
                "--out",
                self.out_path,
                "--hz",
                str(self.hz),
                "--gpu",
                str(self.gpu_index),
                "--max-seconds",
                str(self.max_s),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        # Let it collect an idle baseline BEFORE the model loads. The desktop
        # compositor and any other GPU app live in this number; we subtract it.
        time.sleep(self.baseline_s)
        self._t_baseline_end = time.time()
        return self

    def __exit__(self, *exc) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self.proc.kill()
        return None

    def report(self) -> VramReport:
        samples = []
        try:
            with open(self.out_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            samples.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass  # torn last line from termination
        except FileNotFoundError:
            pass
        if not samples:
            nan = float("nan")
            return VramReport(nan, nan, nan, 0, self.hz, nan)

        base = [s["used_mb"] for s in samples if s["t"] <= self._t_baseline_end]
        after = [s["used_mb"] for s in samples if s["t"] > self._t_baseline_end]
        baseline = min(base) if base else min(s["used_mb"] for s in samples)
        peak = max(after) if after else max(s["used_mb"] for s in samples)
        return VramReport(
            peak_mb=peak,
            baseline_mb=baseline,
            delta_mb=peak - baseline,
            n_samples=len(samples),
            hz=self.hz,
            total_mb=samples[0]["total_mb"],
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample GPU memory to jsonl.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--max-seconds", type=float, default=3600.0)
    a = ap.parse_args()
    sample_loop(a.out, a.gpu, a.hz, a.max_seconds)


if __name__ == "__main__":
    main()
