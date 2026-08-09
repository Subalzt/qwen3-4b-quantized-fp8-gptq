"""Backend adapters: each turns a (model, runtime) into the same three metrics.

The whole point of the harness is that a row from llama.cpp and a row from vLLM
are directly comparable. Rather than reimplement inference, each adapter drives
the proven, already-validated tool for that runtime and parses its output into a
qbench.schema.Row. VRAM is always sampled out-of-process (qbench.vram) because
no in-process counter can see llama.cpp's or vLLM's allocations.

Adapters:
  LlamaCppBackend  -> llama-bench (speed) + llama-perplexity (quality)
  VllmBackend      -> offline LLM engine (speed via batched generate)
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass

from .vram import VramSampler


@dataclass
class SpeedResult:
    prefill_tps: float = float("nan")
    decode_tps: float = float("nan")
    vram_delta_mb: float = float("nan")
    vram_peak_mb: float = float("nan")


@dataclass
class QualityResult:
    ppl: float = float("nan")
    ppl_n_tokens: int = 0


# ---------------------------------------------------------------------------
# llama.cpp
# ---------------------------------------------------------------------------
class LlamaCppBackend:
    def __init__(self, llama_dir: str | None = None):
        self.dir = os.path.expanduser(
            llama_dir or os.environ.get("LLAMA_CPP_DIR", "~/llama.cpp"))
        self.bench = os.path.join(self.dir, "build", "bin", "llama-bench")
        self.ppl = os.path.join(self.dir, "build", "bin", "llama-perplexity")

    def speed(self, model: str, ngl: int = 99, p: int = 512, n: int = 128,
              reps: int = 3) -> SpeedResult:
        # VRAM sampled around a short generation in a separate process.
        with VramSampler(hz=20, baseline_s=1.0) as vs:
            out = subprocess.run(
                [self.bench, "-m", model, "-ngl", str(ngl),
                 "-p", str(p), "-n", str(n), "-r", str(reps)],
                capture_output=True, text=True, timeout=1200,
            ).stdout
        vr = vs.report()
        pre = dec = float("nan")
        for line in out.splitlines():
            if f"pp{p}" in line:
                pre = _tps(line)
            elif f"tg{n}" in line:
                dec = _tps(line)
        return SpeedResult(pre, dec, vr.delta_mb, vr.peak_mb)

    def perplexity(self, model: str, corpus: str, ctx: int = 2048,
                   ngl: int = 99) -> QualityResult:
        out = subprocess.run(
            [self.ppl, "-m", model, "-f", os.path.expanduser(corpus),
             "-c", str(ctx), "-ngl", str(ngl), "-b", str(ctx)],
            capture_output=True, text=True, timeout=7200,
        )
        text = out.stdout + out.stderr
        m = re.search(r"PPL = ([0-9.]+)", text)
        n = 0
        mc = re.search(r"over (\d+) chunks", text)
        if mc:
            n = int(mc.group(1)) * ctx
        return QualityResult(float(m.group(1)) if m else float("nan"), n)


# ---------------------------------------------------------------------------
# vLLM (offline engine; speed only -- quality uses the transformers path)
# ---------------------------------------------------------------------------
class VllmBackend:
    """Runs in the vLLM venv via a subprocess helper (import isolation)."""

    def __init__(self, python: str | None = None):
        self.python = os.path.expanduser(
            python or "~/quant-bench/.venv-vllm/bin/python")

    def speed(self, model: str, prompt_tokens: int = 512,
              max_tokens: int = 128, n_seqs: int = 1) -> SpeedResult:
        helper = os.path.join(os.path.dirname(__file__), "_vllm_speed.py")
        env = dict(os.environ, VLLM_WSL2_ENABLE_PIN_MEMORY="1",
                   VLLM_LOGGING_LEVEL="WARNING")
        with VramSampler(hz=10, baseline_s=1.0) as vs:
            out = subprocess.run(
                [self.python, helper, model, str(prompt_tokens),
                 str(max_tokens), str(n_seqs)],
                capture_output=True, text=True, timeout=1800, env=env,
            ).stdout
        vr = vs.report()
        pre = dec = float("nan")
        for line in out.splitlines():
            if line.startswith("PREFILL_TPS"):
                pre = float(line.split()[1])
            elif line.startswith("DECODE_TPS"):
                dec = float(line.split()[1])
        return SpeedResult(pre, dec, vr.delta_mb, vr.peak_mb)


def _tps(line: str) -> float:
    # llama-bench rows look like:
    #   | qwen3 4B Q4_K - Medium | 2.32 GiB | ... | pp512 | 5660.50 ± 420.62 |
    # The throughput is the value immediately before the '±' -- NOT the first
    # float on the line (that is the model size in GiB).
    m = re.search(r"([0-9]+\.[0-9]+)\s*(?:±|\+/-)", line)
    if m:
        return float(m.group(1))
    # Fallback: last float on the line.
    fs = re.findall(r"([0-9]+\.[0-9]+)", line)
    return float(fs[-1]) if fs else float("nan")
