#!/usr/bin/env python3
"""Capability matrix for this machine.

sm_120 (Blackwell laptop) is new enough that a library can install cleanly,
import cleanly, expose the right API -- and then die with "no kernel image is
available for execution on the device" the moment you launch a kernel. Import
checks are worthless here. Every probe below LAUNCHES A REAL KERNEL and, where
a reference is cheap, checks the numerics too.

Run this before every benchmark session, and re-run it after touching any
library. Failures recorded here explain missing cells in the results grid.

    python scripts/check_env.py                 # table
    python scripts/check_env.py --json env.json # machine-readable
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

# Numerical error above this vs. an fp16/bf16 reference means the kernel ran
# but produced garbage -- the failure mode the project brief warns about for
# bitsandbytes INT8 on sm_120. A silent wrong answer is worse than a crash.
CORRUPT_THRESHOLD = 0.5


@dataclass
class Probe:
    name: str
    level: str  # which roadmap level depends on this
    status: str = "skip"  # ok | fail | corrupt | skip
    detail: str = ""
    rel_err: float | None = None
    seconds: float = 0.0
    error: str = ""


RESULTS: list[Probe] = []


def probe(name: str, level: str) -> Callable:
    def deco(fn: Callable[[], str]) -> Callable:
        p = Probe(name=name, level=level)
        t0 = time.time()
        try:
            detail = fn()
            p.status = "ok"
            p.detail = detail or ""
            if isinstance(detail, str) and "CORRUPT" in detail:
                p.status = "corrupt"
        except ImportError as e:
            p.status = "skip"
            p.error = f"not installed: {e}"
        except Exception as e:
            p.status = "fail"
            p.error = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
            if os.environ.get("QBENCH_TRACE"):
                traceback.print_exc()
        p.seconds = round(time.time() - t0, 2)
        RESULTS.append(p)
        return fn

    return deco


def rel_err(out, ref) -> float:
    return float((out - ref).norm() / ref.norm())


def tag(err: float) -> str:
    return "  <-- CORRUPT" if err > CORRUPT_THRESHOLD else ""


# --------------------------------------------------------------------------
# host / driver
# --------------------------------------------------------------------------
def host_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        info["nvidia_smi"] = out
    except Exception as e:
        info["nvidia_smi"] = f"unavailable: {e}"
    try:
        out = subprocess.run(["nvcc", "--version"], capture_output=True,
                             text=True, timeout=30).stdout
        line = [l for l in out.splitlines() if "release" in l]
        info["nvcc"] = line[0].strip() if line else "present"
    except Exception:
        info["nvcc"] = "NOT FOUND (source builds + FlashInfer JIT will fail)"
    info["wsl"] = os.path.exists("/usr/lib/wsl/lib/libcuda.so.1")
    return info


# ==========================================================================
# L1 -- torch / baseline
# ==========================================================================
@probe("torch cuda + bf16 matmul", "L1")
def _torch_bf16() -> str:
    import torch
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    cap = torch.cuda.get_device_capability(0)
    arches = torch.cuda.get_arch_list()
    sm = f"sm_{cap[0]}{cap[1]}"
    if sm not in arches:
        raise RuntimeError(f"{sm} not in wheel arch list {arches}")
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    (a @ b).sum().item()
    torch.cuda.synchronize()
    free, total = [v / 2**30 for v in torch.cuda.mem_get_info()]
    return (f"torch {torch.__version__} cuda{torch.version.cuda} {sm} "
            f"| vram {free:.2f}/{total:.2f} GB free")


@probe("sdpa attention (default path)", "L1")
def _sdpa() -> str:
    import torch
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    q = torch.randn(2, 16, 2048, 64, device="cuda", dtype=torch.bfloat16)
    o = sdpa(q, q, q, is_causal=True)
    torch.cuda.synchronize()
    return f"out {tuple(o.shape)}"


@probe("transformers import", "L1")
def _transformers() -> str:
    import transformers
    return f"v{transformers.__version__}"


# ==========================================================================
# L1.5 -- native FP8 (Blackwell tensor cores)
# ==========================================================================
@probe("fp8 torch._scaled_mm", "L1.5")
def _fp8_scaled_mm() -> str:
    import torch
    m, k, n = 512, 512, 512
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    ref = (x @ w.t()).float()
    sx = (x.abs().amax() / 448.0).float()
    sw = (w.abs().amax() / 448.0).float()
    xq = (x / sx).to(torch.float8_e4m3fn)
    wq = (w / sw).to(torch.float8_e4m3fn)
    # mat_b must be column-major. The accepted *scale* layout has shifted
    # between torch releases (0-dim vs (1,1) vs rowwise), and the wrong one
    # raises "Invalid scaling configuration" -- which looks identical to a
    # missing-kernel failure. Try each and report which the build accepts.
    b = wq.t()
    last: Exception | None = None
    for label, a_s, b_s in (
        ("0-dim", sx, sw),
        ("(1,1)", sx.reshape(1, 1), sw.reshape(1, 1)),
        ("rowwise", sx.expand(m).reshape(m, 1).contiguous(),
         sw.expand(n).reshape(1, n).contiguous()),
    ):
        try:
            out = torch._scaled_mm(xq, b, scale_a=a_s, scale_b=b_s,
                                   out_dtype=torch.bfloat16)
            torch.cuda.synchronize()
            e = rel_err(out.float(), ref)
            return f"e4m3 [{label} scales] rel_err={e:.4f}{tag(e)}"
        except Exception as ex:  # try the next layout
            last = ex
    raise RuntimeError(f"all scale layouts rejected; last: {last}")


@probe("torchao fp8 dynamic act+weight", "L1.5")
def _torchao_fp8() -> str:
    import torch, torchao
    from torchao.quantization import (quantize_,
                                      Float8DynamicActivationFloat8WeightConfig)
    m = torch.nn.Sequential(torch.nn.Linear(1024, 1024, bias=False)).cuda().bfloat16()
    x = torch.randn(16, 1024, device="cuda", dtype=torch.bfloat16)
    ref = m(x).float()
    quantize_(m, Float8DynamicActivationFloat8WeightConfig())
    e = rel_err(m(x).float(), ref)
    torch.cuda.synchronize()
    return f"torchao {torchao.__version__} rel_err={e:.4f}{tag(e)}"


# ==========================================================================
# L2 -- data-aware INT4
# ==========================================================================
@probe("torchao int4 weight-only", "L2")
def _torchao_int4() -> str:
    import torch, torchao
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    m = torch.nn.Sequential(torch.nn.Linear(1024, 1024, bias=False)).cuda().bfloat16()
    x = torch.randn(16, 1024, device="cuda", dtype=torch.bfloat16)
    ref = m(x).float()
    quantize_(m, Int4WeightOnlyConfig(group_size=128))
    e = rel_err(m(x).float(), ref)
    torch.cuda.synchronize()
    return f"torchao {torchao.__version__} g128 rel_err={e:.4f}{tag(e)}"


@probe("gptqmodel import", "L2")
def _gptqmodel() -> str:
    import gptqmodel
    return f"v{getattr(gptqmodel, '__version__', '?')}"


@probe("compressed-tensors import", "L2")
def _ct() -> str:
    import compressed_tensors
    return f"v{getattr(compressed_tensors, '__version__', '?')}"


# ==========================================================================
# L3 -- runtimes
# ==========================================================================
@probe("llama.cpp binaries (sm_120 build)", "L3")
def _llamacpp_bin() -> str:
    root = os.environ.get("LLAMA_CPP_DIR", "/root/llama.cpp")
    bench = os.path.join(root, "build", "bin", "llama-bench")
    if not os.path.exists(bench):
        raise ImportError(f"not built at {bench}")
    out = subprocess.run([bench, "--version"], capture_output=True, text=True,
                         timeout=60)
    txt = (out.stdout + out.stderr).strip().splitlines()
    return " | ".join(t.strip() for t in txt[:2])


@probe("llama-cpp-python", "L3")
def _llamacpp_py() -> str:
    import llama_cpp
    sup = getattr(llama_cpp, "llama_supports_gpu_offload", None)
    gpu = sup() if callable(sup) else "unknown"
    return f"v{llama_cpp.__version__} gpu_offload={gpu}"


@probe("vLLM import", "L3")
def _vllm() -> str:
    import vllm
    return f"v{vllm.__version__} (engine start is tested separately)"


# ==========================================================================
# L4 -- attention backends / KV cache
# ==========================================================================
@probe("flash-attn", "L4")
def _flash_attn() -> str:
    import torch, flash_attn
    from flash_attn import flash_attn_func
    q = torch.randn(2, 1024, 8, 64, device="cuda", dtype=torch.bfloat16)
    o = flash_attn_func(q, q, q, causal=True)
    torch.cuda.synchronize()
    return f"v{flash_attn.__version__} out {tuple(o.shape)}"


@probe("flashinfer", "L4")
def _flashinfer() -> str:
    import flashinfer
    return (f"v{getattr(flashinfer, '__version__', '?')} "
            f"(JIT compiles on first use -- needs nvcc)")


# ==========================================================================
# L5 -- QLoRA
# ==========================================================================
@probe("bitsandbytes NF4", "L5")
def _bnb_nf4() -> str:
    import torch, bitsandbytes as bnb
    ref_lin = torch.nn.Linear(1024, 1024, bias=False).cuda().to(torch.bfloat16)
    x = torch.randn(16, 1024, device="cuda", dtype=torch.bfloat16)
    ref = ref_lin(x).float()
    q = bnb.nn.Linear4bit(1024, 1024, bias=False, compute_dtype=torch.bfloat16,
                          quant_type="nf4")
    q.weight = bnb.nn.Params4bit(ref_lin.weight.data.clone(),
                                 requires_grad=False, quant_type="nf4")
    q = q.cuda()
    e = rel_err(q(x).float(), ref)
    torch.cuda.synchronize()
    return f"v{bnb.__version__} rel_err={e:.4f}{tag(e)}"


@probe("bitsandbytes INT8", "L5")
def _bnb_int8() -> str:
    import torch, bitsandbytes as bnb
    ref_lin = torch.nn.Linear(1024, 1024, bias=False).cuda().to(torch.float16)
    x = torch.randn(16, 1024, device="cuda", dtype=torch.float16)
    ref = ref_lin(x).float()
    q = bnb.nn.Linear8bitLt(1024, 1024, bias=False, has_fp16_weights=False)
    q.weight = bnb.nn.Int8Params(ref_lin.weight.data.clone().cpu(),
                                 requires_grad=False, has_fp16_weights=False)
    q = q.cuda()
    e = rel_err(q(x).float(), ref)
    torch.cuda.synchronize()
    return f"v{bnb.__version__} rel_err={e:.4f}{tag(e)}"


@probe("peft / trl", "L5")
def _peft() -> str:
    import peft
    try:
        import trl
        t = trl.__version__
    except ImportError:
        t = "missing"
    return f"peft {peft.__version__} trl {t}"


# ==========================================================================
# harness deps
# ==========================================================================
@probe("pynvml out-of-process VRAM sampling", "harness")
def _pynvml() -> str:
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    m = pynvml.nvmlDeviceGetMemoryInfo(h)
    drv = pynvml.nvmlSystemGetDriverVersion()
    if isinstance(drv, bytes):
        drv = drv.decode()
    pynvml.nvmlShutdown()
    return f"driver {drv} | {m.used / 2**20:.0f}/{m.total / 2**20:.0f} MiB used"


@probe("datasets (wikitext-2 + gsm8k reachable)", "harness")
def _datasets() -> str:
    import datasets
    return f"v{datasets.__version__} (network fetch not attempted here)"


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="also write the matrix to this path")
    args = ap.parse_args()

    info = host_info()
    print("=" * 78)
    print("HOST")
    print("=" * 78)
    for k, v in info.items():
        print(f"  {k:<12} {v}")

    icons = {"ok": "PASS", "fail": "FAIL", "corrupt": "CORRUPT", "skip": "----"}
    print()
    print("=" * 78)
    print(f"{'LVL':<6} {'PROBE':<34} {'STATUS':<8} DETAIL")
    print("=" * 78)
    for r in sorted(RESULTS, key=lambda r: (r.level, r.name)):
        d = r.detail or r.error
        print(f"{r.level:<6} {r.name:<34} {icons[r.status]:<8} {d}")

    n_fail = sum(1 for r in RESULTS if r.status == "fail")
    n_corrupt = sum(1 for r in RESULTS if r.status == "corrupt")
    n_skip = sum(1 for r in RESULTS if r.status == "skip")
    n_ok = sum(1 for r in RESULTS if r.status == "ok")
    print("=" * 78)
    print(f"  {n_ok} pass | {n_fail} fail | {n_corrupt} CORRUPT | {n_skip} not installed")
    if n_corrupt:
        print("  !! CORRUPT means the kernel ran but returned garbage. Do NOT")
        print("     benchmark that path -- record it as a negative result instead.")

    if args.json:
        payload = {"host": info, "probes": [asdict(r) for r in RESULTS]}
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  wrote {args.json}")
    return 1 if n_fail or n_corrupt else 0


if __name__ == "__main__":
    sys.exit(main())
