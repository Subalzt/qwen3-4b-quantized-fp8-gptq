#!/usr/bin/env python3
"""Minimal vLLM engine + generation smoke test.

Must be a real file (not stdin/heredoc): vLLM's v1 engine spawns an EngineCore
subprocess that re-imports __main__, which fails for stdin. This is the gate for
L3 (concurrency sweep) and the vLLM FP8 path.
"""
import os
import sys
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/models/hf/R2-GPTQ")
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(
        model=model,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        max_num_seqs=32,
        enforce_eager=True,   # skip CUDA-graph capture (an sm_120 risk)
        dtype="bfloat16",
    )
    print(f"  ENGINE CONSTRUCTED in {time.time() - t0:.0f}s")
    sp = SamplingParams(temperature=0.0, max_tokens=64)
    out = llm.generate(
        ["Write a Python function that returns the nth Fibonacci number."], sp)
    print("  GENERATION OK:")
    print("   ", out[0].outputs[0].text.replace("\n", " ")[:300])
    print("  VLLM_GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
