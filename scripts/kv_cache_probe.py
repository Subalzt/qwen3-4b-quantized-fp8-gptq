#!/usr/bin/env python3
"""L4: measure KV-cache capacity and quality for fp16 vs fp8 KV in vLLM.

On an 8 GB card the KV cache, not the weights, is what caps context length and
concurrency once a 4-bit model is loaded. vLLM's kv_cache_dtype="fp8" halves the
per-token KV footprint, so the question is concrete and measurable: how many
more tokens fit, and does fp8 KV degrade output?

Must be a real file (vLLM spawns an EngineCore subprocess that re-imports
__main__). Reports the engine's own KV-cache block count -> token capacity, then
generates a fixed prompt to sanity-check coherence.

    VLLM_WSL2_ENABLE_PIN_MEMORY=1 python scripts/kv_cache_probe.py fp8
"""
import os
import sys
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

MODEL = os.path.expanduser("~/models/hf/R2-GPTQ")
PROMPT = ("Explain, step by step, how to set up a reverse SSH tunnel from a "
          "machine behind NAT to a public jump host, and how to harden it.")


def main() -> int:
    kv = sys.argv[1] if len(sys.argv) > 1 else "auto"
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=MODEL,
        kv_cache_dtype=kv,               # "auto" (fp16) or "fp8"
        gpu_memory_utilization=0.85,
        # Low enough that fp16 also starts; the KV POOL size is set by
        # gpu_memory_utilization, not this, so the fp8-vs-fp16 pool ratio is
        # what we read out.
        max_model_len=8192,
        enforce_eager=True,
        dtype="bfloat16",
    )
    load_s = time.time() - t0

    # cache_config moved under vllm_config in vLLM 0.26; try known paths.
    eng = llm.llm_engine
    cache = None
    for path in ("cache_config", "vllm_config.cache_config",
                 "model_config.cache_config"):
        obj = eng
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if obj is not None:
                cache = obj
                break
        except AttributeError:
            continue
    block = getattr(cache, "block_size", 0) if cache else 0
    n_blocks = getattr(cache, "num_gpu_blocks", None) if cache else None
    tokens = (n_blocks or 0) * block

    print(f"KV_RESULT dtype={kv} load_s={load_s:.0f} "
          f"block_size={block} gpu_blocks={n_blocks} kv_tokens={tokens}")

    # Quality: generate and print, to confirm fp8 KV is not garbage.
    sp = SamplingParams(temperature=0.0, max_tokens=120)
    out = llm.generate([PROMPT], sp)[0].outputs[0].text
    print(f"KV_SAMPLE dtype={kv} :: {out.replace(chr(10), ' ')[:400]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
