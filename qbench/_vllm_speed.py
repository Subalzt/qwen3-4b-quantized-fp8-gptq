#!/usr/bin/env python3
"""vLLM speed helper — runs inside the vLLM venv, prints prefill/decode t/s.

Split from the main process because vLLM pins its own torch build. Invoked by
qbench.backends.VllmBackend. Argv: model prompt_tokens max_tokens n_seqs
"""
import os
import sys
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")


def main() -> int:
    model = sys.argv[1]
    ptoks = int(sys.argv[2])
    max_tokens = int(sys.argv[3])
    n_seqs = int(sys.argv[4])

    from vllm import LLM, SamplingParams

    llm = LLM(model=model, gpu_memory_utilization=0.85, max_model_len=4096,
              max_num_seqs=max(n_seqs, 1), enforce_eager=True, dtype="bfloat16")
    tok = llm.get_tokenizer()
    # A prompt of roughly the requested token length.
    prompt = ("def solve(x):\n    # " + "compute the result step by step " * 40)
    ids = tok(prompt).input_ids[:ptoks]
    prompt = tok.decode(ids)
    prompts = [prompt] * n_seqs

    # Prefill-only pass (1 token) to isolate prompt-processing throughput.
    sp1 = SamplingParams(temperature=0.0, max_tokens=1)
    t0 = time.time()
    llm.generate(prompts, sp1)
    t_prefill = time.time() - t0
    prefill_tps = (len(ids) * n_seqs) / t_prefill if t_prefill else float("nan")

    # Full generation; decode t/s = generated tokens / (total - prefill).
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    t_total = time.time() - t0
    gen = sum(len(o.outputs[0].token_ids) for o in outs)
    t_decode = max(t_total - t_prefill, 1e-6)
    decode_tps = gen / t_decode

    print(f"PREFILL_TPS {prefill_tps:.2f}")
    print(f"DECODE_TPS {decode_tps:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
