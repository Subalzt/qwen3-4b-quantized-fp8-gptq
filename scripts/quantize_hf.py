#!/usr/bin/env python3
"""Produce compressed-tensors INT4 checkpoints with llm-compressor.

Calibration deliberately reuses the SAME pile-10k documents as round 1's
importance matrix (same seed, same selection). Holding calibration data fixed
means any difference between these checkpoints and round 1's is attributable to
the ALGORITHM rather than to the data -- which is the whole point of running
round 2 at all.

Recipes:
  gptq        GPTQ W4A16 g128                      (data-aware baseline)
  awq         AWQ  W4A16 g128                      (activation-aware)
  quip-gptq   Hadamard rotation -> GPTQ            (rotation lever)
  spin-gptq   learned rotation  -> GPTQ            (rotation lever)
  autoround   AutoRound W4A16 g128

    python scripts/quantize_hf.py --recipe gptq --out ~/models/hf/R2-GPTQ
"""

from __future__ import annotations

import argparse
import os
import random
import time

MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def calibration_dataset(n: int, seed: int = 1234, min_chars: int = 512,
                        max_chars: int = 8000):
    """Same documents round 1's imatrix saw. See scripts/build_calibration.py."""
    from datasets import Dataset, load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    texts = []
    for i in idx:
        t = ds[i]["text"]
        if not t or len(t) < min_chars:
            continue
        texts.append(t[:max_chars])
        if len(texts) >= n:
            break
    return Dataset.from_dict({"text": texts})


def build_recipe(name: str):
    from llmcompressor.modifiers.quantization import GPTQModifier
    ignore = ["lm_head"]

    if name == "gptq":
        return [GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore)]

    if name == "awq":
        from llmcompressor.modifiers.awq import AWQModifier
        return [AWQModifier(targets="Linear", scheme="W4A16", ignore=ignore)]

    if name == "quip-gptq":
        from llmcompressor.modifiers.transform import QuIPModifier
        # Rotation first: spread activation outliers across channels so the
        # subsequent GPTQ error compensation has an easier problem.
        return [QuIPModifier(targets="Linear", ignore=ignore),
                GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore)]

    if name == "spin-gptq":
        from llmcompressor.modifiers.transform import SpinQuantModifier
        return [SpinQuantModifier(targets="Linear", ignore=ignore),
                GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore)]

    if name == "autoround":
        from llmcompressor.modifiers.quantization import AutoRoundModifier
        return [AutoRoundModifier(targets="Linear", scheme="W4A16",
                                  ignore=ignore)]

    if name == "fp8":
        # L1.5 — native Blackwell FP8. FP8_DYNAMIC is weight-static +
        # activation-dynamic E4M3, which needs no calibration data (data-free),
        # so it is the natural FP8 baseline. Blackwell has FP8 tensor cores;
        # torch._scaled_mm was verified working in scripts/check_env.py.
        from llmcompressor.modifiers.quantization import QuantizationModifier
        return [QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC",
                                     ignore=ignore)]

    raise SystemExit(f"unknown recipe: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--seqlen", type=int, default=2048)
    a = ap.parse_args()

    out = os.path.expanduser(a.out)
    if os.path.isdir(out) and os.listdir(out):
        print(f"  {out} already populated; skipping")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot

    print(f"=== {a.recipe} -> {out} ===")
    tok = AutoTokenizer.from_pretrained(a.model)
    # Load to CPU on purpose. llm-compressor's sequential pipeline streams one
    # decoder layer at a time onto the GPU, which is what makes a 7.5 GiB bf16
    # model quantizable on an 8 GiB card at all.
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cpu")

    ds = calibration_dataset(a.samples)
    print(f"  calibration: {len(ds)} docs (pile-10k, seed 1234 -- same as "
          f"round 1 imatrix)")

    t0 = time.time()
    oneshot(
        model=model,
        tokenizer=tok,
        dataset=ds,
        recipe=build_recipe(a.recipe),
        max_seq_length=a.seqlen,
        num_calibration_samples=a.samples,
        output_dir=out,
    )
    dt = time.time() - t0

    total = 0
    for root, _, files in os.walk(out):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
    print(f"  done in {dt:.0f}s -> {total / 2**30:.3f} GiB")
    print(f"  QBENCH_CKPT={out}")


if __name__ == "__main__":
    main()
