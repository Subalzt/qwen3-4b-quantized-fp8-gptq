#!/usr/bin/env python3
"""Perplexity under transformers, using llama.cpp's exact protocol.

Round 2 produces compressed-tensors checkpoints that llama.cpp cannot read, so
those cells must be scored here. For the numbers to sit in the same table as
the GGUF cells, three things must match llama-perplexity exactly:

  1. Non-overlapping windows of n_ctx tokens (llama.cpp's stride == n_ctx).
  2. Every token in the window scored except the first (no context warm-up
     discount, no sliding overlap).
  3. PPL = exp(mean over chunks of the per-chunk mean NLL) -- llama.cpp
     averages per-chunk means, NOT per-token over the whole corpus. With equal
     chunk sizes these coincide, but the running series must match for the
     paired test to work.

It also prints the running series in llama.cpp's own `[n]value,` format, so
scripts/ppl_paired.py works on these logs unchanged.

CROSS-RUNTIME WARNING: a PPL from this script is NOT directly comparable to one
from llama-perplexity -- tokenization edge cases and kernel precision differ.
Score the SAME bf16 weights in both runtimes to measure the offset before
comparing any GGUF cell against any compressed-tensors cell.

    python scripts/ppl_hf.py --model Qwen/Qwen3-4B-Instruct-2507 \
        --file ~/data/wikitext-2-raw/wiki.test.raw --label BF16hf
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--chunks", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-gpu-gib", type=float, default=6.5,
                    help="leave headroom; BF16 4B will spill to CPU on 8GB")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    text = open(os.path.expanduser(a.file), encoding="utf-8").read()
    ids = tok(text, return_tensors="pt").input_ids[0]

    n_chunks = len(ids) // a.ctx
    if a.chunks:
        n_chunks = min(n_chunks, a.chunks)
    print(f"[{a.label}] {len(ids):,} tokens -> {n_chunks} chunks of {a.ctx}")

    # CPU budget must leave headroom: WSL has ~23 GiB total and the eval itself
    # plus the offloaded layers share it. 40 GiB (the old value) got the process
    # OOM-killed. A BF16 4B partially offloads here; a 4-bit checkpoint fits on
    # the GPU outright.
    model = AutoModelForCausalLM.from_pretrained(
        a.model,
        dtype="auto",
        device_map="auto",
        max_memory={0: f"{a.max_gpu_gib}GiB", "cpu": "16GiB"},
    )
    model.eval()

    nlls: list[float] = []
    series: list[str] = []
    t0 = time.time()
    for i in range(n_chunks):
        window = ids[i * a.ctx:(i + 1) * a.ctx].unsqueeze(0).to(model.device)
        with torch.no_grad():
            logits = model(window).logits[:, :-1]  # keep native dtype (bf16)
        tgt = window[:, 1:]
        # Per-token NLL = logsumexp(logits) - logit_at_target. Computing this in
        # sequence sub-chunks caps the float32 intermediate at
        # (sub x vocab x 4 bytes) ~= 150 MB instead of materializing the whole
        # (2048 x 151936) log-softmax tensor (~2.5 GB) that caused the OOM.
        L = logits.shape[1]
        sub = 256
        nll_sum = 0.0
        for s in range(0, L, sub):
            lg = logits[:, s:s + sub].float()
            tg = tgt[:, s:s + sub]
            lse = torch.logsumexp(lg, dim=-1)
            tok = lg.gather(-1, tg.unsqueeze(-1)).squeeze(-1)
            nll_sum += float((lse - tok).sum())
        nlls.append(nll_sum / L)
        running = math.exp(sum(nlls) / len(nlls))
        series.append(f"[{i + 1}]{running:.4f},")
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_chunks}  ppl={running:.4f}  "
                  f"({time.time() - t0:.0f}s)")

    mean_nll = sum(nlls) / len(nlls)
    ppl = math.exp(mean_nll)
    var = sum((x - mean_nll) ** 2 for x in nlls) / (len(nlls) - 1)
    se_nll = math.sqrt(var / len(nlls))
    se_ppl = ppl * se_nll  # delta method

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"ppl_{a.label}.log")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"model: {a.model}\nfile: {a.file}\nctx: {a.ctx}\n"
                 f"chunks: {n_chunks}\nruntime: transformers\n")
        fh.write("".join(series) + "\n")
        fh.write(f"Final estimate: PPL = {ppl:.4f} +/- {se_ppl:.5f}\n")

    print(f"[{a.label}] Final estimate: PPL = {ppl:.4f} +/- {se_ppl:.5f}  "
          f"({time.time() - t0:.0f}s)")
    print(f"[{a.label}] wrote {path}")


if __name__ == "__main__":
    main()
