#!/usr/bin/env python3
"""Merge the LoRA adapter into the base weights and export to GGUF.

A LoRA adapter is useless on its own for llama.cpp inference, so the tuned
model only becomes usable after: merge -> convert -> quantize.

The merge runs on CPU in bf16. It needs ~15 GiB of host RAM (7.5 GiB for the
base plus the merged copy), which is why it does not touch the GPU at all --
the 8 GiB card cannot hold an unquantized 4B model plus its merge target.

    python scripts/export_gguf.py --adapter ~/models/lora-coder \
        --out ~/models/qwen3-4b-coder --quant Q4_K_M
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--llama-cpp", default=os.path.expanduser("~/llama.cpp"))
    ap.add_argument("--imatrix", default="",
                    help="optional imatrix .dat for the quantize step")
    ap.add_argument("--skip-merge", action="store_true")
    a = ap.parse_args()

    out = os.path.expanduser(a.out)
    merged = out + "-merged"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # ---------------------------------------------------------------- merge
    if not a.skip_merge:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("=== merging adapter into base (CPU, bf16) ===")
        base = AutoModelForCausalLM.from_pretrained(
            a.base, dtype=torch.bfloat16, device_map="cpu")
        model = PeftModel.from_pretrained(base, os.path.expanduser(a.adapter))
        model = model.merge_and_unload()
        model.save_pretrained(merged, safe_serialization=True)

        tok = AutoTokenizer.from_pretrained(os.path.expanduser(a.adapter))
        tok.save_pretrained(merged)
        print(f"  merged -> {merged}")
        del model, base

    # -------------------------------------------------------------- convert
    conv = os.path.join(a.llama_cpp, "convert_hf_to_gguf.py")
    if not os.path.exists(conv):
        raise SystemExit(f"missing {conv}")
    bf16_gguf = f"{out}-BF16.gguf"
    print("\n=== converting to GGUF (bf16) ===")
    r = subprocess.run([sys.executable, conv, merged,
                        "--outfile", bf16_gguf, "--outtype", "bf16"],
                       capture_output=True, text=True)
    print((r.stdout + r.stderr).strip().splitlines()[-6:] and
          "\n".join((r.stdout + r.stderr).strip().splitlines()[-6:]))
    if r.returncode != 0:
        raise SystemExit("convert_hf_to_gguf failed")

    # ------------------------------------------------------------- quantize
    qbin = os.path.join(a.llama_cpp, "build", "bin", "llama-quantize")
    qout = f"{out}-{a.quant}.gguf"
    cmd = [qbin]
    if a.imatrix:
        cmd += ["--imatrix", os.path.expanduser(a.imatrix)]
    cmd += [bf16_gguf, qout, a.quant, "16"]
    print(f"\n=== quantizing -> {a.quant} ===")
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-4:]
    print("\n".join(tail))

    print("\n=== artifacts ===")
    for f in (bf16_gguf, qout):
        if os.path.exists(f):
            print(f"  {os.path.basename(f):<44} {os.path.getsize(f)/2**30:.3f} GiB")
    print(f"\n  run it:\n    {a.llama_cpp}/build/bin/llama-server "
          f"-m {qout} -ngl 99 -c 4096")


if __name__ == "__main__":
    main()
