#!/usr/bin/env python3
"""Fetch the anchor model's safetensors checkpoint.

The GGUF files cover llama.cpp only. Everything else in the roadmap --
FP8 (L1.5), GPTQ/AWQ/rotation (L2), vLLM (L3), QLoRA (L5) -- operates on the
original safetensors weights.

    python scripts/fetch_model.py --repo Qwen/Qwen3-4B-Instruct-2507
"""

from __future__ import annotations

import argparse
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import snapshot_download

    t0 = time.time()
    path = snapshot_download(
        repo_id=a.repo,
        # .bin would be a duplicate of the safetensors; .gguf we already have.
        ignore_patterns=["*.gguf", "*.pth", "*.bin", "original/*"],
        max_workers=a.workers,
    )
    dt = time.time() - t0

    total = 0
    print(f"\n  snapshot: {path}")
    for root, _, files in os.walk(path):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            if os.path.islink(fp):
                fp = os.path.realpath(fp)
            try:
                sz = os.path.getsize(fp)
            except OSError:
                continue
            total += sz
            if sz > 1024 * 1024:
                print(f"    {fn:<44} {sz / 2**30:7.3f} GiB")
    print(f"  total: {total / 2**30:.2f} GiB in {dt:.0f}s "
          f"({total / 2**20 / max(dt, 1):.1f} MiB/s)")
    print(f"  QBENCH_MODEL_PATH={path}")


if __name__ == "__main__":
    main()
