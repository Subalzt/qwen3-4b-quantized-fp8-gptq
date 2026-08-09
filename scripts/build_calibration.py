#!/usr/bin/env python3
"""Build an importance-matrix calibration corpus.

Deliberate choice: the corpus is drawn from NeelNanda/pile-10k (web, code,
papers, books, dialogue) and contains NO wikitext.

That is not incidental. Our headline evaluation is wikitext-2 perplexity, so
calibrating on wikitext would let the quantized model fit the evaluation
distribution and produce a flattering number that says nothing about general
quality. Keeping calibration and evaluation domains disjoint is what makes the
comparison against a community quant meaningful rather than circular.

    python scripts/build_calibration.py --out data/calib_pile.txt --mb 1.6
"""

from __future__ import annotations

import argparse
import os
import random


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mb", type=float, default=1.6,
                    help="target size in MB (~4 chars/token)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--min-chars", type=int, default=512,
                    help="skip fragments too short to carry layer statistics")
    ap.add_argument("--max-chars", type=int, default=8000,
                    help="truncate very long docs so one document cannot "
                         "dominate the importance statistics")
    ap.add_argument("--skip-docs", type=int, default=0,
                    help="skip the first N qualifying docs. With the SAME "
                         "seed, this yields a held-out split disjoint from a "
                         "calibration set built with --skip-docs 0.")
    a = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")

    idx = list(range(len(ds)))
    random.Random(a.seed).shuffle(idx)

    target = int(a.mb * 1024 * 1024)
    chunks: list[str] = []
    total = 0
    used = 0
    skipped = 0
    for i in idx:
        t = ds[i]["text"]
        if not t or len(t) < a.min_chars:
            continue
        if skipped < a.skip_docs:
            skipped += 1
            continue
        t = t[: a.max_chars]
        chunks.append(t)
        total += len(t) + 2
        used += 1
        if total >= target:
            break

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(chunks))

    size = os.path.getsize(a.out)
    print(f"  wrote {a.out}")
    print(f"  documents      : {used}")
    print(f"  bytes          : {size} ({size / 2**20:.2f} MiB)")
    print(f"  approx tokens  : ~{size // 4:,}")
    print(f"  source         : NeelNanda/pile-10k, seed={a.seed}")
    print(f"  contains wikitext: NO (by design -- wikitext is the eval set)")


if __name__ == "__main__":
    main()
