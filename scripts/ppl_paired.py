#!/usr/bin/env python3
"""Paired per-chunk comparison of two llama-perplexity runs.

llama.cpp prints a RUNNING perplexity after each chunk, and reports a final
+/- that is the standard error across chunks. Comparing two models with those
independent error bars is underpowered and wrong: both runs scored the exact
same chunks in the same order, so the comparison is paired. Chunk-to-chunk
variance in text difficulty is huge and identical for both models; differencing
it out is what makes a 2% effect measurable at all.

Per-chunk mean NLL is recoverable from the running values:
    log(PPL_n) = (1/n) * sum_{i<=n} c_i      =>  c_n = n*log(PPL_n) - (n-1)*log(PPL_{n-1})

Usage:
    python scripts/ppl_paired.py results/ppl_BF16.log results/ppl_Q4_K_M.log
"""

from __future__ import annotations

import math
import re
import sys


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p under p=0.5 (sign test)."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def running_ppl(path: str) -> list[float]:
    """Pull the [n]value running-perplexity series out of a llama.cpp log."""
    text = open(path, encoding="utf-8", errors="replace").read()
    pairs = re.findall(r"\[(\d+)\](\d+\.\d+)", text)
    by_idx: dict[int, float] = {}
    for n, v in pairs:
        by_idx[int(n)] = float(v)
    if not by_idx:
        raise SystemExit(f"no running-perplexity series found in {path}")
    return [by_idx[i] for i in range(1, max(by_idx) + 1)]


def per_chunk_nll(ppl: list[float]) -> list[float]:
    out = []
    prev = 0.0
    for n, p in enumerate(ppl, start=1):
        cum = n * math.log(p)
        out.append(cum - prev)
        prev = cum
    return out


def final(path: str) -> str:
    for line in open(path, encoding="utf-8", errors="replace"):
        if "Final estimate" in line:
            return line.strip().split("Final estimate:")[-1].strip()
    return "?"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a_path, b_path = sys.argv[1], sys.argv[2]
    a_name = a_path.split("ppl_")[-1].replace(".log", "")
    b_name = b_path.split("ppl_")[-1].replace(".log", "")

    a_ppl, b_ppl = running_ppl(a_path), running_ppl(b_path)
    n = min(len(a_ppl), len(b_ppl))
    if len(a_ppl) != len(b_ppl):
        print(f"WARNING: chunk counts differ ({len(a_ppl)} vs {len(b_ppl)});"
              f" truncating to {n}. Results are NOT comparable unless the"
              f" same text and context were used.")
    a, b = per_chunk_nll(a_ppl)[:n], per_chunk_nll(b_ppl)[:n]

    d = [bi - ai for ai, bi in zip(a, b)]           # b minus a, in nats
    mean_d = sum(d) / n
    var = sum((x - mean_d) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    t = mean_d / se if se else float("inf")
    wins = sum(1 for x in d if x > 0)               # chunks where b is worse

    print("=" * 66)
    print(f"  A = {a_name:<12} final PPL {final(a_path)}")
    print(f"  B = {b_name:<12} final PPL {final(b_path)}")
    print("=" * 66)
    print(f"  chunks compared        : {n}")
    print(f"  PPL(A), PPL(B)         : {a_ppl[-1]:.4f}  ->  {b_ppl[-1]:.4f}")
    print(f"  absolute PPL delta     : {b_ppl[-1] - a_ppl[-1]:+.4f}")
    print(f"  relative PPL delta     : "
          f"{100 * (b_ppl[-1] / a_ppl[-1] - 1):+.2f} %")
    print()
    print("  --- paired, per chunk (this is the correct test) ---")
    print(f"  mean NLL difference    : {mean_d:+.6f} nats")
    print(f"  std error of the mean  : {se:.6f}")
    # Report an actual p-value. An arbitrary |t| cutoff mislabels genuine
    # effects: t=2.73 at df=144 is p~0.007, which is not "noise".
    try:
        from scipy import stats
        p_t = float(2 * stats.t.sf(abs(t), df=n - 1))
    except ImportError:
        p_t = float("nan")
    p_sign = binom_two_sided(min(wins, n - wins), n)

    print(f"  paired t statistic     : {t:.2f}  (df={n - 1})")
    print(f"  paired t p-value       : {p_t:.3g}")
    print(f"  chunks where B is worse: {wins}/{n}")
    print(f"  exact sign-test p      : {p_sign:.3g}")

    # The two tests answer different questions. The t-test is sensitive to a
    # few large-magnitude chunks; the sign test asks whether the direction is
    # consistent. Agreement is strong evidence; disagreement means the effect
    # is carried by magnitude on a minority of chunks and should be reported
    # as weaker than the t-value alone suggests.
    sig_t, sig_s = p_t < 0.05, p_sign < 0.05
    if sig_t and sig_s:
        verdict = "REAL — consistent in both direction and magnitude"
    elif sig_t and not sig_s:
        verdict = ("WEAK — magnitude-driven; direction is not consistent "
                   "across chunks")
    elif sig_s and not sig_t:
        verdict = "REAL in direction, small in magnitude"
    else:
        verdict = "NOT separable from noise at n=%d" % n
    print(f"  verdict                : {verdict}")
    print("=" * 66)


if __name__ == "__main__":
    main()
