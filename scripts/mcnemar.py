#!/usr/bin/env python3
"""Paired comparison of two GSM8K runs (McNemar's test).

Two models evaluated on the same 200 problems are paired data. Comparing their
accuracies with independent binomial error bars throws away that pairing and is
badly underpowered: at n=200 and acc~0.7 the unpaired SE is ~3.2pp, so a real
5pp gap sits inside overlapping bars and looks like noise.

McNemar's test looks only at the DISCORDANT pairs -- problems where exactly one
model succeeded. Problems both get right (or both get wrong) carry no
information about which is better, and including them is what destroys the
power.

    python scripts/mcnemar.py results/gsm8k_BF16.json results/gsm8k_Q4_K_M.json
"""

from __future__ import annotations

import json
import math
import sys


def load(path: str):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    recs = {r["i"]: bool(r["correct"]) for r in d["records"]}
    return d, recs


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p under p=0.5 (exact McNemar)."""
    if n == 0:
        return 1.0
    def c(a, b):
        return math.comb(a, b)
    tail = sum(c(n, i) for i in range(0, min(k, n - k) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    da, ra = load(sys.argv[1])
    db, rb = load(sys.argv[2])
    shared = sorted(set(ra) & set(rb))
    if not shared:
        raise SystemExit("no overlapping problem indices")

    both = sum(1 for i in shared if ra[i] and rb[i])
    neither = sum(1 for i in shared if not ra[i] and not rb[i])
    only_a = sum(1 for i in shared if ra[i] and not rb[i])   # A right, B wrong
    only_b = sum(1 for i in shared if rb[i] and not ra[i])   # B right, A wrong
    n = len(shared)

    acc_a = sum(ra[i] for i in shared) / n
    acc_b = sum(rb[i] for i in shared) / n

    disc = only_a + only_b
    p = binom_two_sided(min(only_a, only_b), disc)

    A, B = da["label"], db["label"]
    print("=" * 68)
    print(f"  A = {A}   accuracy {acc_a:.4f}   ({sum(ra[i] for i in shared)}/{n})")
    print(f"  B = {B}   accuracy {acc_b:.4f}   ({sum(rb[i] for i in shared)}/{n})")
    print(f"  paired on {n} identical problems")
    print("=" * 68)
    print(f"  accuracy delta (B - A)   : {100 * (acc_b - acc_a):+.2f} pp")
    print()
    print("  contingency:")
    print(f"    both correct           : {both}")
    print(f"    both wrong             : {neither}")
    print(f"    only {A:<10} right : {only_a}")
    print(f"    only {B:<10} right : {only_b}")
    print()
    print(f"  discordant pairs         : {disc}   (only these carry signal)")
    print(f"  exact McNemar two-sided p: {p:.4g}")
    if disc:
        se = math.sqrt(disc) / n          # SE of the paired difference
        print(f"  paired SE of delta       : {100 * se:.2f} pp")
        print(f"  unpaired SE (for contrast): "
              f"{100 * math.sqrt(acc_a * (1 - acc_a) / n):.2f} pp each")
    print()
    if p < 0.05:
        worse = B if only_a > only_b else A
        print(f"  VERDICT: difference is significant (p<0.05); {worse} is worse.")
    else:
        print("  VERDICT: no significant difference at n="
              f"{n}. This is NOT evidence of equivalence -- report the")
        print("           confidence interval, and raise n if it matters.")
    print("=" * 68)


if __name__ == "__main__":
    main()
