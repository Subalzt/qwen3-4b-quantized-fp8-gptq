#!/usr/bin/env python3
"""Concurrency sweep against an OpenAI-compatible server (vLLM or llama.cpp).

Measures the throughput CURVE: aggregate decode tokens/sec as a function of the
number of simultaneous requests. This is the measurement that separates the two
runtimes -- vLLM's continuous batching (PagedAttention) should keep scaling
aggregate throughput as concurrency rises, while llama.cpp's server saturates
much earlier. A single-request number hides this entirely.

Both servers expose /v1/completions, so the same client drives both.

    python scripts/concurrency_client.py --port 8100 --label vllm \
        --levels 1,4,16,32
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

PROMPT = ("Write a detailed technical explanation of how the TCP three-way "
          "handshake works, including SYN, SYN-ACK, and ACK, and why it "
          "prevents half-open connections. Be thorough.")


def one_request(url: str, max_tokens: int) -> dict:
    t0 = time.time()
    r = requests.post(url, json={
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }, timeout=600)
    r.raise_for_status()
    dt = time.time() - t0
    j = r.json()
    usage = j.get("usage", {})
    gen = usage.get("completion_tokens")
    if gen is None:  # fall back to counting
        gen = len(j["choices"][0].get("text", "").split())
    return {"latency": dt, "gen_tokens": gen}


def sweep_level(url: str, conc: int, max_tokens: int,
                warmup: bool = False) -> dict:
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        results = list(ex.map(
            lambda _: one_request(url, max_tokens), range(conc)))
    wall = time.time() - t0
    total_tokens = sum(r["gen_tokens"] for r in results)
    lats = sorted(r["latency"] for r in results)
    return {
        "concurrency": conc,
        "wall_s": round(wall, 3),
        "total_gen_tokens": total_tokens,
        "aggregate_tok_s": round(total_tokens / wall, 1),
        "per_req_tok_s": round(total_tokens / wall / conc, 1),
        "latency_s_mean": round(statistics.mean(lats), 3),
        "latency_s_p50": round(lats[len(lats) // 2], 3),
        "latency_s_p95": round(lats[min(len(lats) - 1,
                                        int(0.95 * len(lats)))], 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--levels", default="1,4,16,32")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    url = f"http://127.0.0.1:{a.port}/v1/completions"
    levels = [int(x) for x in a.levels.split(",")]

    # Warm up so level 1 is fair. vLLM pays a large one-time cost on its first
    # FULL-SIZE decode (not captured by a tiny 8-token ping), which otherwise
    # cratered the concurrency-1 number. Warm at the real max_tokens.
    for _ in range(3):
        try:
            one_request(url, a.max_tokens)
        except Exception as e:
            print(f"  warmup failed: {e}", file=sys.stderr)
            break

    rows = []
    print(f"[{a.label}] {'conc':>5} {'agg tok/s':>10} {'per-req':>8} "
          f"{'p50 lat':>8} {'p95 lat':>8}")
    for c in levels:
        row = sweep_level(url, c, a.max_tokens)
        rows.append(row)
        print(f"[{a.label}] {row['concurrency']:>5} "
              f"{row['aggregate_tok_s']:>10} {row['per_req_tok_s']:>8} "
              f"{row['latency_s_p50']:>8} {row['latency_s_p95']:>8}")

    path = f"{a.out}/concurrency_{a.label}.json"
    with open(path, "w") as fh:
        json.dump({"label": a.label, "max_tokens": a.max_tokens,
                   "rows": rows}, fh, indent=1)
    print(f"[{a.label}] wrote {path}")


if __name__ == "__main__":
    main()
