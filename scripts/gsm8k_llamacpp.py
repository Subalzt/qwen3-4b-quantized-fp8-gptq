#!/usr/bin/env python3
"""GSM8K 8-shot exact-match against a llama.cpp server.

Perplexity is a floor, not a verdict: it averages next-token likelihood over
ordinary prose and is structurally blind to multi-step reasoning failure, which
is exactly what low-bit quantization tends to cause. This is the measurement
that can actually distinguish "2% worse at predicting text" from "noticeably
worse at arithmetic reasoning".

Design choices that matter for comparability:
  * The SAME problems, in the same order, for every model -> paired analysis
    (see scripts/mcnemar.py). Unpaired accuracy comparison at n=200 is badly
    underpowered.
  * Few-shot exemplars come from the TRAIN split, never from test.
  * Greedy decoding (temperature 0) so a rerun reproduces exactly.
  * Plain completion format, not the chat template -- this is the standard
    few-shot GSM8K protocol and keeps results comparable to published numbers.

    python scripts/gsm8k_llamacpp.py --model ~/models/gguf/x.gguf \
        --label Q4_K_M --ngl 99 --n 200
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def gold_answer(text: str) -> str:
    m = ANS_RE.search(text)
    return norm_num(m.group(1)) if m else ""


def norm_num(s: str) -> str:
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def extract_pred(text: str) -> str:
    """Prefer the '#### x' the few-shot format teaches; else last number."""
    m = ANS_RE.search(text)
    if m:
        return norm_num(m.group(1))
    nums = NUM_RE.findall(text)
    return norm_num(nums[-1]) if nums else ""


def build_prompt(shots, question: str) -> str:
    parts = []
    for q, a in shots:
        parts.append(f"Question: {q}\nAnswer: {a}")
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


def wait_health(port: int, timeout: float = 600) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--shots", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--server", default=os.path.expanduser(
        "~/llama.cpp/build/bin/llama-server"))
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    from datasets import load_dataset
    test = load_dataset("openai/gsm8k", "main", split="test")
    train = load_dataset("openai/gsm8k", "main", split="train")

    # Fixed, seed-free selection: the first N. Deterministic and trivially
    # reproducible by anyone re-running this.
    shots = [(train[i]["question"], train[i]["answer"]) for i in range(a.shots)]
    items = [(test[i]["question"], gold_answer(test[i]["answer"]))
             for i in range(min(a.n, len(test)))]

    print(f"[{a.label}] {len(items)} problems, {a.shots}-shot, ngl={a.ngl}, "
          f"ctx={a.ctx}, parallel={a.parallel}")

    proc = subprocess.Popen(
        [a.server, "-m", a.model, "-ngl", str(a.ngl), "-c", str(a.ctx),
         "-np", str(a.parallel), "--host", "127.0.0.1", "--port", str(a.port),
         "--no-webui"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        if not wait_health(a.port):
            print("FATAL: server never became healthy")
            out, _ = proc.communicate(timeout=5)
            print((out or "")[-2000:])
            return 1
        print(f"[{a.label}] server up")

        url = f"http://127.0.0.1:{a.port}/completion"

        def ask(idx_q_gold):
            idx, q, gold = idx_q_gold
            body = {
                "prompt": build_prompt(shots, q),
                "temperature": 0.0,
                "top_k": 1,
                "n_predict": a.max_tokens,
                "stop": ["\n\nQuestion:", "\nQuestion:"],
                "cache_prompt": True,
            }
            for attempt in range(3):
                try:
                    r = requests.post(url, json=body, timeout=600)
                    r.raise_for_status()
                    text = r.json().get("content", "")
                    pred = extract_pred(text)
                    return {"i": idx, "gold": gold, "pred": pred,
                            "correct": bool(pred) and pred == gold,
                            "raw": text[-400:]}
                except requests.RequestException as e:
                    if attempt == 2:
                        return {"i": idx, "gold": gold, "pred": "",
                                "correct": False, "raw": f"ERROR {e}"}
                    time.sleep(3)

        t0 = time.time()
        recs = [None] * len(items)
        with ThreadPoolExecutor(max_workers=a.parallel) as ex:
            for k, rec in enumerate(ex.map(
                    ask, [(i, q, g) for i, (q, g) in enumerate(items)])):
                recs[rec["i"]] = rec
                if (k + 1) % 25 == 0:
                    acc = sum(r["correct"] for r in recs if r) / (k + 1)
                    print(f"  {k + 1}/{len(items)}  running acc={acc:.3f}  "
                          f"({time.time() - t0:.0f}s)")
        wall = time.time() - t0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()

    n_ok = sum(r["correct"] for r in recs)
    acc = n_ok / len(recs)
    # Binomial standard error -- the UNPAIRED one. Use mcnemar.py to compare
    # two models; this bar is far too wide to separate them on its own.
    se = (acc * (1 - acc) / len(recs)) ** 0.5
    print(f"[{a.label}] accuracy = {acc:.4f} ({n_ok}/{len(recs)})  "
          f"+/- {se:.4f}  wall={wall:.0f}s")

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"gsm8k_{a.label}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"label": a.label, "model": a.model, "ngl": a.ngl,
                   "shots": a.shots, "n": len(recs), "accuracy": acc,
                   "n_correct": n_ok, "stderr_unpaired": se,
                   "wall_s": wall, "ctx": a.ctx, "parallel": a.parallel,
                   "records": recs}, fh, indent=1)
    print(f"[{a.label}] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
