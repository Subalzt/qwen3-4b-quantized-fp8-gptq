#!/usr/bin/env python3
"""Multiple-choice eval against a running llama-server.

Scores a model on preemware/pentesting-eval (241 held-out security MCQs). This
set was deliberately kept out of every training mixture, so it measures
generalization, not memorization.

Writes results/mcq_<label>.json in the same {records:[{i,correct}]} shape that
scripts/mcnemar.py consumes, so the base-vs-v1-vs-v2 comparison is paired
(McNemar) rather than a set of underpowered independent accuracies.

Scoring: the model is asked to reply with just a letter. We parse the first
A/B/C/D token. Greedy decoding (temperature 0) so a rerun reproduces exactly.

    python scripts/eval_mcq.py --port 8090 --label v2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import time

import requests

LETTERS = string.ascii_uppercase


def build_prompt(question: str, choices: list[str]) -> str:
    lines = [question, ""]
    for i, c in enumerate(choices):
        lines.append(f"{LETTERS[i]}. {c}")
    lines.append("")
    lines.append("Answer with ONLY the single letter of the correct option.")
    return "\n".join(lines)


def parse_letter(text: str, n_choices: int) -> int:
    """Return a 0-based choice index, or -1 if unparseable."""
    if not text:
        return -1
    # First standalone A-D (optionally like "A." or "(A)" or "Answer: A").
    m = re.search(r"\b([A-Z])\b", text.upper())
    if m:
        idx = LETTERS.index(m.group(1))
        if 0 <= idx < n_choices:
            return idx
    return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("preemware/pentesting-eval")["train"]
    n = len(ds) if a.n <= 0 else min(a.n, len(ds))

    url = f"http://127.0.0.1:{a.port}/v1/chat/completions"
    recs = []
    t0 = time.time()
    for i in range(n):
        row = ds[i]
        choices = list(row["choices"])
        gold = int(row["answer"])
        prompt = build_prompt(row["question"], choices)
        body = {
            "messages": [
                {"role": "system",
                 "content": "You are a cybersecurity expert taking a "
                            "multiple-choice exam. Respond with only the "
                            "letter of the correct answer."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 4,
        }
        pred = -1
        raw = ""
        for attempt in range(3):
            try:
                r = requests.post(url, json=body, timeout=120)
                r.raise_for_status()
                raw = r.json()["choices"][0]["message"]["content"]
                pred = parse_letter(raw, len(choices))
                break
            except requests.RequestException:
                if attempt == 2:
                    raw = "ERROR"
                time.sleep(2)
        recs.append({"i": i, "gold": gold, "pred": pred,
                     "correct": pred == gold, "raw": raw[:40]})
        if (i + 1) % 40 == 0:
            acc = sum(r["correct"] for r in recs) / len(recs)
            print(f"  [{a.label}] {i + 1}/{n}  acc={acc:.3f}  "
                  f"({time.time() - t0:.0f}s)")

    n_ok = sum(r["correct"] for r in recs)
    n_bad = sum(1 for r in recs if r["pred"] < 0)
    acc = n_ok / len(recs)
    se = (acc * (1 - acc) / len(recs)) ** 0.5
    print(f"[{a.label}] accuracy = {acc:.4f} ({n_ok}/{len(recs)})  "
          f"+/-{se:.4f}  unparsed={n_bad}  wall={time.time() - t0:.0f}s")

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"mcq_{a.label}.json")
    with open(path, "w") as fh:
        json.dump({"label": a.label, "n": len(recs), "accuracy": acc,
                   "n_correct": n_ok, "unparsed": n_bad,
                   "stderr_unpaired": se, "records": recs}, fh, indent=1)
    print(f"[{a.label}] wrote {path}")


if __name__ == "__main__":
    main()
