#!/usr/bin/env python3
"""HumanEval pass@1 against a running llama-server.

This is the eval the MCQ test could not provide: an OBJECTIVE, EXECUTABLE
measure of code-generation ability. Each of the 164 problems ships hidden unit
tests; a completion counts only if it actually passes them. That makes it the
right instrument for the one question that matters for a fine-tune -- did
training on the domain mixture preserve, improve, or DEGRADE real coding
ability (catastrophic forgetting)?

Greedy decoding (temperature 0) so pass@1 is reproducible. Writes
results/he_<label>.json in the {records:[{i,correct}]} shape that
scripts/mcnemar.py consumes, so base-vs-v1-vs-v2 is paired.

SAFETY: model-generated code is executed. Each candidate runs in a separate
subprocess with a hard wall-clock timeout and its own temp file. This is the
standard HumanEval risk; the timeout + non-root user bound the blast radius.

    python scripts/eval_humaneval.py --port 8090 --label v2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import requests

FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.S | re.I)


def extract_code(text: str, entry_point: str) -> str:
    """Pull the function definition out of a chat completion."""
    blocks = FENCE.findall(text)
    candidates = blocks if blocks else [text]
    # Prefer a block that actually defines the target function.
    for b in candidates:
        if f"def {entry_point}" in b:
            return b.strip()
    # else: the whole first block, or raw text
    return (blocks[0] if blocks else text).strip()


def build_program(prompt: str, completion: str, test: str,
                  entry_point: str) -> str:
    """Assemble a runnable program: completion + test harness + call."""
    # If the model re-emitted the signature, use the completion as-is;
    # otherwise the completion is a bare body to append to the prompt.
    if f"def {entry_point}" in completion:
        head = completion
    else:
        head = prompt + "\n" + completion
    return f"{head}\n\n{test}\n\ncheck({entry_point})\n"


def run_program(src: str, timeout: float = 12.0) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(src)
        path = fh.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=timeout, text=True)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval")["test"]
    n = len(ds) if a.n <= 0 else min(a.n, len(ds))

    url = f"http://127.0.0.1:{a.port}/v1/chat/completions"
    recs = []
    t0 = time.time()
    for i in range(n):
        row = ds[i]
        prompt, test = row["prompt"], row["test"]
        ep = row["entry_point"]
        body = {
            "messages": [
                {"role": "system",
                 "content": "You are an expert Python programmer. Complete the "
                            "function. Return only a single Python code block "
                            "with the full function, no explanation."},
                {"role": "user",
                 "content": f"Complete this function:\n\n```python\n{prompt}```"},
            ],
            "temperature": 0.0,
            "max_tokens": 640,
        }
        ok = False
        for attempt in range(3):
            try:
                r = requests.post(url, json=body, timeout=180)
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
                code = extract_code(text, ep)
                prog = build_program(prompt, code, test, ep)
                ok = run_program(prog)
                break
            except requests.RequestException:
                if attempt == 2:
                    ok = False
                time.sleep(2)
        recs.append({"i": i, "task_id": row["task_id"], "correct": ok})
        if (i + 1) % 20 == 0:
            acc = sum(x["correct"] for x in recs) / len(recs)
            print(f"  [{a.label}] {i + 1}/{n}  pass@1={acc:.3f}  "
                  f"({time.time() - t0:.0f}s)")

    n_ok = sum(x["correct"] for x in recs)
    acc = n_ok / len(recs)
    se = (acc * (1 - acc) / len(recs)) ** 0.5
    print(f"[{a.label}] pass@1 = {acc:.4f} ({n_ok}/{len(recs)})  "
          f"+/-{se:.4f}  wall={time.time() - t0:.0f}s")

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"he_{a.label}.json")
    with open(path, "w") as fh:
        json.dump({"label": a.label, "n": len(recs), "pass@1": acc,
                   "n_correct": n_ok, "stderr_unpaired": se,
                   "records": recs}, fh, indent=1)
    print(f"[{a.label}] wrote {path}")


if __name__ == "__main__":
    main()
