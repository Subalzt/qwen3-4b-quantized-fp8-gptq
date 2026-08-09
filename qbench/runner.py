"""Grid runner: read a YAML grid, run each cell, emit one uniform CSV.

This is the "first deliverable" — ONE entry point that loads any (format,
runtime) cell and writes the same three metrics (VRAM, split speed, quality) to
a single comparable CSV. Every one-off result in results/ can be reproduced as a
cell here.

    python -m qbench.runner configs/grid.yaml --out results/grid.csv
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import platform
import socket
import subprocess
import traceback

from .backends import LlamaCppBackend, VllmBackend
from .schema import Row, append_row


def _env_row() -> dict:
    info = {"host": socket.gethostname(),
            "gpu_name": "", "gpu_total_mb": float("nan"), "driver": ""}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        name, total, drv = [x.strip() for x in out.split(",")]
        info.update(gpu_name=name, gpu_total_mb=float(total), driver=drv)
    except Exception:
        pass
    return info


def run_cell(cell: dict, corpus: str, env: dict) -> Row:
    now = _dt.datetime.now().isoformat(timespec="seconds")
    row = Row(
        run_id=cell.get("label", "") + "_" + now,
        timestamp=now,
        label=cell.get("label", ""),
        level=cell.get("level", ""),
        backend=cell["backend"],
        model_id=cell["model"],
        weight_format=cell.get("weight_format", ""),
        quant_bits=str(cell.get("quant_bits", "")),
        task=cell.get("task", "speed+ppl"),
        exec_mode="single",
        concurrency=1,
        prompt_tokens=cell.get("p", 512),
        max_new_tokens=cell.get("n", 128),
        **{k: env[k] for k in ("host", "gpu_name", "gpu_total_mb", "driver")},
    )
    try:
        model = os.path.expanduser(cell["model"])
        if cell["backend"] == "llamacpp":
            be = LlamaCppBackend()
            s = be.speed(model, ngl=cell.get("ngl", 99),
                         p=cell.get("p", 512), n=cell.get("n", 128))
            row.prefill_tps, row.decode_tps = s.prefill_tps, s.decode_tps
            row.vram_delta_mb, row.vram_peak_mb = s.vram_delta_mb, s.vram_peak_mb
            if cell.get("task", "").find("ppl") >= 0 or "ppl" in cell:
                q = be.perplexity(model, corpus, ctx=cell.get("ctx", 2048),
                                  ngl=cell.get("ngl", 99))
                row.ppl, row.ppl_n_tokens = q.ppl, q.ppl_n_tokens
                row.ppl_dataset = "wikitext-2-raw-v1"
                row.ppl_seqlen = cell.get("ctx", 2048)
        elif cell["backend"] == "vllm":
            be = VllmBackend()
            s = be.speed(model, prompt_tokens=cell.get("p", 512),
                         max_tokens=cell.get("n", 128),
                         n_seqs=cell.get("n_seqs", 1))
            row.prefill_tps, row.decode_tps = s.prefill_tps, s.decode_tps
            row.vram_delta_mb, row.vram_peak_mb = s.vram_delta_mb, s.vram_peak_mb
        else:
            raise ValueError(f"unknown backend {cell['backend']}")
        row.status = "ok"
    except Exception as e:  # a failing cell must not abort the grid
        row.status = "error"
        row.error = f"{type(e).__name__}: {e}"
        row.notes = traceback.format_exc().splitlines()[-1][:200]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default="results/grid.csv")
    ap.add_argument("--corpus",
                    default="~/data/wikitext-2-raw/wiki.test.raw")
    ap.add_argument("--only", default="", help="run only cells whose label "
                    "contains this substring")
    a = ap.parse_args()

    import yaml
    with open(a.config) as fh:
        cfg = yaml.safe_load(fh)
    cells = cfg["cells"]
    if a.only:
        cells = [c for c in cells if a.only in c.get("label", "")]

    env = _env_row()
    print(f"running {len(cells)} cells on {env['gpu_name']} -> {a.out}")
    for c in cells:
        print(f"  [{c.get('label')}] {c['backend']} {c.get('weight_format')} ...")
        row = run_cell(c, a.corpus, env)
        append_row(a.out, row)
        status = row.status.upper()
        print(f"    {status}  prefill={row.prefill_tps:.0f} "
              f"decode={row.decode_tps:.1f} vram={row.vram_delta_mb:.0f}MB "
              f"ppl={row.ppl:.4f}"
              + (f"  ({row.error})" if row.status == "error" else ""))
    print(f"done -> {a.out}")


if __name__ == "__main__":
    main()
