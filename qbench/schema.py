"""Uniform result schema. Every (format x runtime) cell emits rows with THESE columns.

Rule: never add a metric without adding it here first. The whole point of the
harness is that a row from llama.cpp and a row from vLLM are directly comparable.
"""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any

SCHEMA_VERSION = "1"


@dataclass
class Row:
    # --- identity -------------------------------------------------------
    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    timestamp: str = ""
    label: str = ""  # human name of the grid cell, e.g. "l1_bf16"
    level: str = ""  # roadmap level: L1, L1.5, L2, L3, L4, L5

    # --- what was run ---------------------------------------------------
    backend: str = ""  # hf | llamacpp | vllm | openai | dummy
    model_id: str = ""  # HF repo id or local path
    weight_format: str = ""  # bf16 | fp8 | int8 | gguf_q8_0 | awq_int4 | gptq_int4 ...
    quant_bits: str = ""  # 16 | 8 | 4 | mixed
    kv_cache_dtype: str = ""  # auto | fp16 | fp8 | q8_0 | q4_0
    attn_impl: str = ""  # sdpa | flash_attention_2 | flashinfer | xformers | n/a

    # --- task config ----------------------------------------------------
    task: str = ""  # speed | ppl | gsm8k
    exec_mode: str = ""  # single | batch | concurrent  <- do NOT compare across
    concurrency: int = 0
    prompt_tokens: int = 0
    max_new_tokens: int = 0
    n_requests: int = 0
    warmup: int = 0
    seed: int = 0

    # --- metric 1: memory (sampled out-of-process, see qbench/vram.py) ---
    vram_peak_mb: float = float("nan")
    vram_baseline_mb: float = float("nan")
    vram_delta_mb: float = float("nan")  # peak - baseline; the number to report
    vram_samples: int = 0
    vram_sample_hz: float = float("nan")
    host_ram_peak_mb: float = float("nan")

    # --- metric 2: speed, SPLIT (a combined tok/s hides the real story) --
    prefill_tps: float = float("nan")  # prompt tokens / prefill probe time
    decode_tps: float = float("nan")  # generated tokens / (total - prefill)
    e2e_tps: float = float("nan")  # generated tokens / wall clock, all reqs
    ttft_ms_mean: float = float("nan")
    ttft_ms_p50: float = float("nan")
    ttft_ms_p95: float = float("nan")
    ttft_source: str = ""  # stream | prefill_probe
    latency_s_mean: float = float("nan")
    latency_s_p95: float = float("nan")
    total_gen_tokens: int = 0
    wall_s: float = float("nan")

    # --- metric 3: quality ----------------------------------------------
    ppl: float = float("nan")
    ppl_dataset: str = ""
    ppl_seqlen: int = 0
    ppl_stride: int = 0
    ppl_n_tokens: int = 0  # scored target tokens; must match across cells
    ppl_tokenizer: str = ""  # tokenizer identity -- PPL is only comparable
    #                          across cells that share this string
    gsm8k_acc: float = float("nan")
    gsm8k_n: int = 0
    gsm8k_shots: int = 0

    # --- environment ----------------------------------------------------
    gpu_name: str = ""
    gpu_total_mb: float = float("nan")
    driver: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    arch_list: str = ""
    runtime_version: str = ""  # transformers / llama_cpp / vllm version
    host: str = ""

    # --- bookkeeping ----------------------------------------------------
    status: str = "ok"  # ok | error | skipped
    error: str = ""
    notes: str = ""
    extra: str = ""  # JSON blob for backend-specific knobs

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


FIELDS: list[str] = [f.name for f in fields(Row)]


def append_row(csv_path: str, row: Row) -> None:
    """Append one row, writing the header iff the file is new/empty."""
    path = os.path.abspath(csv_path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row.as_dict())
