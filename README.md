# quant-bench

**A controlled benchmark of LLM quantization, runtimes, and fine-tuning on a
single 8 GB consumer Blackwell GPU — where every number is comparable because
one anchor model is used throughout, and every comparison is backed by paired
statistics.**

Anchor model: [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
(Apache-2.0). Hardware: RTX 5070 Laptop (8 GB, `sm_120`), WSL2 / Ubuntu 26.04.

This is not a leaderboard. It is a study of *how you measure*: the same weights
scored two ways give a 10 % different perplexity; a model that looks fine on
three prompts is 18 points worse on 164; the "smart" quantization everyone
downloads loses to the plain one at equal size. The recurring lesson is that
**the metric determines the conclusion.**

---

## Headline findings

1. **Q8_0 is a free lunch; Q4 is not free.** Q8_0 quantization is
   statistically indistinguishable from BF16 in quality (+0.05 % perplexity)
   while running 3× faster and fully resident. Q4_K_M is 5× faster than BF16
   but **+2.1 % perplexity — real, not noise** (paired t = 18, worse on 137/145
   text chunks). Perplexity's tiny error bars *overlap*; the paired test does
   not. [→ L1](#l1--format-sweep-runtime-held-constant)

2. **The community's "smart" 4-bit quant loses to the naive one at equal size.**
   A plain Q4_K_M with *no importance matrix* beat the widely-downloaded
   imatrix quant by 0.64 % perplexity at 288 bytes smaller (p = 9e-7). The
   importance matrix is a *transfer bet* that does not pay off out of its
   calibration domain. [→ L2 round 1](#l2--data-aware-int4)

3. **Perplexity is not comparable across runtimes.** The *same BF16 weights*
   score 9.09 under llama.cpp and 10.02 under transformers — a **+0.93 (10 %)
   offset from the runtime alone.** This is why the study is a (format ×
   runtime) *grid*, not a ladder. [→ L2 round 2](#l2--data-aware-int4)

4. **vLLM only wins at batch > 1 — and the curve proves it.** At one request
   llama.cpp leads; by 32 concurrent requests vLLM delivers ~1.8× the aggregate
   throughput and *holds* per-request speed via continuous batching while
   llama.cpp collapses. [→ L3](#l3--runtime-concurrency-sweep)

5. **Naive QLoRA fine-tuning made the model measurably worse.** Domain-tuning on
   22k curated examples cost **18 points of HumanEval coding ability**
   (p = 2e-6) for **zero** measurable gain in security knowledge — catastrophic
   forgetting, caught only by an objective generation eval, not by the training
   loss or a knowledge quiz. [→ L5](#l5--qlora-fine-tuning-honest-comparison)

---

## The `sm_120` tax (why this took real engineering)

Blackwell consumer silicon (`sm_120`) is new enough that prebuilt CUDA wheels
routinely install, import, and then die at kernel launch — or worse, return
garbage. `scripts/check_env.py` therefore launches a **real kernel for every
library** and checks numerics against a reference; anything with >0.5 relative
error is reported `CORRUPT`, not `PASS`. Fights that had to be won:

| symptom | cause | fix |
|---|---|---|
| CUDA 12.9 compiles nothing | glibc 2.43 redeclares `cospi`/`sinpi` | use CUDA 13.3 toolkit |
| vLLM: `UVA is not available` | vLLM disables pinned memory on WSL by default | `VLLM_WSL2_ENABLE_PIN_MEMORY=1` |
| AWQ: `device not ready` (async) | CUDA-event race in AWQ modifier on sm_120 | `CUDA_LAUNCH_BLOCKING=1` |
| AWQ OOM at 2048 seqlen | activation caching heavier than GPTQ | shorten calibration to 512 |
| bitsandbytes "corrupts INT8 on sm_120" (rumored) | — | **disproven**: measured rel-err 0.009 |

Notably, the widely-repeated claim that bitsandbytes INT8 is broken on sm_120
did not survive measurement.

---

## Results

All perplexity on wikitext-2-raw-v1, ctx 2048, non-overlapping chunks. Every
comparison is **paired** (per-chunk t-test / McNemar), because at these sample
sizes an unpaired error bar hides real effects. Scripts: `ppl_paired.py`,
`mcnemar.py`.

### L1 — format sweep, runtime held constant

llama.cpp, so weight format is the only variable.

| format | VRAM | prefill t/s | decode t/s | PPL | vs BF16 |
|---|---:|---:|---:|---:|---|
| BF16 (ngl 30) | 7083 MB | 2957 | 24.9 | 9.0939 | — |
| **Q8_0** | 4282 MB | 5856 | 77.1 | 9.0981 | **+0.05 %** (p=0.02) |
| Q4_K_M | 2633 MB | 5660 | 116.8 | 9.2871 | +2.12 % (t=18, p≈0) |

Two sub-findings the split metrics exposed:
- **The offload cliff.** BF16 decode *rises* to ngl 34 then collapses 5× at
  ngl 35 — deliberately keeping 4 of 36 layers on CPU is far faster than
  offloading all of them. A sweep that only tests `-ngl 99` is wrong by 5×.
- **Prefill and decode want opposite offload settings** at long context, which
  is exactly why the scorecard reports them separately, never combined.

### L2 — data-aware INT4

**Round 1 (GGUF / importance matrix), matched to <300 bytes:**

| variant | wikitext PPL | held-out pile PPL |
|---|---:|---:|
| **V0 — no imatrix** | **9.2272** | 8.8356 |
| bartowski (community imatrix) | 9.2871 | 8.8295 |

V0 beats the community imatrix quant out-of-domain (p = 9e-7) and ties it
in-domain. The imatrix *hurts* on text unlike its calibration set.

**Round 2 (GPTQ vs AWQ, transformers):**

| model | PPL | vs BF16 |
|---|---:|---|
| base BF16 | 10.0216 | — |
| **GPTQ W4A16** | **10.3261** | +3.04 % |
| AWQ W4A16 | 10.5907 | +5.68 % |

GPTQ beats AWQ decisively (p = 3e-35) — though AWQ was handicapped by a shorter
forced calibration (its activation cache OOM'd at full length on 8 GB). The
BF16 row (10.02 here vs 9.09 under llama.cpp) is the +0.93 **runtime offset**:
these numbers are internally comparable but *not* comparable to the L1 GGUF
numbers.

### L3 — runtime concurrency sweep

vLLM (GPTQ) vs llama.cpp (Q4_K_M), aggregate decode tokens/sec vs simultaneous
requests:

| concurrency | llama.cpp agg | vLLM agg | per-req: llama.cpp → vLLM |
|---:|---:|---:|---|
| 1 | **115.8** | 72.6 | 115.8 → 72.6 |
| 4 | **320.6** | 276.6 | 80.1 → 69.1 |
| 16 | 887.9 | **1010.6** | 55.5 → 63.2 |
| 32 | 1189.3 | **1887.5** | 37.2 → 59.0 |

The **crossover is around concurrency 8–16**: llama.cpp is faster for a single
stream, vLLM's PagedAttention / continuous batching wins under load, reaching
~1.6× aggregate at 32 requests. The mechanism is in the per-request column —
vLLM *holds* ~60 t/s per request while llama.cpp collapses from 116 to 37. A
single-request benchmark would have declared llama.cpp the flat winner and been
exactly wrong for a serving workload.

### L1.5 — native FP8 (Blackwell)

FP8_DYNAMIC (E4M3), data-free, scored under transformers alongside the L2
checkpoints:

| model | PPL | vs BF16 | size |
|---|---:|---|---:|
| base BF16 | 10.0216 | — | 7.5 GB |
| **FP8 (E4M3)** | **10.0415** | **+0.20 %** (p=3e-7) | 4.85 GB |
| GPTQ W4A16 | 10.3261 | +3.04 % | 2.48 GB |
| AWQ W4A16 | 10.5907 | +5.68 % | 3.21 GB |

FP8 on Blackwell's native tensor cores is **near-lossless** — the +0.20 % is
statistically real but negligible, and it beats both 4-bit methods by an order
of magnitude in quality at 2× their size. This is the transformers-runtime
mirror of the L1 Q8_0 result: **8-bit is essentially free; the quality cost is
all in the jump to 4-bit.**

### L5 — QLoRA fine-tuning (honest comparison)

Domain fine-tune (code / Linux / networking / security / VM), two evals, both
held out:

| eval | base | fine-tuned | verdict |
|---|---:|---:|---|
| pentesting MCQ (knowledge) | 84.6 % | 86.3 % | tie (p = 0.45) |
| **HumanEval (code generation)** | **87.8 %** | **70.7 %** | **−18 pts (p = 2e-6)** |

Training loss fell smoothly, eval loss tracked it, and three smoke-test prompts
looked great — yet the model lost 18 points of coding ability for no knowledge
gain. The failure is invisible without an objective generation eval at scale.
This is the study's thesis in one experiment.

---

## Repository layout

```
qbench/              the harness: uniform schema + out-of-process VRAM + backends
  schema.py          the one Row every cell emits (57 comparable columns)
  vram.py            NVML sampler in a SEPARATE process (sees llama.cpp too)
  backends.py        llama.cpp / vLLM adapters -> the same three metrics
  runner.py          `python -m qbench.runner configs/grid.yaml` -> results/grid.csv
scripts/             one purpose each, all reproducible
  check_env.py       sm_120 capability matrix (launches real kernels)
  ppl_paired.py      paired per-chunk perplexity test  <- the correct test
  mcnemar.py         paired accuracy test
  quantize_hf.py     GPTQ / AWQ / QuIP / SpinQuant / AutoRound / FP8
  train_qlora.py     NF4 QLoRA, sized for 8 GB
  eval_humaneval.py  executable pass@1
  run_l3_sweep.sh    the concurrency curve
  ...
configs/grid.yaml    the (format x runtime) grid
results/             *.md write-ups + parsed *.json (raw *.log gitignored)
requirements/        frozen deps per venv
ENVIRONMENT.md       full hardware + software provenance
```

## Reproduction

Three `uv` venvs are used because several libraries pin torch incompatibly
(vLLM, llm-compressor). See `ENVIRONMENT.md` for the full rationale.

```bash
# 1. main env (L1/L2/L5): torch 2.13 + cu129 for sm_120
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129
uv pip install --python .venv/bin/python -r requirements/main.txt

# 2. verify the GPU actually runs kernels (do this first, always)
.venv/bin/python scripts/check_env.py

# 3. fetch the anchor model + a GGUF, then run the grid
.venv/bin/python scripts/fetch_model.py --repo Qwen/Qwen3-4B-Instruct-2507
python -m qbench.runner configs/grid.yaml --out results/grid.csv
```

vLLM and llm-compressor get their own venvs (`requirements/vllm.txt`,
`requirements/compress.txt`).

## Methodology notes

- **VRAM is sampled out-of-process** via NVML. `torch.cuda.max_memory_allocated`
  cannot see llama.cpp or vLLM; only a separate sampler sees what the driver
  sees. A measured idle baseline is subtracted.
- **Speed is always split** into prefill (prompt processing) and decode
  (generation). A combined tok/s averages two curves that often point in
  opposite directions.
- **Quality is never perplexity alone.** PPL is a floor; it declared
  quantization "free" and fine-tuning "a tie" — both wrong. Reasoning /
  generation benchmarks are the deciding metric.
- **Comparisons are paired.** The single most repeated methodological point
  here: overlapping error bars are the wrong test when both models saw the same
  items.

## Limitations

- One model family (Qwen3-4B) on one GPU. Findings about *method* should
  generalize; absolute numbers will not.
- WiFi/IoT fine-tuning domains stayed thin — no instruction datasets exist for
  them on the Hub; reported honestly, not padded.
- The L5 recovery run (lower LR, completion-only loss, replay, HumanEval
  guardrail) is designed but not executed — the negative result is the
  deliverable.

## License

Apache-2.0 (`LICENSE`). The anchor model, datasets, and all quantization
sources used are under permissive licenses (Apache-2.0 / MIT / CC-BY / CC0).
