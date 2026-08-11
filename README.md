# Qwen3-4B Quantization — FP8 & GPTQ

**Taking one small model — [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
(Apache-2.0) — from BF16 all the way to a quantized, fine-tuned coder on a single
8 GB laptop GPU, measuring every trade-off along the way.**

> Hardware: RTX 5070 Laptop · 8 GB · Blackwell `sm_120` · WSL2 / Ubuntu 26.04
> One anchor model throughout, so every number is comparable. Every comparison
> is backed by **paired statistics**, not eyeballed error bars.

Models produced here are on the Hub:
[FP8](https://huggingface.co/Subalzt/Qwen3-4B-Instruct-2507-FP8) ·
[GPTQ-W4A16](https://huggingface.co/Subalzt/Qwen3-4B-Instruct-2507-GPTQ-W4A16) ·
[abliterated-FP8](https://huggingface.co/Subalzt/Qwen3-4B-Instruct-2507-abliterated-FP8) (research artifact)

---

## Goals

- Measure how **quantization** (BF16 → FP8 → INT4) trades VRAM, speed, and quality.
- Separate **format effects** from **runtime effects** (llama.cpp vs vLLM vs transformers).
- Do it all on **8 GB consumer hardware**, where the constraints are real.
- Push past benchmarking: **fine-tune a domain "coder"** and honestly test whether it worked.
- Prove every claim with the **right metric and a paired test** — not vibes.

## What's installed / built

- **Runtimes:** llama.cpp (built from source for `sm_120`), vLLM 0.26, transformers 5.
- **Quantizers:** llm-compressor (GPTQ / AWQ / FP8), GGUF `llama-quantize` + imatrix, bitsandbytes NF4.
- **Training:** PEFT + TRL QLoRA, sized for 8 GB.
- **Toolchain:** CUDA 13.3, PyTorch 2.13 + cu129, 3 isolated `uv` venvs (torch pins conflict).
- **Harness:** `qbench/` — one schema, out-of-process VRAM sampler, backend adapters → uniform CSV.

## What was done (the grid)

| level | experiment | status |
|---|---|:--:|
| **L1** | format sweep BF16 / Q8_0 / Q4_K_M (llama.cpp) | done |
| **L1.5** | native Blackwell FP8 (E4M3) | done |
| **L2·1** | GGUF importance-matrix calibration ablation | done |
| **L2·2** | data-aware INT4: GPTQ vs AWQ | done |
| **L3** | vLLM vs llama.cpp concurrency curve | done |
| **L4** | fp8 KV-cache + long context | done |
| **L5** | QLoRA domain "coder" fine-tune + honest eval | done |
| **L6** | abliteration (refusal-direction removal) + measured cost | done |

---

## Headline findings (what was accomplished)

1. **8-bit is a free lunch; 4-bit is not.** Q8_0 / FP8 are quality-indistinguishable
   from BF16 (≤ +0.2 %) at 2–3× the speed / half the memory. The quality cost is
   *entirely* in the jump to 4-bit (+2–6 %).
2. **The community's "smart" 4-bit quant loses to the plain one.** A no-imatrix
   Q4_K_M beat the widely-downloaded imatrix quant at equal size (p = 9e-7). The
   importance matrix is a *transfer bet* that doesn't pay off out of domain.
3. **Perplexity isn't comparable across runtimes.** Same BF16 weights = 9.09
   (llama.cpp) vs 10.02 (transformers): a **+0.93 offset from the runtime alone.**
   Hence a *grid*, not a ladder.
4. **GPTQ > AWQ** on this model (p = 3e-35) — and it was far more robust on 8 GB.
5. **vLLM only wins past ~8 concurrent requests** — the crossover the throughput
   curve makes explicit; a single-request test would pick the wrong runtime.
6. **fp8 KV cache is a free 2×** context/concurrency (25,968 → 51,936 tokens),
   zero quality loss.
7. **The "coder" fine-tune backfired.** −18 HumanEval points for zero knowledge
   gain — catastrophic forgetting, invisible to training loss and smoke tests,
   caught only by an objective eval at scale. **The metric determines the conclusion.**
8. **Abliteration costs more than quantization.** Removing the refusal direction
   (Arditi 2024) cut refusals 100 % → ~17 %, but raised perplexity **+19.5 %** —
   roughly **6× the +3.04 %** cost of 4-bit GPTQ — while HumanEval fell only −5.5 pts
   (p = 0.093, *not* significant). Code eval says "fine," per-token PPL says "clearly
   degraded" (145/145 chunks worse). The metric determines the conclusion, again.

---

## Results — the comparison matrix

### Quantization scorecard — llama.cpp / GGUF (speed + VRAM + quality, one runtime)

| format | bits | VRAM | decode t/s | PPL | Δ quality |
|---|:--:|---:|---:|---:|---|
| BF16 | 16 | 7083 MB | 24.9 | 9.0939 | baseline |
| **Q8_0** | 8 | 4282 MB | 77.1 | 9.0981 | **+0.05 %** (negligible) |
| **Q4_K_M** | 4 | 2633 MB | **116.8** | 9.2871 | +2.12 % (real, p≈0) |

### Quantization scorecard — safetensors, (the quantized checkpoints)

| format | bits | VRAM | decode t/s | PPL | Δ vs BF16 |
|---|:--:|---:|---:|---:|---|
| BF16 | 16 | >8 GB (spills) | — | 10.0216 | baseline |
| **FP8 (E4M3)** | 8 | 5.5 GB | 47.8 | 10.0415 | **+0.20 %** (near-lossless) |
| **GPTQ W4A16** | 4 | **2.7 GB** | **58.0** | 10.3261 | +3.04 % |
| AWQ W4A16 | 4 | 3.8 GB | 57.5 | 10.5907 | +5.68 % |

Note the ordering: **4-bit decodes faster than 8-bit** (fewer weight bytes to read
per token — decode is memory-bandwidth bound), and GPTQ is smallest+fastest while
FP8 keeps the best quality. VRAM = transformers load (incl. ~1.3 GB CUDA context);
decode t/s = single-stream vLLM; weights-on-disk are 4.85 / 2.48 / 3.21 GB.

*(this table and the GGUF one above use different runtimes — see finding #3 — so read within a table, not across)*

### Runtime under load — L3 concurrency (aggregate decode t/s)

| concurrent requests | llama.cpp | vLLM | winner |
|---:|---:|---:|:--:|
| 1 | **115.8** | 72.6 | llama.cpp |
| 4 | **320.6** | 276.6 | llama.cpp |
| 16 | 887.9 | **1010.6** | vLLM |
| 32 | 1189.3 | **1887.5** | vLLM (1.6×) |

### The coder fine-tune — L5 (both evals held out)

| eval | base | fine-tuned | verdict |
|---|---:|---:|---|
| pentesting MCQ (knowledge) | 84.6 % | 86.3 % | tie (p = 0.45) |
| **HumanEval (code generation)** | **87.8 %** | **70.7 %** | **−18 pts (p = 2e-6)** |

### Abliteration — L6 (paired, vs the non-abliterated FP8)

| metric | base FP8 | abliterated FP8 | Δ | paired test |
|---|---:|---:|---:|---|
| refusal (AdvBench held-out) | 100 % | ~17 % | −83 pp | intended effect |
| HumanEval pass@1 | 86.6 % | 81.1 % | −5.5 pp | McNemar p = 0.093 (n.s.) |
| perplexity (wikitext-2) | 10.0415 | 11.9987 | **+19.5 %** | paired t, p ≈ 2e-112 |

The two capability metrics disagree — a functional code eval sees no significant
loss, a dense per-token measure sees a large one. Harness validated: the base FP8
reproduced PPL 10.0415 to the digit. → [`results/L6-abliteration.md`](results/L6-abliteration.md)

<details>
<summary><b>More detail per level</b> (offload cliff, imatrix ablation, KV cache)</summary>

- **L1 offload cliff:** BF16 decode *rises* to `ngl 34` then collapses 5× at
  `ngl 35` — keeping 4 of 36 layers on CPU beats offloading all of them. A sweep
  that only tests `-ngl 99` is wrong by 5×. Prefill and decode want *opposite*
  offload settings, which is why speed is always reported split.
- **L2·1 imatrix ablation** (matched to < 300 bytes): no-imatrix `9.2272` vs
  community imatrix `9.2871` on wikitext (out-of-domain, p = 9e-7); a tie on
  held-out pile (in-domain). The imatrix *hurts* on text unlike its calibration set.
- **L4 KV cache:** `3246 = 2 × 1623` GPU blocks exactly; fp16 can't serve a 32k
  request (~26k cap), fp8 reaches ~52k, with near-identical greedy output.
  → [`results/L4_kv_cache.md`](results/L4_kv_cache.md)
- Full write-ups: [`results/`](results/) (one `.md` per level).
</details>

---

## The `sm_120` tax (why this needed real engineering)

Blackwell consumer silicon is new enough that prebuilt CUDA wheels install,
import, then die at kernel launch — or return garbage. `scripts/check_env.py`
launches a **real kernel per library** and checks numerics; > 0.5 rel-error is
reported `CORRUPT`, not `PASS`. Fights won:

| symptom | cause | fix |
|---|---|---|
| CUDA 12.9 compiles nothing | glibc 2.43 redeclares `cospi`/`sinpi` | CUDA **13.3** toolkit |
| vLLM `UVA is not available` | vLLM disables pinned memory on WSL | `VLLM_WSL2_ENABLE_PIN_MEMORY=1` |
| AWQ `device not ready` | CUDA-event race in AWQ on sm_120 | `CUDA_LAUNCH_BLOCKING=1` |
| AWQ OOM at 2048 seqlen | activation caching heavier than GPTQ | calibration seqlen 512 |
| "bitsandbytes INT8 corrupts on sm_120" | rumor | **disproven** — measured rel-err 0.009 |

---

## Repository layout

```
qbench/              the harness
  schema.py          the one Row every cell emits (57 comparable columns)
  vram.py            NVML sampler in a SEPARATE process (sees llama.cpp too)
  backends.py        llama.cpp / vLLM adapters -> the same 3 metrics
  runner.py          python -m qbench.runner configs/grid.yaml -> results/grid.csv
scripts/             one purpose each, all reproducible
  check_env.py       sm_120 capability matrix (launches real kernels)
  ppl_paired.py      paired per-chunk perplexity test   <- the correct test
  mcnemar.py         paired accuracy test
  quantize_hf.py     GPTQ / AWQ / QuIP / SpinQuant / AutoRound / FP8
  train_qlora.py     NF4 QLoRA, sized for 8 GB
  eval_humaneval.py  executable pass@1
  abliterate.py      refusal-direction removal (Arditi 2024) + layer sweep
configs/grid.yaml    the (format x runtime) grid
results/             *.md write-ups + parsed *.json  (grid.csv = the uniform output)
ENVIRONMENT.md       full hardware + software provenance
```

## Reproduction

```bash
# 1. main env (L1/L2/L5): torch 2.13 + cu129 for sm_120
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129
uv pip install --python .venv/bin/python -r requirements/main.txt

# 2. verify the GPU actually runs kernels (do this first, ALWAYS)
.venv/bin/python scripts/check_env.py

# 3. fetch the model + a GGUF, then run the grid
.venv/bin/python scripts/fetch_model.py --repo Qwen/Qwen3-4B-Instruct-2507
python -m qbench.runner configs/grid.yaml --out results/grid.csv
```

vLLM and llm-compressor get their own venvs (`requirements/vllm.txt`,
`requirements/compress.txt`) — several libraries pin torch incompatibly. See
`ENVIRONMENT.md` for the full rationale.

## Methodology (the four rules)

- **VRAM sampled out-of-process** (NVML) — `torch.cuda.max_memory_allocated`
  can't see llama.cpp or vLLM. Idle baseline subtracted.
- **Speed always split** into prefill vs decode — a combined tok/s averages two
  curves that point opposite ways.
- **Quality never perplexity alone** — PPL called quantization "free" and the
  fine-tune "a tie"; both wrong. Generation/reasoning benchmarks decide.
- **Comparisons are paired** — overlapping error bars are the wrong test when
  both models saw the same items.

## Limitations

- One model family on one GPU. *Method* findings should generalize; absolute
  numbers won't.
- WiFi/IoT fine-tune domains stayed thin — no instruction datasets exist for them
  on the Hub; reported honestly, not padded.
- The coder **recovery run** (lower LR, completion-only loss, replay, HumanEval
  guardrail) is designed but not executed — the negative result is the deliverable.
- L4's "free 2×" is proven for VRAM/near-term quality, not adversarial
  long-context recall (needle-in-a-haystack).

## License

Apache-2.0 (`LICENSE`). Anchor model, datasets, and all quantization sources are
permissive (Apache-2.0 / MIT / CC-BY / CC0). Derivative checkpoints inherit
Apache-2.0 with attribution to the Qwen team.
