# L1 scouting — llama.cpp / CUDA 13.3 / sm_120

First real measurements. Not paper data yet (single runtime, no perplexity, no
reasoning bench), but enough to fix the L1 story and expose one methodology
problem that has to be settled before any cell is recorded.

- Host: RTX 5070 Laptop 8151 MiB, driver 610.47, WSL2 Ubuntu 26.04
- llama.cpp `a1f96d4`, built `-DCMAKE_CUDA_ARCHITECTURES=120`, CUDA 13.3
- Model: `Qwen3-4B-Instruct-2507`, 4.02B params, 36 layers
- Idle GPU baseline **386 MiB** (after moving display to iGPU + closing apps —
  was 1440 MiB with the dGPU driving the desktop)
- `llama-bench -p 512 -n 128`, r=3 (Q4_K_M) / r=2 (BF16)

## Results

| format | ngl | prefill t/s | decode t/s | VRAM delta |
|---|---:|---:|---:|---:|
| Q4_K_M | 36/36 | **5660.5 ± 420.6** | **116.82 ± 1.80** | **2633 MiB** |
| BF16 | 24/36 | 1328.6 ± 62.4 | 17.04 ± 0.29 | — |
| BF16 | 28/36 | 1815.9 ± 82.4 | 18.36 ± 0.18 | — |
| BF16 | **32/36** | **2410.4 ± 8.9** | **22.80 ± 0.13** | — |
| BF16 | 36/36 | 262.7 ± 0.9 | 4.87 ± 0.01 | 7169 MiB |

## Finding 1 — more GPU layers is not monotonically better

BF16 decode throughput rises 17.0 → 18.4 → 22.8 t/s from ngl 24 → 32, then
**collapses to 4.87 t/s at ngl 36**. Prefill collapses harder: 2410 → 263 t/s,
a 9.2x drop.

Deliberately keeping 4 of 36 layers on the CPU is **4.7x faster at decode and
9.2x faster at prefill** than the naive "offload everything" setting. The
optimum is an interior point, and a benchmark that only tests `-ngl 99` would
report BF16 at 4.87 t/s and be wrong by nearly 5x.

## Finding 2 — the collapse is driver sysmem fallback, and it is a confound

BF16 peak was **7555 MiB of 8151** (7169 above baseline) — the model did not
fit, but nothing OOMed. On this driver, CUDA silently spills the overflow to
host memory over PCIe instead of failing. So the failure mode on this hardware
is not a crash; it is a **quiet 5–9x slowdown that looks like a real
measurement**.

This has to be controlled before recording any cell:

- **Option A** — disable it (NVIDIA Control Panel → Manage 3D Settings →
  *CUDA — Sysmem Fallback Policy* → **Prefer No Sysmem Fallback**). Configs
  that don't fit then OOM honestly, and every recorded number is a
  genuinely-resident one.
- **Option B** — leave it on and treat "spilled" as a labelled condition, with
  the spill fraction recorded per row.

Either is defensible; mixing them silently within one table is not. **Option A
for the main grid** is cleaner, with a small deliberate Option-B sweep as its
own result — the ngl curve above is already most of that experiment.

The harness must therefore record `n_gpu_layers` and a `spilled` flag on every
row, not just VRAM peak.

## Finding 3 — the headline L1 comparison

Against the *best* BF16 configuration (ngl 32), not the naive one:

| | Q4_K_M vs best BF16 | Q4_K_M vs naive BF16 (ngl 36) |
|---|---:|---:|
| decode | **5.1x faster** | 24.0x faster |
| prefill | **2.3x faster** | 21.5x faster |
| VRAM | **2.7x smaller** | 2.7x smaller |

Quoting the 24x number would be the easy, dishonest version of this result. The
5.1x figure — quantized vs. the strongest baseline the hardware actually
supports — is the one that belongs in the paper.

## Finding 4 — the knee is at 34, and it moves with context

The coarse sweep (24/28/32/36) put the collapse between 32 and 36. A fine sweep
locates it exactly, at **512-token prompts**:

| ngl | prefill t/s | decode t/s |
|---:|---:|---:|
| 30 | 2009.5 | 20.93 |
| 31 | 2322.3 | 21.84 |
| 32 | 2490.3 | 23.14 |
| 33 | 2746.9 | 22.98 |
| **34** | **2957.1** | **24.93** |
| 35 | 245.4 | 5.11 |
| 36 | 260.1 | 5.05 |

One layer — 34 → 35 — costs **12x prefill and 4.9x decode**. The cliff is a
single layer wide, so any sweep with a step size >1 will mislocate it.

## Finding 5 — prefill and decode want DIFFERENT offload settings

Repeating at a realistic 4096-token prompt inverts the conclusion:

| ngl | pp4096 t/s | tg128 t/s |
|---:|---:|---:|
| **30** | **1929.7** | 19.54 |
| 32 | 1501.6 | 22.69 |
| 34 | 117.4 | 23.48 |
| **36** | 106.7 | **24.75** |

At 4096 context, **prefill is fastest at ngl 30 and decode is fastest at
ngl 36** — the opposite ends of the range. Prefill degrades 16x across that
span while decode *improves* 27%.

The mechanism: prefill streams 4096 tokens through every layer, so each spilled
layer costs a PCIe round trip per token and dominates. Decode processes one
token at a time and is memory-bandwidth-bound, so keeping more weights resident
wins even while the KV cache spills.

**There is no single "best BF16 configuration."** The optimum depends on the
workload's prompt:generation ratio. This is the sharpest possible vindication of
the scorecard rule that prefill and decode must be reported separately — a
combined tok/s number here would average two curves that point in opposite
directions and produce a figure that is wrong for every workload.

## Finding 6 — peak VRAM cannot detect spilling

Sampled peak VRAM across the cliff:

| ngl | peak MiB | decode t/s |
|---:|---:|---:|
| 34 | 7685 | 24.93 |
| 35 | 7719 | 5.11 |
| 36 | 7687 | 5.05 |

A **34 MiB** difference accompanies a **4.9x** performance collapse, and ngl 36
reports *less* peak VRAM than ngl 35. Once sysmem fallback engages the driver
pins VRAM near the ceiling and moves the overflow to host RAM, so the memory
number goes flat exactly where the interesting thing happens.

Consequence for the harness: **do not infer residency from peak VRAM.** Parse
llama.cpp's own buffer report (`CUDA0 model buffer size` vs `CPU model buffer
size`) and record the split explicitly.

## Finding 7 — quality: +2.12% perplexity, and it is unambiguously real

`llama-perplexity`, wikitext-2-raw-v1 test (1,287,656 bytes, 4358 rows),
**ctx 2048, 145 non-overlapping chunks, identical file and tokenizer** for both
cells. Only the weight format differs.

| format | ngl | PPL | VRAM delta | wall |
|---|---:|---:|---:|---:|
| BF16 | 30/36 | **9.0939 ± 0.06990** | 7083 MiB | 164 s |
| Q4_K_M | 36/36 | **9.2871 ± 0.07148** | 3136 MiB | 70 s |

Absolute delta **+0.1932 PPL**, relative **+2.12 %**.

### The reported error bars are the wrong test

Those ±0.07 figures are standard errors across chunks, and the two intervals
overlap. Comparing them as independent would suggest the difference is
marginal. It is not: **both runs scored the same 145 chunks in the same order**,
so the comparison is paired. Chunk-to-chunk variation in text difficulty is
enormous (running PPL ranges 7.99 → 10.66) and *identical* for both models.
Differencing it out is what makes a 2% effect measurable.

Paired per-chunk analysis (`scripts/ppl_paired.py`, recovering per-chunk NLL
from the running series via `c_n = n·log(PPL_n) − (n−1)·log(PPL_{n−1})`):

```
mean NLL difference    : +0.021022 nats
std error of the mean  :  0.001155
paired t statistic     : 18.20   (df = 144)
chunks where Q4 is worse: 137/145   (sign test p < 1e-25)
```

The degradation is real. Report the paired statistic, not the overlapping bars.

### Control: layer placement does not affect the PPL value

BF16 ran with 6 layers on CPU while Q4_K_M ran fully on GPU, so part of the gap
could in principle be CPU-vs-CUDA floating-point accumulation rather than
quantization. Tested directly (40 chunks):

| BF16 ngl | PPL |
|---:|---:|
| 20 | 8.5015 ± 0.12170 |
| 30 | 8.5015 ± 0.12170 |

Identical to four decimals. Offload is numerically neutral; the +0.1932 gap is
**entirely quantization**. This control should be re-run whenever a cell mixes
CPU and GPU execution.

## L1 summary (llama.cpp runtime held constant)

| metric | BF16 | Q4_K_M | Q4_K_M advantage |
|---|---:|---:|---|
| VRAM | 7083 MiB | 3136 MiB | **2.26x smaller** |
| prefill @512 (best ngl) | 2957 t/s | 5660 t/s | 1.9x |
| decode (best ngl) | 24.93 t/s | 116.82 t/s | **4.7x** |
| perplexity | 9.0939 | 9.2871 | **+2.12 % worse** |

Runtime is held constant, so weight format is the only variable. This is a
clean L1 row.

## The trap this result sets

+2.12% perplexity reads as "essentially free", and that is exactly the false
conclusion the scorecard warns about. Perplexity is an averaged next-token
likelihood over ordinary prose; it is dominated by easy tokens and is
structurally insensitive to the multi-step-reasoning failures that quantization
tends to cause. A model can lose 2% PPL and 15% GSM8K.

**Nothing about quality is settled until a reasoning benchmark is run.** The
perplexity number is a floor, not a verdict.

## Still open

- No Q8_0 yet. It is the largest fully-resident format and the most
  informative single L1 cell.
- **No reasoning benchmark.** GSM8K 8-shot is the immediate next measurement;
  without it the quality column is one-dimensional.
- llama.cpp only. Runtime effects and format effects are still confounded until
  the same weights run under vLLM/HF.
- BF16 numbers are all sysmem-fallback-eligible. Rerun under
  *Prefer No Sysmem Fallback* before these become paper figures.
