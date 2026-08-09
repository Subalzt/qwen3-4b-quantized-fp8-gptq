# Round 2 — data-aware INT4: GPTQ vs AWQ (+ the runtime offset)

Round 1 exhausted the GGUF/imatrix levers (all <1% PPL). Round 2 uses the
safetensors checkpoint and `llm-compressor` to run the heavier algorithms:
GPTQ and AWQ, both W4A16 g128, calibrated on the SAME pile-10k data (seed 1234)
as round 1.

## Environment fight (this is a result too)

GPTQ ran first try (33 min, 2.48 GiB). AWQ needed three fixes on the 8 GB
sm_120 card:

1. **OOM at seqlen 2048.** AWQ caches activations across all calibration
   samples to search per-channel scales — far heavier than GPTQ's layer-by-layer
   pass. Fixed by dropping calibration seqlen to 512.
2. **Hard crash wedged the GPU.** The OOM aborted with a corrupted pinned-memory
   allocator; every subsequent run failed with `CUDA error: device not ready`
   until a full `wsl --shutdown` reset the passthrough. A fresh-process health
   check passed in between — the wedge only bit real workloads.
3. **`cudaErrorNotReady` from `createEvent`.** An async CUDA-event race in the
   AWQ modifier on sm_120; fixed with `CUDA_LAUNCH_BLOCKING=1` (serializes the
   launches).

GPTQ hit none of these. On this hardware GPTQ is simply the more robust method —
independent of output quality.

## Checkpoints

| method | scheme | calib seqlen | size |
|---|---|---:|---:|
| GPTQ | W4A16 g128 sym | 2048 | 2.48 GiB |
| AWQ | W4A16 g128 asym | **512** | 3.21 GiB |

AWQ is larger (asymmetric stores an extra zero-point per group) and used a
shorter calibration (forced by the OOM). So this is **not** a perfectly matched
GPTQ-vs-AWQ head-to-head — it is two points on the size/quality frontier, each
shaped by what the 8 GB card allowed.

## Results — wikitext-2, ALL under transformers, ctx 2048, paired

| model | PPL | vs base | paired t | p |
|---|---:|---:|---:|---:|
| base BF16 | 10.0216 | — | — | — |
| **GPTQ** | **10.3261** | +3.04 % | 26.8 | 8e-58 |
| AWQ | 10.5907 | +5.68 % | 35.2 | 1e-72 |

GPTQ vs AWQ directly: **GPTQ is better by 2.56 %** (t=16.6, p=3e-35), winning
134/145 chunks.

### Finding 1 — GPTQ beats AWQ on this model

Clear and overwhelmingly significant. Caveat kept honest: AWQ was handicapped by
the shorter forced calibration, so the fair claim is "GPTQ-2048 beats AWQ-512 on
this hardware," not "the GPTQ algorithm beats the AWQ algorithm." The likely
reason AWQ lost is the 4x shorter calibration, not the method.

### Finding 2 — both degrade PPL vs BF16 (as expected at 4-bit)

+3% (GPTQ) to +5.7% (AWQ). These are larger relative drops than round 1's GGUF
Q4_K_M showed (+2.1%), but see the runtime caveat below before concluding GGUF
wins.

## Finding 3 — the runtime offset, measured

Same BF16 weights, two runtimes:

| runtime | BF16 wikitext-2 PPL |
|---|---:|
| llama.cpp (round 1) | 9.0939 |
| transformers (round 2) | 10.0216 |

**A +0.93 PPL (+10%) offset from the runtime alone**, on identical weights and
the same eval protocol. This is exactly why the project is a (format × runtime)
GRID and not a ladder: a perplexity number is meaningless without its runtime.

Consequence: the round-2 numbers (transformers) are **not** directly comparable
to the round-1 GGUF numbers (llama.cpp). GPTQ's 10.33 cannot be placed against
Q4_K_M's 9.29 without accounting for this offset. Naively subtracting it puts
GPTQ at ~9.40 — slightly worse than Q4_K_M's 9.29, but at **~4.25 bpw vs
~4.95 bpw**, i.e. a smaller model. On the size/quality Pareto they are close,
different points, and the honest comparison is the frontier, not a single
number.

## Bottom line

- **GPTQ is the better and more robust 4-bit method here** — lower PPL than AWQ,
  and it dodged all three sm_120 failures AWQ hit.
- **AWQ is usable on 8 GB only with a shortened calibration**, which costs it
  quality — a genuine hardware constraint, not a free choice.
- **The +0.93 runtime offset is the headline methodological result**: it
  quantifies why cross-runtime PPL comparison is invalid and vindicates the grid
  design. It should be measured once per model and reported alongside every
  cross-runtime claim.

## Not done

- Rotation (QuIP / SpinQuant → GPTQ) — the modifiers import on sm_120; a rotated
  variant would test whether spreading outliers recovers the GPTQ gap to BF16.
- A matched-calibration AWQ (seqlen 2048) — needs either a smaller calibration
  set that still fits, or activation offload, to make the GPTQ-vs-AWQ comparison
  clean.
