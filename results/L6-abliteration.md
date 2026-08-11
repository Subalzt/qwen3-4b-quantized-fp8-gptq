# L6 — Abliteration: the measured cost of removing refusal

_Removing the single "refusal direction" (Arditi et al. 2024, arXiv:2406.11717)
from the anchor model, then quantizing to FP8 — measured with the same paired
statistics as every other level._

## Method

`scripts/abliterate.py`. Closed-form, no training:

1. Collect last-token residual activations for 128 **refused** prompts (AdvBench
   `harmful_behaviors`, `goal` column) vs. 128 **harmless** prompts (Alpaca,
   no-input), per layer, on the BF16 anchor.
2. `refusal_dir[layer] = normalize(mean(harmful) − mean(harmless))`.
3. **Sweep** candidate layers: ablate that layer's direction from *every* layer at
   inference, measure refusal rate on 24 held-out harmful prompts, pick the winner.
4. Permanently orthogonalize every residual-writing matrix against the winning
   direction — `embed_tokens`, and each layer's `self_attn.o_proj` and
   `mlp.down_proj`: `W ← W − r̂(r̂ᵀW)`.
5. Save BF16 → re-quantize to FP8 with `scripts/quantize_hf.py --recipe fp8`
   (the *same* FP8_DYNAMIC recipe as the shipped R2-FP8).

You cannot edit an FP8/GPTQ checkpoint in place — the projection breaks the
quantization. The pipeline is bf16 → abliterate → re-quantize, by necessity.

## Layer sweep

| layer | depth | held-out refusal |
|---|---|---|
| 12 | 0.33 | 96% |
| 15 | 0.42 | 83% |
| 18 | 0.50 | 83% |
| 21 | 0.58 | 96% |
| **24** | **0.67** | **29%** ← winner |
| 27 | 0.75 | 33% |

Refusal is sharply localized around layer 24 (depth 0.67). Baseline (no ablation)
refuses 100%. After the permanent weight edit — better than the inference-hook
estimate because it also cleans `embed_tokens` — **held-out refusal = 17%**, and
FP8 quantization preserved that exactly (17% on both BF16 and FP8). ~1 of the 4
residual "refusals" is a false positive (a self-harm prompt answered with an
empathetic crisis redirect that trips the "I'm sorry" marker), so the effective
rate is ~12%.

## The cost — paired, on the FP8 checkpoints

Harness validated: original FP8 reproduced **PPL = 10.0415** to the digit
(matches L1.5/L2).

| metric | original FP8 | abliterated FP8 | Δ | paired test |
|---|---|---|---|---|
| Refusal (AdvBench held-out) | 100% | 17% | **−83 pp** | intended effect |
| HumanEval pass@1 | 0.8659 (142/164) | 0.8110 (133/164) | **−5.49 pp** | McNemar exact **p = 0.093** — n.s. |
| Perplexity (wikitext-2, ctx 2048) | 10.0415 | 11.9987 | **+19.49%** | paired t=69.2, **p = 2e-112**, 145/145 chunks worse |

## The finding: the metric determines the conclusion (again)

The two capability metrics **disagree**, and the disagreement is the result:

- **HumanEval (sparse, pass/fail, narrow capability):** −5.5 pp, does not reach
  significance. A functional code-generation skill largely survives.
- **Perplexity (dense, per-token, whole distribution):** +19.5%, every single one
  of 145 chunks worse, p ≈ 10⁻¹¹². The global distributional shift that a pass/fail
  eval hides is unmistakable to a dense measure.

This is the project's recurring thesis landing on abliteration: a 3-prompt smoke
test (all coherent — Fibonacci, a word problem, CAP theorem, all correct) and even
a 164-problem functional eval call it "fine"; the paired perplexity test calls it
"clearly degraded." Both are true; they answer different questions.

**Scale check.** +19.5% PPL is ~6× the +3.04% cost of going to 4-bit GPTQ (L2).
Removing the refusal direction is a *larger* quality intervention than
quantization itself — a fact that is invisible unless you measure it paired and
out of the box on a held-out corpus, which the published abliterated models do not.

## Artifacts

- `scripts/abliterate.py` — the pipeline (sweep + orthogonalize)
- `results/he_orig-fp8.json`, `results/he_ablit-fp8.json` — HumanEval records (paired via `mcnemar.py`)
- `results/ppl_orig-fp8.log`, `results/ppl_ablit-fp8.log` — per-chunk PPL series (paired via `ppl_paired.py`)
- Checkpoints (not in repo): `~/models/hf/R2-abliterated-{bf16,FP8}`

## Open

- Finer sweep (layers 22–26, step 1; more calibration prompts) to push refusal
  toward single digits — the coarse step-3 sweep found the region, not the optimum.
- Does the +19.5% PPL cost shrink if the direction is estimated from more prompts
  or ablated at fewer layers? The cost/effect frontier is unmeasured.
