# Round 1 — can we beat the community Q4_K_M?

Goal: produce a 4-bit Qwen3-4B that beats the widely-used community quant at
equal size. Round 1 attacks the two levers available without the safetensors
checkpoint: **importance-matrix calibration data** and **per-tensor bit
allocation**.

## Setup

Source: `Qwen3-4B-Instruct-2507` BF16 GGUF (7.49 GiB), quantized locally with
`llama-quantize` (llama.cpp `a1f96d4`).

| variant | recipe | bytes | vs incumbent |
|---|---|---:|---:|
| **bartowski** (incumbent) | Q4_K_M, their imatrix | 2,497,280,736 | — |
| **V0-noimat** | Q4_K_M, **no imatrix at all** | 2,497,280,448 | −288 B |
| **V1-pileimat** | Q4_K_M, our pile-10k imatrix | 2,497,280,672 | −64 B |
| **V2-v6d6** | V1 + `attn_v=q6_K`, `ffn_down=q6_K` | 2,625,014,432 | **+5.1 %** |

V0, V1 and the incumbent are within **288 bytes** of each other — same format,
same recipe, so calibration is genuinely the only variable.

Calibration corpus: 532 documents / 1.61 MiB from `NeelNanda/pile-10k`,
containing **no wikitext by construction**, so wikitext-2 stays a true
out-of-domain evaluation. A disjoint 428-document held-out pile split (same
seed, `--skip-docs 532`) gives the in-domain evaluation.

Both evaluations: ctx 2048, non-overlapping chunks, identical file and
tokenizer per eval. All comparisons paired per chunk.

## Results

### Out-of-domain (wikitext-2, 145 chunks)

| variant | PPL |
|---|---:|
| BF16 reference | 9.0939 |
| **V0-noimat** | **9.2272** |
| V2-v6d6 | 9.2275 |
| V1-pileimat | 9.2695 |
| bartowski | 9.2871 |

### In-domain (held-out pile, 161 chunks)

| variant | PPL |
|---|---:|
| **V2-v6d6** | **8.7922** |
| V1-pileimat | 8.8274 |
| bartowski | 8.8295 |
| V0-noimat | 8.8356 |

**The ranking inverts.** V0 is best out-of-domain and worst in-domain.

## Paired tests

| comparison | eval | Δ PPL | t (df) | p (t) | p (sign) | verdict |
|---|---|---:|---:|---:|---:|---|
| V0 vs bartowski | wikitext | **−0.64 %** | −5.13 (144) | 9.1e-07 | 4.5e-04 | **REAL** |
| V1 vs bartowski | wikitext | −0.19 % | −2.73 (144) | 0.0071 | 0.096 | weak |
| V0 vs V1 | wikitext | −0.46 % | 3.85 (144) | 1.8e-04 | 1.2e-04 | **REAL** |
| V0 vs V1 | pile | +0.09 % | −0.95 (160) | 0.34 | 0.53 | not separable |
| V0 vs bartowski | pile | +0.07 % | 0.68 (160) | 0.50 | 0.34 | not separable |
| V1 vs V2 | pile | −0.40 % | −8.87 (160) | 1.4e-15 | 1.2e-11 | **REAL** |

## Finding 1 — the imatrix-free quant beats the community imatrix quant

**V0, with no importance matrix at all, beats the incumbent by 0.64 % perplexity
at 288 bytes smaller** (p = 9.1e-07, and both the t-test and the sign test
agree). On held-out pile the two are statistically indistinguishable.

So at equal size, the imatrix quant is *worse* out-of-domain and *not better*
in-domain. That is a direct challenge to the near-universal assumption that
imatrix quants dominate plain ones.

## Finding 2 — the imatrix is the cause, and it does not transfer

Isolating the matrix itself (V0 vs V1, identical except calibration):

- wikitext (out of domain): imatrix **hurts**, −0.46 %, p = 1.8e-04, both tests
- pile (in domain): imatrix helps **0.09 %**, p = 0.34 — not separable from noise

An importance matrix reweights quantization error toward the calibration
distribution. That is a *transfer* bet: it pays only if deployment text
resembles calibration text. Here the bet loses measurably out-of-domain and
fails to pay in-domain.

Caveat worth stating precisely: this shows the pile-calibrated matrix does not
transfer to wikitext. bartowski's calibration corpus is not public, so the
finding is about *these* checkpoints, not a general proof about all imatrix
recipes.

## Finding 3 — bit allocation works, but is not free

V2 (extra bits on `attn_v` and `ffn_down`) is the strongest in-domain result,
beating V1 by 0.40 % with overwhelming significance (p = 1.4e-15).

But it costs 5.1 % more bytes, and on wikitext it lands at **9.2275 vs V0's
9.2272** — a dead heat with a model 128 MB smaller. **On the wikitext Pareto
frontier, V0 dominates V2 outright.** Any claim for V2 has to be made on the
size/quality frontier, never head-to-head.

## Honest assessment

Round 1 produced a real, well-powered win: **V0 beats the incumbent by 0.64 %
perplexity at equal size**, and the accompanying negative result about imatrix
transfer is arguably the more interesting contribution.

But the effects are small — every lever here moves perplexity by under 1 % — and
**the GGUF/imatrix branch is close to exhausted**. Calibration composition and
per-tensor bit allocation are the only knobs `llama-quantize` exposes, and both
have now been measured.

Two things must happen before any of this can claim to "beat the field":

1. **Downstream evaluation.** A 0.64 % perplexity win may correspond to zero
   task-accuracy difference. Perplexity is the floor metric, and no conclusion
   about model quality is safe without GSM8K/MMLU.
2. **The larger algorithmic levers** — rotation (QuaRot / SpinQuant) and
   data-aware GPTQ/AWQ — which typically move 4-bit quality by several percent
   rather than fractions of one. These require the safetensors checkpoint
   (`Qwen/Qwen3-4B-Instruct-2507`) and `llm-compressor` in `.venv-compress`.
