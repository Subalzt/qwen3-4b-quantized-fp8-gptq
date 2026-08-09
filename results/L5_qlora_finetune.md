# L5 — QLoRA domain fine-tune of Qwen3-4B (honest comparison)

Goal: turn the anchor model into a coding/security domain expert via QLoRA, and
measure — honestly — whether it worked.

## Setup

- Base: `Qwen/Qwen3-4B-Instruct-2507`, NF4 double-quant, LoRA r=32/α=64 on all 7
  projections, seqlen 1536, 1 epoch, lr 2e-4, effective batch 16.
- Ran entirely on the 8 GB card (5.4 GiB peak). bitsandbytes NF4 verified on
  sm_120 first.
- Two data mixes:
  - **v1** — 17,197 examples: code, linux, networking, security, thin iot/wifi
  - **v2** — 22,239 examples: adds a VM/IaC domain and code-heavy security
    (nmap, priv-esc, Sigma rules, cloud PoCs, terraform security)
- Training was healthy on its own terms: loss 1.52 → 0.72, eval ≈ train (no
  overfit), token accuracy 79%.

Data build was mostly a filtering problem — 61k–93k junk rows and 150k–260k
duplicates were removed, and several "obvious" sources were dropped for cause
(a 287k-row pentest set that was CVE dumps with blank `** RESERVED **`
placeholders; an "Arduino" set that was pre-tokenized pretraining data; a
non-commercial-licensed code set). WiFi (116) and IoT (581) stayed thin because
no instruction datasets for them exist on the Hub — reported, not padded.

## Two held-out evaluations

Both use paired statistics (McNemar), because at n≈200 an unpaired accuracy gap
hides inside overlapping error bars — the same lesson as the perplexity and
quantization rounds.

### MCQ — security knowledge (preemware/pentesting-eval, 241 Qs, never trained on)

| model | accuracy | |
|---|---|---|
| base | 84.65% | 204/241 |
| v1 | 86.31% | 208/241 |
| v2 | 86.31% | 208/241 |

base vs v2: +1.66 pp, **p = 0.45** — not significant. v1 vs v2: identical,
p = 1.0. **Fine-tuning added no measurable security knowledge.** Unsurprising:
the base already scored 84.6%, and QLoRA on instruction data adjusts behaviour,
it does not inject facts.

### HumanEval — code generation (164 problems, executable pass@1)

| model | pass@1 | |
|---|---|---|
| **base** | **87.80%** | 144/164 |
| v1 | 69.51% | 114/164 |
| v2 | 70.73% | 116/164 |

base vs v2: **p = 1.9e-06, v2 is worse.** Discordant pairs: base-right/v2-wrong
= 32, v2-right/base-wrong = 4. base vs v1: p = 1.5e-08. An **~18-point drop in
coding ability.**

## Finding — catastrophic forgetting, cleanly measured

Combining the two evals: the fine-tune **cost ~18 points of general coding
ability and bought zero measurable security knowledge.** For real use the
untuned base is the better model.

The failure is the classic QLoRA one — too much drift: lr 2e-4, rank 32 on all
projections, full-sequence loss (the model learned to imitate instruction
phrasing, not just to answer), no general-data replay, and no mid-training
guardrail watching for forgetting. Training loss looked great throughout;
loss is not capability.

## Why this matters

Three cheap signals all said the fine-tune was fine — training loss fell
smoothly, eval loss tracked train, and 3 smoke-test prompts produced clean,
correct domain code. Only an objective, executable eval at n=164 with paired
statistics revealed an 18-point regression. Most fine-tuning workflows stop at
the first three signals and ship a worse model believing it is better.

This is the project's core thesis restated at the fine-tuning layer: **the
metric determines the conclusion.** Perplexity said quantization was "free";
MCQ said the fine-tune was "a tie"; only the right generation eval told the
truth.

## Artifacts

- `~/models/lora-coder`, `~/models/lora-coder-v2` — LoRA adapters
- `~/models/qwen3-4b-coder-v2-Q4_K_M.gguf` — merged/quantized, runs at ~112 t/s
- `results/mcq_{base,v1,v2}.json`, `results/he_{base,v1,v2}.json` — per-item
  records, paired-analysis ready

## Not done (deliberately)

A recovery run (v3: lr 5e-5, completion-only loss, rank 16, ~25% general-code
replay, HumanEval as a mid-training guardrail) would likely recover most of the
lost coding ability while retaining domain behaviour. Not run — the negative
result is the deliverable, and v3 is uncertain additional compute.
