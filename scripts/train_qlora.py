#!/usr/bin/env python3
"""QLoRA fine-tune of Qwen3-4B-Instruct on the domain mixture.

Sized for an 8 GiB card. Budget at r=32, seqlen 1536, batch 1:

    NF4 base weights          ~2.3 GiB
    LoRA params + grads       ~0.3 GiB   (~66M trainable)
    paged AdamW 8-bit state   ~0.15 GiB
    activations (checkpointed) ~1-2 GiB
    ------------------------------------
    total                     ~5 GiB, against ~7.6 GiB usable

Notes on choices:
  * NF4 double-quant base. bitsandbytes NF4 was verified on sm_120 at
    rel_err 0.0922 (see scripts/check_env.py) -- not assumed to work.
  * LoRA on ALL seven projections, not just q/v. Attention-only adapters
    underfit badly when the goal is domain knowledge rather than style.
  * packing is OFF. It requires a Flash-Attention variant to mask across
    sample boundaries, and sm_120 has no flash-attn build: the prebuilt
    kernels-community/flash-attn2 has no variant for this system, and
    vllm-flash-attn3 loads and then dies with "no kernel image is available
    for execution on the device". With sdpa + packing, samples in a packed
    sequence attend to each other and silently corrupt training. At batch
    size 1 packing also saves nothing (no padding to eliminate), so turning
    it off costs no throughput.
    (TRL 1.9's SFTConfig has no group_by_length, so throughput is recovered
    by raising the micro-batch instead.)
  * paged optimizer so a long-sequence spike pages to host RAM instead of
    killing a multi-hour run.

    python scripts/train_qlora.py --data ~/data/sft_mix --out ~/models/lora-coder
"""

from __future__ import annotations

import argparse
import json
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=1536)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="disable gradient checkpointing (~30%% faster, more VRAM)")
    ap.add_argument("--dry-run", action="store_true",
                    help="load everything, run 3 steps, report VRAM, exit")
    a = ap.parse_args()

    import torch
    from datasets import load_from_disk
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from trl import SFTConfig, SFTTrainer

    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_from_disk(os.path.expanduser(a.data))
    if a.max_samples:
        ds = ds.select(range(min(a.max_samples, len(ds))))

    def to_text(batch):
        texts = []
        for ins, resp in zip(batch["instruction"], batch["response"]):
            msgs = [{"role": "user", "content": ins},
                    {"role": "assistant", "content": resp}]
            texts.append(tok.apply_chat_template(msgs, tokenize=False))
        return {"text": texts}

    ds = ds.map(to_text, batched=True, remove_columns=[
        c for c in ds.column_names if c not in ("domain",)])
    split = ds.train_test_split(test_size=a.eval_frac, seed=1234)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"  train {len(train_ds)} / eval {len(eval_ds)}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        a.model, quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa")
    model.config.use_cache = False
    grad_ckpt = not a.no_grad_ckpt
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=grad_ckpt)

    peft_cfg = LoraConfig(
        r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M "
          f"({100*trainable/total:.2f}%)")

    cfg = SFTConfig(
        output_dir=out,
        per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.accum,
        num_train_epochs=3 if a.dry_run else a.epochs,
        max_steps=3 if a.dry_run else -1,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=3,
        eval_strategy="no" if a.dry_run else "steps",
        eval_steps=250,
        bf16=True,
        max_length=a.seqlen,
        packing=False,           # see module docstring: no flash-attn on sm_120
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to=[],
        dataset_text_field="text",
        seed=1234,
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds,
                         eval_dataset=None if a.dry_run else eval_ds,
                         processing_class=tok)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"\n  wall {dt/60:.1f} min | torch peak {peak:.2f} GiB")
    print("  (torch peak understates true VRAM; NVML is the number of record)")

    if a.dry_run:
        print("  DRY RUN OK -- it fits. Re-run without --dry-run.")
        return

    trainer.save_model(out)
    tok.save_pretrained(out)
    with open(os.path.join(out, "qbench_run.json"), "w") as fh:
        json.dump({"model": a.model, "rank": a.rank, "alpha": a.alpha,
                   "seqlen": a.seqlen, "epochs": a.epochs, "lr": a.lr,
                   "effective_batch": a.batch * a.accum,
                   "train_rows": len(train_ds), "wall_min": dt / 60,
                   "torch_peak_gib": peak}, fh, indent=1)
    print(f"  adapter -> {out}")


if __name__ == "__main__":
    main()
