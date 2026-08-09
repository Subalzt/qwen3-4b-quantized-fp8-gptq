# L4 — attention: fp8 KV-cache quantization + long context

Once a 4-bit model is loaded on an 8 GB card, the **KV cache — not the weights —
is what caps context length and concurrency.** L4 measures what vLLM's fp8 KV
cache buys, using the GPTQ-W4A16 checkpoint (2.48 GiB weights) so the KV pool
gets maximum room.

FlashAttention never built on sm_120 (see the env notes), so this runs on
vLLM's default attention backend, which is where the KV-cache dtype lever lives.

## Setup

vLLM 0.26, `gpu_memory_utilization=0.85`, `max_model_len=8192` (low enough that
fp16 also starts). The KV **pool** size is fixed by the memory budget, not by
max_model_len, so `num_gpu_blocks` is the honest capacity readout. Requires
`VLLM_WSL2_ENABLE_PIN_MEMORY=1` (see L3 / env notes) or vLLM will not start
under WSL2.

## Result

| kv_cache_dtype | GPU blocks | KV tokens | vs fp16 |
|---|---:|---:|---:|
| auto (fp16) | 1623 | **25,968** | 1.00× |
| **fp8 (e4m3)** | 3246 | **51,936** | **2.00×** |

`3246 = 2 × 1623` exactly — fp8 halves per-token KV to the block, doubling
capacity with no rounding slop.

### Quality: fp8 KV does not degrade output

Same greedy prompt ("set up and harden a reverse SSH tunnel from behind NAT"),
both dtypes:

- fp16: *"# Step-by-Step Guide: Setting Up a Reverse SSH Tunnel from a Home
  Network to a Public Jump Host … useful for developers who need to secu…"*
- fp8:  *"# Step-by-Step Guide: Setting Up a Reverse SSH Tunnel from a Home
  Machine to a Public Jump Host … allows secure access to internal services…"*

Near-identical and equally coherent. Unlike **weight** quantization (L2 GPTQ/AWQ
cost +3–6% PPL), **KV** quantization to fp8 is effectively free here — the KV
cache stores activations that are far more tolerant of low precision than the
weights.

## Finding

On an 8 GB card, **fp8 KV cache is a free 2× on the binding long-context /
concurrency constraint.** Concretely:

- Long context: fp16 tops out at ~26k tokens and *cannot* serve a 32k request
  (vLLM refuses with "estimated maximum model length is 25968"); fp8 reaches
  ~52k.
- Concurrency: at any fixed context, fp8 fits twice as many simultaneous
  sequences — compounding the L3 throughput story, since vLLM's advantage grows
  with batch size.

This is the cleanest cost/benefit in the whole study: exactly 2× the scarce
resource, no measurable quality loss. On memory-bound consumer hardware it
should be the default, not an option.

## Caveat

Measured on one model with greedy decoding and a coherence check, not a
long-context accuracy benchmark. fp8 KV can matter more on tasks that depend on
precise long-range retrieval (e.g. needle-in-a-haystack); the "free" claim here
is for VRAM capacity and near-term generation quality, not proven for
adversarial long-context recall.
