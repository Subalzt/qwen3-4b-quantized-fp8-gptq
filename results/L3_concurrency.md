# L3 — runtime concurrency sweep: vLLM vs llama.cpp

The brief's prediction: "vLLM only wins at batch >1. Report a throughput CURVE."
This measures exactly that.

## Setup

- **vLLM 0.26** serving GPTQ-W4A16 (`enforce_eager`, gpu-mem-util 0.85,
  max-model-len 4096, max-num-seqs 64).
- **llama.cpp server** serving Q4_K_M GGUF (`-np 32 -c 16384`).
- Client (`scripts/concurrency_client.py`) fires N simultaneous
  `/v1/completions` requests (fixed prompt, 128 max tokens), measures aggregate
  decode throughput and per-request throughput. Both servers expose the same
  OpenAI API, so one client drives both. 3 warm-up requests at full size (vLLM
  pays a large one-time cost on its first full-size decode).
- Weight format differs by runtime (each runtime's native 4-bit). The property
  under test is the **curve shape**, which is a runtime characteristic.

Enabling vLLM on this box required `VLLM_WSL2_ENABLE_PIN_MEMORY=1` — vLLM
disables pinned memory on WSL by default, which makes it report UVA unavailable
and refuse to start the v1 engine. The kernel (6.18) supports pinned memory;
the flag re-enables it.

## Results — aggregate decode tokens/sec

| concurrency | llama.cpp | vLLM | winner |
|---:|---:|---:|---|
| 1 | **115.8** | 72.6 | llama.cpp (1.6×) |
| 4 | **320.6** | 276.6 | llama.cpp (1.16×) |
| 16 | 887.9 | **1010.6** | vLLM (1.14×) |
| 32 | 1189.3 | **1887.5** | vLLM (1.59×) |

## Results — per-request tokens/sec (the mechanism)

| concurrency | llama.cpp | vLLM |
|---:|---:|---:|
| 1 | 115.8 | 72.6 |
| 4 | 80.1 | 69.1 |
| 16 | 55.5 | 63.2 |
| 32 | 37.2 | **59.0** |

## Findings

1. **Crossover at concurrency ~8–16.** llama.cpp is the better single-stream
   engine on this hardware (115.8 vs 72.6 t/s at concurrency 1). vLLM overtakes
   it once enough requests are in flight to fill its continuous batch.

2. **vLLM holds per-request throughput; llama.cpp does not.** As concurrency
   goes 1→32, llama.cpp's per-request rate collapses 116→37 t/s (it time-slices
   a fixed compute budget), while vLLM stays near 60–72. This is the
   PagedAttention / continuous-batching benefit, made visible.

3. **A single-request benchmark would be actively misleading.** At concurrency
   1 llama.cpp looks 1.6× faster; at concurrency 32 vLLM is 1.6× faster. Which
   runtime "wins" is entirely a function of the workload's concurrency — the
   single number everyone quotes answers the wrong question.

## Deployment takeaway

- **Single user / interactive / edge**: llama.cpp — lower latency per stream,
  far simpler to run, smaller memory floor.
- **Serving many concurrent users**: vLLM — aggregate throughput scales and
  per-user latency stays flat under load.
