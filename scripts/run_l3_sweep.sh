#!/usr/bin/env bash
# L3 concurrency sweep: vLLM (PagedAttention) vs llama.cpp server.
# Both serve a 4-bit Qwen3-4B (vLLM: GPTQ-W4A16; llama.cpp: Q4_K_M GGUF).
# The weight format differs by runtime, but the CURVE SHAPE -- does aggregate
# throughput keep scaling with concurrency? -- is the runtime property measured.
#
# IMPORTANT: pkill patterns here are specific (build/bin/llama-server, bin/vllm)
# and this script is run as a FILE, so its own command line does not contain
# those substrings -- otherwise `pkill -f` would kill the script itself.
set -uo pipefail
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=$HOME/.cache/huggingface
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_WSL2_ENABLE_PIN_MEMORY=1   # WSL2 kernel supports pinned mem; vLLM needs this
export TOKENIZERS_PARALLELISM=false
cd "$HOME/quant-bench"
LEVELS=${LEVELS:-1,4,16,32}
GGUF=$HOME/models/gguf/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf
GPTQ=$HOME/models/hf/R2-GPTQ

echo "########## llama.cpp server sweep ##########"
for p in $(pgrep -f "build/bin/llama-server"); do kill -9 "$p" 2>/dev/null; done
sleep 2
$HOME/llama.cpp/build/bin/llama-server -m "$GGUF" \
  -ngl 99 -c 16384 -np 32 --host 127.0.0.1 --port 8101 --no-webui \
  > /tmp/l3_llamacpp.log 2>&1 &
LPID=$!
for i in $(seq 1 60); do curl -s http://127.0.0.1:8101/health 2>/dev/null | grep -q '"ok"' && break; sleep 1; done
.venv/bin/python scripts/concurrency_client.py --port 8101 --label llamacpp --levels "$LEVELS"
kill -9 $LPID 2>/dev/null; sleep 3

echo
echo "########## vLLM server sweep ##########"
.venv-vllm/bin/vllm serve "$GPTQ" \
  --port 8100 --gpu-memory-utilization 0.85 --max-model-len 4096 \
  --max-num-seqs 64 --enforce-eager > /tmp/l3_vllm.log 2>&1 &
VPID=$!
for i in $(seq 1 360); do curl -s http://127.0.0.1:8100/health >/dev/null 2>&1 && break; sleep 1; done
.venv/bin/python scripts/concurrency_client.py --port 8100 --label vllm --levels "$LEVELS"
kill -9 $VPID 2>/dev/null; sleep 3

echo
echo "################## L3 THROUGHPUT CURVE ##################"
.venv/bin/python - <<'PY'
import json
def load(l):
    try: return {r["concurrency"]: r for r in json.load(open(f"results/concurrency_{l}.json"))["rows"]}
    except Exception: return {}
lc, vl = load("llamacpp"), load("vllm")
print(f"  {'conc':>4} | {'llama.cpp agg':>13} {'per-req':>8} | {'vLLM agg':>10} {'per-req':>8}")
print("  " + "-"*54)
for c in sorted(set(lc) | set(vl)):
    l, v = lc.get(c), vl.get(c)
    ls = f"{l['aggregate_tok_s']:>13} {l['per_req_tok_s']:>8}" if l else " "*22
    vs = f"{v['aggregate_tok_s']:>10} {v['per_req_tok_s']:>8}" if v else " "*19
    print(f"  {c:>4} | {ls} | {vs}")
PY
echo "=== L3_SWEEP_DONE ==="
