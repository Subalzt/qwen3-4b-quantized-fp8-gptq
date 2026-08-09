# quant-bench — reproduction entry points.
# Environments are three uv venvs (see README "Reproduction"); most targets use
# the main .venv, vLLM/compress targets use their own.

PY      := .venv/bin/python
PYVLLM  := .venv-vllm/bin/python
PYCOMP  := .venv-compress/bin/python
LLAMA   := $(HOME)/llama.cpp/build/bin
WIKI    := $(HOME)/data/wikitext-2-raw/wiki.test.raw

.PHONY: help env grid ppl-paired mcnemar l3-sweep clean-logs

help:
	@echo "targets:"
	@echo "  env         verify every backend runs REAL kernels on this GPU (sm_120)"
	@echo "  grid        run the (format x runtime) grid -> results/grid.csv"
	@echo "  l3-sweep    concurrency curve, vLLM vs llama.cpp"
	@echo "  ppl-paired  paired per-chunk perplexity test (A.log B.log)"
	@echo "  mcnemar     paired accuracy test for two eval json files"

# Capability matrix — catches 'no kernel image' / silent-corruption BEFORE a run.
env:
	$(PY) scripts/check_env.py --json results/env_main.json

# The uniform grid: one CSV, every cell comparable.
grid:
	$(PY) -m qbench.runner configs/grid.yaml --out results/grid.csv

# L3 throughput curve (starts/stops both servers).
l3-sweep:
	bash scripts/run_l3_sweep.sh

# Paired stats used throughout (why overlapping error bars are the wrong test).
ppl-paired:
	$(PY) scripts/ppl_paired.py $(A) $(B)

mcnemar:
	$(PY) scripts/mcnemar.py $(A) $(B)

clean-logs:
	rm -f results/*.log
