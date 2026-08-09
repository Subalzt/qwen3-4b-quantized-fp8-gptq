"""quant-bench: a controlled (format x runtime) LLM quantization benchmark.

Public surface:
    schema.Row / schema.append_row  -- the uniform result row and CSV writer
    vram.VramSampler                -- out-of-process NVML VRAM sampling
    backends.LlamaCppBackend / VllmBackend
    runner.main                     -- `python -m qbench.runner <grid.yaml>`
"""

__version__ = "0.1.0"
