# Environment

Every number in `results/` was produced on this exact stack. Re-run
`python scripts/check_env.py --json results/env.json` before a session and
commit the JSON alongside the results — a missing grid cell should always be
traceable to a recorded probe failure, not to a vague "it didn't work".

## Hardware

| | |
|---|---|
| Laptop | Lenovo Legion 7i |
| CPU | Intel Core Ultra 9 275HX (24 threads; 20 exposed to WSL) |
| GPU | NVIDIA RTX 5070 Laptop, **8151 MiB**, Blackwell **sm_120** |
| Host RAM | 32 GB (24 GB allocated to WSL) |
| Driver | 610.47 (Windows KMD) / 610.43.02 (WSL UMD), CUDA UMD 13.3 |

The 8 GB card is the binding constraint throughout. A BF16 4B model is ~8 GB of
weights alone, so the BF16 baseline is expected to spill to CPU — that spill is
a *result*, not a bug, and the harness records it rather than hiding it.

## Why WSL2 and not native Windows

- `vllm` has no Windows wheels. Ever. L3 is impossible on native Windows.
- `bitsandbytes`, `flash-attn`, `flashinfer` all assume Linux for CUDA builds.
- `torch.compile` / Triton support on Windows lags badly.

LM Studio on the Windows side is still fine for the quick L1 sanity loop; it
just isn't where the reproducible numbers come from.

## WSL configuration

`C:\Users\periy\.wslconfig`:

```ini
[wsl2]
memory=24GB
processors=20
swap=16GB

[experimental]
autoMemoryReclaim=gradual
sparseVhd=true
```

`sparseVhd` matters: GGUF + safetensors copies of a 4B model in several formats
run to tens of GB, and without it the ext4 VHDX never gives the space back.

`/etc/wsl.conf` sets `[interop] appendWindowsPath = false`. This is not
cosmetic — the inherited Windows `PATH` contains directory names with spaces and
parentheses that break `bash -c` quoting and slow every shell start.

## The sm_120 problem

Blackwell consumer silicon is new enough that the usual failure is not an
install error. A wheel installs, imports, exposes the right API, and then dies
at the first kernel launch with:

```
CUDA error: no kernel image is available for execution on the device
```

Worse, some paths *run* and return garbage. `scripts/check_env.py` therefore
launches a real kernel for every library and, where a cheap reference exists,
compares numerics against it. Anything with relative error > 0.5 is reported as
`CORRUPT` rather than `PASS`.

**Rule: never record a benchmark row from a path that check_env flags.** Record
it as a negative result in the paper instead — "bitsandbytes INT8 produces
incorrect output on sm_120" is a genuinely useful finding.

## Software stack

Chosen versions and the reason for each:

| Component | Version | Why |
|---|---|---|
| Ubuntu | 26.04 LTS (WSL2, kernel 6.18.33.1) | preinstalled distro |
| Python | 3.12.13 (via `uv`) | distro ships 3.14, which most ML wheels don't build for yet |
| PyTorch | 2.13.0+**cu129** | newest torch that has a CUDA 12.x build; ships `sm_120` |
| CUDA toolkit | 12.9 | matches torch's CUDA **major and minor** so source builds and FlashInfer's JIT don't hit a version-mismatch guard |
| Host compiler | **gcc-14** | nvcc rejects the distro default gcc-15 |

CUDA 13.x was available (up to 13.3) and the driver supports it, but CUDA 12.x
still has far broader third-party wheel coverage. Matching the toolkit to
torch's `cu129` is what keeps `flashinfer` JIT and any source build working.

### Virtual environments

Three, deliberately. Several libraries pin torch hard, and one shared env would
mean the last install silently decides everyone's torch build.

| Path | Contents | Purpose |
|---|---|---|
| `.venv` | torch 2.13+cu129, transformers 5.x, torchao, bitsandbytes, peft, trl, gptqmodel, compressed-tensors | main benchmark env (L1, L1.5, L2, L4, L5) |
| `.venv-vllm` | vllm + its pinned torch, flashinfer | L3 runtime comparison |
| `.venv-compress` | llm-compressor | **authoring** quantized checkpoints, not benchmarking them |

`llm-compressor` was isolated because it resolves torch down to `2.12.0` — the
generic build, with no cu129 wheel and therefore no guaranteed sm_120 kernels.
Letting it into `.venv` would have silently degraded every other measurement.

Because each grid cell runs in a fresh subprocess anyway (so VRAM is genuinely
released between cells), pointing different cells at different interpreters
costs nothing.

## Environment variables

Set in `/root/.bashrc`:

```sh
export CUDA_HOME=/usr/local/cuda            # -> /usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export CUDAHOSTCXX=/usr/bin/g++-14
export CC=/usr/bin/gcc-14
export CXX=/usr/bin/g++-14
export TORCH_CUDA_ARCH_LIST=12.0            # build for sm_120 only
export HF_HOME=/root/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
```

`/usr/lib/wsl/lib` on `LD_LIBRARY_PATH` is required: that is where WSL puts the
passthrough `libcuda.so.1`. Leave it out and CUDA init fails inside WSL even
though `nvidia-smi` works.

## Accessing the repo from Windows

The repo lives on the WSL ext4 filesystem (`/root/quant-bench`) — not under
`/mnt/c` — because model loading and git operations across the 9p mount are
several times slower. From Windows it is reachable at:

```
\\wsl.localhost\Ubuntu\root\quant-bench
```

For editing, VS Code with the WSL remote extension is the path of least
resistance.
