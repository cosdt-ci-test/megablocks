# dMoE language model example (Ascend NPU)

This example trains and runs inference with a small GPT-style transformer whose
FFN blocks are MegaBlocks `dMoE` (dropless-MoE) layers, end to end on a single
Ascend NPU. It exercises the NPU-supported megablocks pathway: grouped-GEMM
experts (`torch_npu.npu_grouped_matmul`), triton-ascend permute kernels, and
the load-balancing auxiliary loss, with bf16/fp16 parameters.

The model is intentionally small (learned embeddings, manual causal attention,
`dMoE` blocks) so the scripts stay readable and no full training framework is
required. The default configuration (~209M params, untied embeddings) is: 6 layers,
hidden 512, 8 experts with top-2 routing, GLU MLPs, sequence length 512.

## Environment

- CANN toolkit (source `/usr/local/Ascend/ascend-toolkit/set_env.sh`), a 910B-class NPU.
- `torch >= 2.7.0` with the matching `torch_npu`, plus `triton-ascend`.
- MegaBlocks installed from this repository: `pip install -e .`
- Data dependencies: `pip install datasets transformers` (and `modelscope` with
  `pyarrow` only for the `--dataset-source modelscope` path).

## 1. Prepare the data

```bash
# Default: Hugging Face (wikitext-2-raw-v1 + GPT-2 tokenizer).
export HF_ENDPOINT=https://hf-mirror.com  # mirror, when huggingface.co is unreachable
python data.py prepare --data-dir data --dataset-source hf

# Alternative: load the text through the ModelScope SDK.
python data.py prepare --data-dir data --dataset-source modelscope
```

This writes flat uint16 token streams `data/tokens_train.bin` and
`data/tokens_val.bin` plus the tokenizer under `data/tokenizer/`. Any other
uint16 token file can be used directly via `--data-path` (train) and
`--eval-data-path` (validation).

## 2. Train

```bash
python train.py --steps 1000 --save checkpoints
```

`train.py` automatically prepares the dataset when the token files are
missing. Useful flags (see `--help` for all): `--seq-length`, `--micro-batch-size`,
`--num-layers`, `--num-experts`, `--top-k`, `--mlp-type {mlp,glu}`, `--dtype {bf16,fp16}`,
`--lr`, `--warmup-steps`, `--load` (resume), `--eval-interval`.

Progress is logged as `step=... lm_loss=... aux_loss=... lr=... tokens_per_sec=...`,
with periodic `val_loss=...` lines. The first `device: npu:0 (torch_npu ...)`
line names the device actually used.

A quick smoke run on random tokens, without downloading a dataset:

```bash
python -c "import numpy as np; np.random.default_rng(0).integers(0,50257,size=200000,dtype=np.uint16).tofile('tokens.bin')"
python train.py --data-path tokens.bin --steps 20 --num-layers 2 --log-interval 5
```

## 3. Generate

```bash
python generate.py --checkpoint checkpoints/checkpoint.pt --prompt 'Once upon a time'
```

Sampling flags: `--max-new-tokens`, `--temperature`, `--top-k`, `--top-p`,
`--greedy`. Without `--prompt` the first validation tokens are used as the
context.

## Troubleshooting

On some CANN 9.1.0 + torch_npu 2.7.x installs, the first triton-ascend kernel
launch fails while compiling triton's `npu_utils` shim with
``'aclOpExecutor' was not declared in this scope``. Work around it by compiling
through a wrapper that force-includes a forward typedef, before running any of
the scripts:

```bash
mkdir -p .triton_cc
echo "typedef struct aclOpExecutor aclOpExecutor;" > .triton_cc/stub.h
printf '#!/bin/sh\nexec g++ -include %s/.triton_cc/stub.h "$@"\n' "$PWD" > .triton_cc/gxx.sh
chmod +x .triton_cc/gxx.sh
export CC="$PWD/.triton_cc/gxx.sh"
```

Newer CANN releases ship the declaration themselves and do not need this.

## Expectations and limits

- The goal is to demonstrate the supported NPU dMoE pathway end to end (training
  loss goes down, generation produces text), not to produce a good language
  model: wikitext-2 is tiny and the default schedule is short.
- Standard (capacity-based) MoE, `mlp_impl='sparse'`, and multi-rank expert
  parallelism are not supported on NPU yet; the trainer refuses `torchrun`.
- Half precision (bf16/fp16) is required.
- The first training step is slow: triton-ascend compiles the permute kernels
  once and caches them under `.triton_cache/`.
- The same scripts also run on a CUDA GPU (install `megablocks[gg]` for the
  grouped-GEMM dependency), but the CUDA path is not what this example verifies.
