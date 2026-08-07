# Run MMPFN × VT-Bench on an existing Linux server environment

This package contains the MMPFN source plus the minimal VT-Bench adapter for
the **Adoption** and **Breast Cancer** formal evaluation datasets. It excludes
datasets, model checkpoints, embeddings, package caches, and local virtual
environments.

## 1. Copy and unpack

```bash
unzip mmpfn_vtbench_server_package.zip
cd MultiModalPFN-main
```

Do not create the repository's original Conda environment by default; its lock
file was made for a different machine. Activate the server environment you
already use, then inspect it:

```bash
python verify_server_env.py
```

The experiment should use a CUDA-enabled PyTorch installation. If a module is
reported missing, install only that missing dependency in the existing
environment. The repository's `requirements.txt` plus `pandas`, `optuna`, and
`transformers` is the reference list; do not blindly reinstall PyTorch if the
server already has a compatible CUDA build.

## 2. Install this checkout into that environment

```bash
python -m pip install -e . --no-deps
```

## 3. Required external assets

These assets are intentionally not in the archive:

1. The original MMPFN classifier checkpoint at
   `mmpfn/parameters/tabpfn-v2-classifier.ckpt`.
2. A DINOv2 ViT-B/14 checkpoint; provide its absolute path to the command.
3. One VT-Bench exported feature directory for each dataset. The exact files
   are listed in `VT_BENCH_ADAPTER.md`.

Keep large data and checkpoints outside this repository. The adapter writes
generated embeddings and fine-tuned checkpoints under its `--output-dir`.

## 4. Run Full MMPFN with the fixed VT-Bench split

```bash
python -m mmpfn.run_vtbench \
  --dataset adoption \
  --data-root /path/to/vtbench/adoption/features \
  --dino-checkpoint /path/to/dinov2_vitb14_pretrain.pth \
  --output-dir /path/to/experiment_outputs \
  --device cuda \
  --seed 42
```

Use `--dataset breast` and the Breast Cancer feature directory for the second
dataset. The command preserves VT-Bench train/validation/test partitions and
stores `metrics.json` for that seed.

## Scope of this package

This bridge runs **Full MMPFN** only. Image-only, tabular-only, and MCR
ablations should be added after the full model has completed successfully, so
their definitions remain aligned with VT-Bench.
