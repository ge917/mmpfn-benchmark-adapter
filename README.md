# MMPFN Benchmark Adapter

This repository provides a reproducible evaluation layer for **Multi-Modal Prior-data Fitted Networks (MMPFN)** on image--tabular benchmarks. It extends the upstream MMPFN source tree with a unified command-line workflow for VT-Bench and the image--tabular, `Text=0` subset of MulTaBench.

The original MMPFN source code and its license are retained in this repository. This repository's added code focuses on dataset preparation, experiment orchestration, and result summarisation; it does **not** redistribute datasets, model weights, credentials, or experiment artifacts.

## What is included

- A registry for 11 VT-Bench discriminative datasets and 11 MulTaBench image--tabular datasets with `Text=0`.
- One interface for `full`, `image_only`, and `tabular_only` MMPFN evaluation.
- MulTaBench preparation from its official Kaggle release format: `data.csv`, `metadata.json`, and `images/`.
- Resumable sequential runs, per-mode logs, per-run metrics, a performance summary, and an MCR-style negative-transfer summary.
- Existing adapters for locally exported VT-Bench data, including Breast Cancer, Pneumonia, Infarction, Respiratory Rate, Adoption, and Pawpularity.

## Repository layout

```text
mmpfn/benchmarking/             Dataset registry, loaders, and summary tables
mmpfn/prepare_multabench.py     MulTaBench download and preparation
mmpfn/run_benchmark_dataset.py  One dataset / one modality-mode evaluation
mmpfn/run_benchmark_suite.py    Resumable multi-dataset orchestrator
run_mmpfn_benchmarks.sh         Server-oriented convenience wrapper
BENCHMARK_SUITE.md              Detailed Chinese usage notes
```

## Installation

Create an environment that satisfies `environment.yaml` or `requirements.txt`, then install the package in editable mode:

```bash
python -m pip install -e . --no-deps
```

The benchmark runner requires the upstream MMPFN dependencies, a compatible PyTorch/CUDA environment, and a local DINOv2 checkpoint. MulTaBench automatic download additionally requires `kagglehub`.

## Quick start

List registered datasets:

```bash
python -m mmpfn.run_benchmark_suite --list-datasets
```

Run all three modality modes for a selected MulTaBench dataset:

```bash
bash run_mmpfn_benchmarks.sh \
  --datasets mt_mango_mass \
  --modes full image_only tabular_only
```

The wrapper is configured for the shared-server layout used during development. Before use elsewhere, set the paths through environment variables:

```bash
export MMPFN_WORK_ROOT=/path/to/workspace
export MMPFN_GPU=0
export MMPFN_PYTHON=/path/to/python
```

For a locally prepared VT-Bench export, call the Python entry point and provide the data root explicitly:

```bash
python -m mmpfn.run_benchmark_suite \
  --datasets vt_pneumonia \
  --dataset-root vt_pneumonia=/path/to/pneumonia/features \
  --modes full image_only tabular_only \
  --dino-checkpoint /path/to/dinov2_vitb14_pretrain.pth
```

See [BENCHMARK_SUITE.md](BENCHMARK_SUITE.md) for the full command reference, storage layout, and monitoring commands.

## Outputs and data policy

By default, data, downloaded images, caches, checkpoints, logs, and results are stored outside this source tree. Do not commit or upload any of the following:

- raw datasets, images, or patient-level data;
- DINOv2 / TabPFN checkpoints;
- run logs, generated metrics, or checkpoint files;
- Kaggle tokens, PhysioNet cookies, passwords, or other credentials.

The MulTaBench workflow currently supports deterministic fold `0` by default. For benchmark-level reporting, run multiple folds and report aggregate statistics rather than a single split.

## Upstream attribution and license

This project is derived from the MMPFN codebase. Please retain the existing source notices and [Apache-2.0 license](LICENSE) when reusing or redistributing the upstream code. TabPFN components may carry additional attribution requirements; see the upstream project materials and [TabPFN license information](https://priorlabs.ai/tabpfn-license/).
