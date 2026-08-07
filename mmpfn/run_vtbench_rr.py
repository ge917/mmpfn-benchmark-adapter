"""Run fixed-split MMPFN regression on VT-Bench Respiratory Rate.

By default, fine-tuning uses the original VT-Bench respiratory-rate labels.
``--target-standardization train_zscore`` is available only as an explicit
numerical-stability fallback; metrics are always reported in native units.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mmpfn.datasets.vtbench import VTBenchSplitDataset
from mmpfn.models.mmpfn import MMPFNRegressor
from mmpfn.models.mmpfn.constants import ModelInterfaceConfig
from mmpfn.models.mmpfn.preprocessing import PreprocessorConfig
from mmpfn.scripts_finetune_mm.constant_utils import SupportedDevice, SupportedValidationMetric, TaskType
from mmpfn.vtbench_runner_utils import fixed_subset_indices, predict_in_chunks, subset_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "image_only", "tabular_only"), default="full")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--max-train-context", type=int, default=6000)
    parser.add_argument("--max-val-context", type=int, default=2000)
    parser.add_argument("--prediction-batch-size", type=int, default=512)
    parser.add_argument("--validation-chunk-size", type=int, default=0)
    parser.add_argument("--save-all-checkpoints", action="store_true")
    parser.add_argument(
        "--target-standardization",
        choices=("none", "train_zscore"),
        default="none",
        help="Use 'none' to match the original VT-Bench label scale (default).",
    )
    parser.add_argument("--finetune-steps", type=int, default=100)
    parser.add_argument("--time-limit", type=int, default=900)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Respiratory Rate evaluation requires CUDA for DINOv2 embedding extraction.")
    os.chdir(Path(__file__).resolve().parent)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    splits = {
        name: VTBenchSplitDataset(args.data_root, "rr", name, "uint8")
        for name in ("train", "val", "test")
    }
    root = args.output_dir.resolve() / "rr"
    output = root / args.mode / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    use_tabular, use_images = args.mode != "image_only", args.mode != "tabular_only"
    embeddings = (
        {
            name: split.get_embeddings(
                args.dino_checkpoint,
                root / f"seed_{args.seed}" / "embeddings" / f"{name}_dinov2_vitb14.pt",
                args.embedding_batch_size,
                "cuda",
            )
            for name, split in splits.items()
        }
        if use_images
        else {name: None for name in splits}
    )

    categorical = splits["train"].categorical_features if use_tabular else None
    train_indices = fixed_subset_indices(
        splits["train"].y, args.max_train_context, seed=args.seed, task="regression"
    )
    val_indices = fixed_subset_indices(
        splits["val"].y, args.max_val_context, seed=args.seed + 1, task="regression"
    )
    x_train = splits["train"].x[train_indices]
    y_train = splits["train"].y[train_indices]
    x_val = splits["val"].x[val_indices]
    y_val = splits["val"].y[val_indices]
    image_train = subset_embeddings(embeddings["train"], train_indices)
    image_val = subset_embeddings(embeddings["val"], val_indices)
    target_mean = float(np.mean(y_train))
    target_std = float(np.std(y_train))
    if not np.isfinite(target_std) or target_std == 0:
        raise ValueError("Respiratory-rate train targets have zero or invalid variance.")
    if args.target_standardization == "train_zscore":
        y_train_tuned = pd.Series((y_train - target_mean) / target_std)
        y_val_tuned = pd.Series((y_val - target_mean) / target_std)
    else:
        y_train_tuned = pd.Series(y_train)
        y_val_tuned = pd.Series(y_val)

    from mmpfn.scripts_finetune_mm.finetune_mmpfn_main import fine_tune_mmpfn

    checkpoint = output / "finetuned_mmpfn.ckpt"
    fine_tune_mmpfn(
        mixer_type="MGM+CAP",
        mgm_heads=8,
        cap_heads=8,
        features_per_group=2,
        save_path_to_fine_tuned_model=checkpoint,
        time_limit=args.time_limit,
        finetuning_config={"learning_rate": 1e-5, "batch_size": 1, "max_steps": args.finetune_steps},
        validation_metric=SupportedValidationMetric.RMSE,
        categorical_features_index=categorical,
        task_type=TaskType.REGRESSION,
        device=SupportedDevice.GPU,
        X_train=pd.DataFrame(x_train) if use_tabular else None,
        image_train=image_train if use_images else None,
        y_train=y_train_tuned,
        X_val=pd.DataFrame(x_val) if use_tabular else None,
        image_val=image_val if use_images else None,
        y_val=y_val_tuned,
        validation_chunk_size=args.validation_chunk_size or None,
        save_all_checkpoints_dir=output / "all_checkpoints" if args.save_all_checkpoints else None,
        random_seed=args.seed,
        logger_level=20,
        freeze_input=True,
    )

    config = ModelInterfaceConfig(
        FINGERPRINT_FEATURE=False,
        PREPROCESS_TRANSFORMS=[PreprocessorConfig(name="none")],
    )
    model = MMPFNRegressor(
        model_path=checkpoint,
        inference_config=config,
        ignore_pretraining_limits=True,
        device="cuda",
        mixer_type="MGM+CAP",
        mgm_heads=8,
        cap_heads=8,
        features_per_group=2,
        categorical_features_indices=categorical,
        random_state=args.seed,
    ).fit(
        pd.DataFrame(x_train) if use_tabular else None,
        image_train if use_images else None,
        y_train,
    )
    prediction = predict_in_chunks(
        model,
        splits["test"].x if use_tabular else None,
        embeddings["test"] if use_images else None,
        batch_size=args.prediction_batch_size,
    )
    y_test = splits["test"].y.astype(np.float64)
    metrics = {
        "dataset": "rr",
        "mode": args.mode,
        "seed": args.seed,
        "mae": float(mean_absolute_error(y_test, prediction)),
        "rmse": float(mean_squared_error(y_test, prediction) ** 0.5),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "n_train_available": int(len(splits["train"].y)),
        "n_val_available": int(len(splits["val"].y)),
        "max_train_context": args.max_train_context,
        "max_val_context": args.max_val_context,
        "validation_chunk_size": args.validation_chunk_size or None,
        "all_checkpoints_dir": str(output / "all_checkpoints") if args.save_all_checkpoints else None,
        "categorical_features": categorical,
        "target_train_mean": target_mean,
        "target_train_std": target_std,
        "target_standardization": args.target_standardization,
        "checkpoint": str(checkpoint),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
