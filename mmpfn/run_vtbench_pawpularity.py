"""Run fixed-split Full and unimodal MMPFN regression on Pawpularity."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full", "image_only", "tabular_only"), default="full")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--finetune-steps", type=int, default=100)
    parser.add_argument("--time-limit", type=int, default=900)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Pawpularity evaluation requires CUDA for DINOv2 embedding extraction.")
    os.chdir(Path(__file__).resolve().parent)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    splits = {name: VTBenchSplitDataset(args.data_root, "pawpularity", name, "zero_one") for name in ("train", "val", "test")}
    root = args.output_dir.resolve() / "pawpularity"
    output = root / args.mode / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    use_tabular, use_images = args.mode != "image_only", args.mode != "tabular_only"
    embeddings = (
        {name: split.get_embeddings(args.dino_checkpoint, root / f"seed_{args.seed}" / "embeddings" / f"{name}_dinov2_vitb14.pt", args.embedding_batch_size, "cuda") for name, split in splits.items()}
        if use_images else {name: None for name in splits}
    )
    categorical = splits["train"].categorical_features if use_tabular else None
    # The original fine-tuning loop's regression loss operates directly on the
    # supplied validation targets.  Pawpularity's native 1--100 scale can make
    # its mixed-precision gradients overflow, so normalize using train only.
    target_mean = float(np.mean(splits["train"].y))
    target_std = float(np.std(splits["train"].y))
    if not np.isfinite(target_std) or target_std == 0:
        raise ValueError("Pawpularity training targets have zero or invalid variance.")
    y_train_tuned = pd.Series((splits["train"].y - target_mean) / target_std)
    y_val_tuned = pd.Series((splits["val"].y - target_mean) / target_std)
    from mmpfn.scripts_finetune_mm.finetune_mmpfn_main import fine_tune_mmpfn
    checkpoint = output / "finetuned_mmpfn.ckpt"
    fine_tune_mmpfn(
        mixer_type="MGM+CAP", mgm_heads=8, cap_heads=8, features_per_group=2,
        save_path_to_fine_tuned_model=checkpoint, time_limit=args.time_limit,
        finetuning_config={"learning_rate": 1e-5, "batch_size": 1, "max_steps": args.finetune_steps},
        validation_metric=SupportedValidationMetric.RMSE, categorical_features_index=categorical,
        task_type=TaskType.REGRESSION, device=SupportedDevice.GPU,
        X_train=pd.DataFrame(splits["train"].x) if use_tabular else None,
        image_train=embeddings["train"] if use_images else None, y_train=y_train_tuned,
        X_val=pd.DataFrame(splits["val"].x) if use_tabular else None,
        image_val=embeddings["val"] if use_images else None, y_val=y_val_tuned,
        random_seed=args.seed, logger_level=20, freeze_input=True,
    )
    config = ModelInterfaceConfig(FINGERPRINT_FEATURE=False, PREPROCESS_TRANSFORMS=[PreprocessorConfig(name="none")])
    model = MMPFNRegressor(model_path=checkpoint, inference_config=config, ignore_pretraining_limits=True,
        device="cuda", mixer_type="MGM+CAP", mgm_heads=8, cap_heads=8, features_per_group=2,
        categorical_features_indices=categorical, random_state=args.seed).fit(
            pd.DataFrame(splits["train"].x) if use_tabular else None,
            embeddings["train"] if use_images else None, splits["train"].y)
    prediction = model.predict(pd.DataFrame(splits["test"].x) if use_tabular else None, embeddings["test"] if use_images else None)
    y = splits["test"].y.astype(np.float64)
    metrics = {"dataset": "pawpularity", "mode": args.mode, "seed": args.seed,
        "mae": float(mean_absolute_error(y, prediction)), "rmse": float(mean_squared_error(y, prediction) ** .5),
        "n_train": len(splits["train"].y), "n_val": len(splits["val"].y), "n_test": len(y),
        "categorical_features": categorical, "target_train_mean": target_mean,
        "target_train_std": target_std, "checkpoint": str(checkpoint)}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
