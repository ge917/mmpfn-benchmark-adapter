"""Run one registered image-tabular dataset/mode with MMPFN."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from mmpfn.benchmarking.data import load_benchmark_splits
from mmpfn.benchmarking.registry import get_dataset_spec
from mmpfn.models.mmpfn import MMPFNClassifier, MMPFNRegressor
from mmpfn.models.mmpfn.constants import ModelInterfaceConfig
from mmpfn.models.mmpfn.preprocessing import PreprocessorConfig
from mmpfn.scripts_finetune_mm.constant_utils import (
    SupportedDevice,
    SupportedValidationMetric,
    TaskType,
)
from mmpfn.vtbench_runner_utils import (
    fixed_subset_indices,
    predict_in_chunks,
    predict_proba_in_chunks,
    subset_embeddings,
)


PACKAGE_ROOT = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", choices=("full", "image_only", "tabular_only"), default="full")
    parser.add_argument("--data-root", type=Path, required=True, help="One prepared fold or legacy VT export")
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument(
        "--max-train-context",
        type=int,
        default=None,
        help="Maximum training rows; omitted uses the dataset protocol, 0 forces all rows.",
    )
    parser.add_argument("--max-val-context", type=int, default=0, help="0 keeps the full validation split.")
    parser.add_argument("--prediction-batch-size", type=int, default=512)
    parser.add_argument("--validation-chunk-size", type=int, default=512)
    parser.add_argument("--finetune-steps", type=int, default=100)
    parser.add_argument("--finetune-batch-size", type=int, default=1)
    parser.add_argument("--time-limit", type=int, default=43_200)
    parser.add_argument("--save-all-checkpoints", action="store_true")
    parser.add_argument("--mixer-type", default="MGM+CAP", choices=("MGM", "MGM+CAP", "MoE"))
    parser.add_argument("--mgm-heads", type=int, default=8)
    parser.add_argument("--cap-heads", type=int, default=8)
    parser.add_argument("--features-per-group", type=int, default=2)
    parser.add_argument("--categorical-indices", type=int, nargs="*", default=None)
    parser.add_argument(
        "--target-standardization",
        choices=("auto", "none", "train_zscore"),
        default="auto",
        help="Regression fine-tuning stability option; inference metrics always use native units.",
    )
    return parser.parse_args()


def _classification_metrics(y_true: np.ndarray, labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, labels)),
    }
    if probabilities.shape[1] == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities[:, 1]))
    else:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, probabilities, multi_class="ovo", average="macro")
        )
    return metrics


def main() -> None:
    args = _parse_args()
    spec = get_dataset_spec(args.dataset)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
    if args.mode != "tabular_only" and not args.dino_checkpoint.is_file():
        raise FileNotFoundError(f"DINOv2 checkpoint not found: {args.dino_checkpoint}")

    os.chdir(PACKAGE_ROOT)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    splits = load_benchmark_splits(spec, args.data_root)
    max_train_context = (
        spec.default_max_train_context if args.max_train_context is None else args.max_train_context
    )

    run_root = (
        args.output_dir.expanduser().resolve()
        / spec.key
        / f"fold_{args.fold}"
        / args.mode
        / f"seed_{args.seed}"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    embedding_root = (
        args.output_dir.expanduser().resolve()
        / spec.key
        / f"fold_{args.fold}"
        / f"seed_{args.seed}"
        / "embeddings"
    )
    use_tabular = args.mode != "image_only"
    use_images = args.mode != "tabular_only"
    embeddings = (
        {
            split: dataset.get_embeddings(
                args.dino_checkpoint,
                embedding_root / f"{split}_dinov2_vitb14.pt",
                args.embedding_batch_size,
                args.device,
            )
            for split, dataset in splits.items()
        }
        if use_images
        else {split: None for split in splits}
    )
    categorical = (
        args.categorical_indices
        if args.categorical_indices is not None
        else list(splits["train"].categorical_features)
    )
    if not use_tabular:
        categorical = None

    train_indices = fixed_subset_indices(
        splits["train"].y,
        max_train_context,
        seed=args.seed,
        task=spec.task,
    )
    val_indices = fixed_subset_indices(
        splits["val"].y,
        args.max_val_context,
        seed=args.seed + 1,
        task=spec.task,
    )
    x_train, y_train = splits["train"].x[train_indices], splits["train"].y[train_indices]
    x_val, y_val = splits["val"].x[val_indices], splits["val"].y[val_indices]
    image_train = subset_embeddings(embeddings["train"], train_indices)
    image_val = subset_embeddings(embeddings["val"], val_indices)

    if spec.task == "classification":
        n_classes = len(np.unique(y_train))
        task_type = TaskType.BINARY_CLASSIFICATION if n_classes == 2 else TaskType.MULTICLASS_CLASSIFICATION
        if spec.primary_metric == "roc_auc":
            validation_metric = (
                SupportedValidationMetric.ROC_AUC
                if n_classes == 2
                else SupportedValidationMetric.ROC_AUC_MULTICLASS
            )
        else:
            validation_metric = SupportedValidationMetric.ACCURACY
        y_train_tuned, y_val_tuned = pd.Series(y_train), pd.Series(y_val)
        target_mean = target_std = None
        target_standardization = "none"
    else:
        task_type = TaskType.REGRESSION
        validation_metric = (
            SupportedValidationMetric.R2
            if spec.primary_metric == "r2"
            else SupportedValidationMetric.RMSE
        )
        target_mean, target_std = float(np.mean(y_train)), float(np.std(y_train))
        if not np.isfinite(target_std) or target_std == 0:
            raise ValueError(f"{spec.key} training targets have zero or invalid variance.")
        target_standardization = (
            spec.target_standardization
            if args.target_standardization == "auto"
            else args.target_standardization
        )
        if target_standardization == "train_zscore":
            y_train_tuned = pd.Series((y_train - target_mean) / target_std)
            y_val_tuned = pd.Series((y_val - target_mean) / target_std)
        else:
            y_train_tuned, y_val_tuned = pd.Series(y_train), pd.Series(y_val)

    run_config = {
        "dataset": spec.key,
        "benchmark": spec.benchmark,
        "task": spec.task,
        "mode": args.mode,
        "fold": args.fold,
        "seed": args.seed,
        "data_root": str(args.data_root.expanduser().resolve()),
        "max_train_context": max_train_context,
        "max_val_context": args.max_val_context,
        "validation_chunk_size": args.validation_chunk_size or None,
        "target_standardization": target_standardization,
    }
    (run_root / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    from mmpfn.scripts_finetune_mm.finetune_mmpfn_main import fine_tune_mmpfn

    checkpoint = run_root / "finetuned_mmpfn.ckpt"
    fine_tune_mmpfn(
        mixer_type=args.mixer_type,
        mgm_heads=args.mgm_heads,
        cap_heads=args.cap_heads,
        features_per_group=args.features_per_group,
        save_path_to_fine_tuned_model=checkpoint,
        time_limit=args.time_limit,
        finetuning_config={
            "learning_rate": 1e-5,
            "batch_size": args.finetune_batch_size,
            "max_steps": args.finetune_steps,
        },
        validation_metric=validation_metric,
        categorical_features_index=categorical,
        task_type=task_type,
        device=SupportedDevice(args.device),
        X_train=pd.DataFrame(x_train) if use_tabular else None,
        image_train=image_train if use_images else None,
        y_train=y_train_tuned,
        X_val=pd.DataFrame(x_val) if use_tabular else None,
        image_val=image_val if use_images else None,
        y_val=y_val_tuned,
        validation_chunk_size=args.validation_chunk_size or None,
        save_all_checkpoints_dir=run_root / "all_checkpoints" if args.save_all_checkpoints else None,
        random_seed=args.seed,
        logger_level=20,
        freeze_input=True,
    )

    interface = ModelInterfaceConfig(
        FINGERPRINT_FEATURE=False,
        PREPROCESS_TRANSFORMS=[PreprocessorConfig(name="none")],
    )
    common_model_args = dict(
        model_path=checkpoint,
        inference_config=interface,
        ignore_pretraining_limits=True,
        device=args.device,
        mixer_type=args.mixer_type,
        mgm_heads=args.mgm_heads,
        cap_heads=args.cap_heads,
        features_per_group=args.features_per_group,
        categorical_features_indices=categorical,
        random_state=args.seed,
    )
    fit_x = pd.DataFrame(x_train) if use_tabular else None
    fit_image = image_train if use_images else None
    test_x = splits["test"].x if use_tabular else None
    test_image = embeddings["test"] if use_images else None
    if spec.task == "classification":
        model = MMPFNClassifier(**common_model_args).fit(fit_x, fit_image, pd.Series(y_train))
        probabilities = predict_proba_in_chunks(
            model,
            test_x,
            test_image,
            batch_size=args.prediction_batch_size,
        )
        labels = model.label_encoder_.inverse_transform(np.argmax(probabilities, axis=1))
        scores = _classification_metrics(splits["test"].y, labels, probabilities)
    else:
        model = MMPFNRegressor(**common_model_args).fit(fit_x, fit_image, np.asarray(y_train))
        prediction = predict_in_chunks(
            model,
            test_x,
            test_image,
            batch_size=args.prediction_batch_size,
        )
        y_test = splits["test"].y.astype(np.float64)
        scores = {
            "mae": float(mean_absolute_error(y_test, prediction)),
            "rmse": float(mean_squared_error(y_test, prediction) ** 0.5),
            "r2": float(r2_score(y_test, prediction)),
        }

    metrics = {
        **run_config,
        "display_name": spec.display_name,
        **scores,
        "primary_metric": spec.primary_metric,
        "primary_value": scores[spec.primary_metric],
        "higher_is_better": spec.higher_is_better,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(splits["test"].y)),
        "n_train_available": int(len(splits["train"].y)),
        "n_val_available": int(len(splits["val"].y)),
        "categorical_features": categorical,
        "target_train_mean": target_mean,
        "target_train_std": target_std,
        "all_checkpoints_dir": str(run_root / "all_checkpoints") if args.save_all_checkpoints else None,
        "checkpoint": str(checkpoint),
    }
    (run_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
