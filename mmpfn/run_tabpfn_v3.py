"""Evaluate the standalone TabPFN-3 baseline on one registered dataset.

This intentionally evaluates the tabular modality only.  It is not an MMPFN
replacement: MMPFN still uses its released TabPFN-v2 backbone internally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from mmpfn.benchmarking.data import load_benchmark_splits
from mmpfn.benchmarking.registry import get_dataset_spec
from mmpfn.vtbench_runner_utils import fixed_subset_indices


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-context", type=int, default=0, help="0 keeps every training row.")
    parser.add_argument("--prediction-batch-size", type=int, default=2_048)
    return parser.parse_args()


def _iter_batches(values: np.ndarray, batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def main() -> None:
    args = _parse_args()
    spec = get_dataset_spec(args.dataset)
    splits = load_benchmark_splits(spec, args.data_root)
    train_indices = fixed_subset_indices(
        splits["train"].y,
        args.max_train_context,
        seed=args.seed,
        task=spec.task,
    )
    x_train = splits["train"].x[train_indices]
    y_train = splits["train"].y[train_indices]
    x_test, y_test = splits["test"].x, splits["test"].y
    categorical = list(splits["train"].categorical_features)

    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
    except ImportError as error:
        raise RuntimeError(
            "TabPFN-3 is required. Activate the tabpfn-v3 environment and install 'tabpfn>=8.0.0'."
        ) from error

    common = {
        "device": args.device,
        "random_state": args.seed,
        "categorical_features_indices": categorical or None,
    }
    if spec.task == "classification":
        model = TabPFNClassifier.create_default_for_version(ModelVersion.V3, **common)
        model.fit(x_train, y_train)
        probabilities = np.concatenate(
            [model.predict_proba(batch) for batch in _iter_batches(x_test, args.prediction_batch_size)]
        )
        labels = model.classes_[np.argmax(probabilities, axis=1)]
        metrics: dict[str, float | int | str | list[int] | None] = {
            "accuracy": float(accuracy_score(y_test, labels)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, labels)),
        }
        if probabilities.shape[1] == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities[:, 1]))
        else:
            metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities, multi_class="ovo", average="macro"))
    else:
        model = TabPFNRegressor.create_default_for_version(ModelVersion.V3, **common)
        model.fit(x_train, y_train)
        prediction = np.concatenate([model.predict(batch) for batch in _iter_batches(x_test, args.prediction_batch_size)])
        metrics = {
            "mae": float(mean_absolute_error(y_test, prediction)),
            "rmse": float(mean_squared_error(y_test, prediction) ** 0.5),
            "r2": float(r2_score(y_test, prediction)),
        }

    run_root = args.output_dir.expanduser().resolve() / spec.key / f"fold_{args.fold}" / "tabpfn_v3" / f"seed_{args.seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    result = {
        "dataset": spec.key,
        "benchmark": spec.benchmark,
        "display_name": spec.display_name,
        "task": spec.task,
        "model": "TabPFN-3",
        "fold": args.fold,
        "seed": args.seed,
        "primary_metric": spec.primary_metric,
        "primary_value": metrics[spec.primary_metric],
        "higher_is_better": spec.higher_is_better,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_train_available": int(len(splits["train"].x)),
        "categorical_features": categorical,
        "max_train_context": args.max_train_context,
        **metrics,
    }
    (run_root / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
