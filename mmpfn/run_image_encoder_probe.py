"""Frozen-encoder linear probe for a registered image-tabular dataset.

This is deliberately a cheap pre-evaluation: one frozen visual encoder plus a
linear classifier/regressor.  It validates that every image encoder and its
cache work before spending GPU time on MMPFN fine-tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mmpfn.benchmarking.data import load_benchmark_splits
from mmpfn.benchmarking.image_encoders import IMAGE_ENCODERS
from mmpfn.benchmarking.registry import get_dataset_spec
from mmpfn.vtbench_runner_utils import fixed_subset_indices


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--image-encoder", choices=IMAGE_ENCODERS, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, default=None)
    parser.add_argument("--image-model-id", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--max-train", type=int, default=0, help="0 keeps every training row.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    spec = get_dataset_spec(args.dataset)
    if spec.secondary_modality != "image":
        raise ValueError(f"{spec.key} has text rather than image inputs; use the benchmark runner instead.")
    splits = load_benchmark_splits(spec, args.data_root)
    cache_root = args.output_dir.expanduser().resolve() / spec.key / f"fold_{args.fold}" / "encoder_embeddings"
    embeddings = {
        split: dataset.get_embeddings(
            args.dino_checkpoint,
            cache_root / f"{split}_{args.image_encoder}_raw.pt",
            args.embedding_batch_size,
            args.device,
            image_encoder=args.image_encoder,
            image_model_id=args.image_model_id,
        )
        for split, dataset in splits.items()
    }
    train_indices = fixed_subset_indices(splits["train"].y, args.max_train, seed=args.seed, task=spec.task)
    x_train = embeddings["train"][train_indices, 0, :].numpy()
    y_train = splits["train"].y[train_indices]
    x_test = embeddings["test"][:, 0, :].numpy()
    y_test = splits["test"].y

    if spec.task == "classification":
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2_000, multi_class="auto", random_state=args.seed),
        )
        probe.fit(x_train, y_train)
        probabilities = probe.predict_proba(x_test)
        labels = probe.classes_[np.argmax(probabilities, axis=1)]
        scores: dict[str, float] = {
            "accuracy": float(accuracy_score(y_test, labels)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, labels)),
        }
        if probabilities.shape[1] == 2:
            scores["roc_auc"] = float(roc_auc_score(y_test, probabilities[:, 1]))
        else:
            scores["roc_auc"] = float(roc_auc_score(y_test, probabilities, multi_class="ovo", average="macro"))
    else:
        probe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        probe.fit(x_train, y_train)
        prediction = probe.predict(x_test)
        scores = {
            "mae": float(mean_absolute_error(y_test, prediction)),
            "rmse": float(mean_squared_error(y_test, prediction) ** 0.5),
            "r2": float(r2_score(y_test, prediction)),
        }

    run_root = args.output_dir.expanduser().resolve() / spec.key / f"fold_{args.fold}" / "image_encoder_probe" / args.image_encoder / f"seed_{args.seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    result = {
        "dataset": spec.key,
        "benchmark": spec.benchmark,
        "display_name": spec.display_name,
        "task": spec.task,
        "model": "frozen_encoder_linear_probe",
        "image_encoder": args.image_encoder,
        "image_model_id": args.image_model_id,
        "fold": args.fold,
        "seed": args.seed,
        "primary_metric": spec.primary_metric,
        "primary_value": scores[spec.primary_metric],
        "higher_is_better": spec.higher_is_better,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "embedding_dim": int(x_train.shape[1]),
        **scores,
    }
    (run_root / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
