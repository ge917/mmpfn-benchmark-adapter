"""Evaluate selected saved MMPFN classification checkpoints on a VT-Bench split.

This utility is intentionally evaluation-only: it never trains, saves model
weights, or reads the test split.  It is designed to confirm checkpoint
selection made with a smaller validation context.  Test evaluation should use
only the final validation-selected checkpoint.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

from mmpfn.datasets.vtbench import VTBenchSplitDataset
from mmpfn.models.mmpfn import MMPFNClassifier
from mmpfn.models.mmpfn.constants import ModelInterfaceConfig
from mmpfn.models.mmpfn.preprocessing import PreprocessorConfig
from mmpfn.vtbench_runner_utils import predict_in_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate selected MMPFN checkpoints on a complete VT-Bench split."
    )
    parser.add_argument("--dataset", choices=("pneumonia",), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "image_only", "tabular_only"), default="tabular_only")
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="Evaluation split. Use test only after checkpoint selection is final.",
    )
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip checkpoints already recorded in --output-csv.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prediction-batch-size", type=int, default=512)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--dino-checkpoint", type=Path)
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        help="Existing or new DINOv2 embedding cache directory; required when images are used.",
    )
    parser.add_argument("--mixer-type", choices=("MGM", "MGM+CAP", "MoE"), default="MGM+CAP")
    parser.add_argument("--mgm-heads", type=int, default=8)
    parser.add_argument("--cap-heads", type=int, default=8)
    parser.add_argument("--features-per-group", type=int, default=2)
    return parser.parse_args()


def checkpoint_step(path: Path) -> int | None:
    match = re.fullmatch(r"step_(\d+)\.ckpt", path.name)
    return int(match.group(1)) if match else None


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
    if args.prediction_batch_size < 1:
        raise ValueError("--prediction-batch-size must be positive.")
    if args.embedding_batch_size < 1:
        raise ValueError("--embedding-batch-size must be positive.")
    use_tabular = args.mode != "image_only"
    use_images = args.mode != "tabular_only"
    if use_images and (args.dino_checkpoint is None or args.embedding_dir is None):
        raise ValueError("--dino-checkpoint and --embedding-dir are required for full or image_only mode.")

    missing = [str(path) for path in args.checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError("Checkpoint(s) not found:\n" + "\n".join(missing))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train = VTBenchSplitDataset(args.data_root, args.dataset, "train", "uint8")
    evaluation = VTBenchSplitDataset(args.data_root, args.dataset, args.split, "uint8")
    categorical = train.categorical_features
    train_embeddings = None
    evaluation_embeddings = None
    if use_images:
        assert args.dino_checkpoint is not None and args.embedding_dir is not None
        train_embeddings = train.get_embeddings(
            dino_checkpoint=args.dino_checkpoint,
            cache_path=args.embedding_dir / "train_dinov2_vitb14.pt",
            batch_size=args.embedding_batch_size,
            device=args.device,
        )
        evaluation_embeddings = evaluation.get_embeddings(
            dino_checkpoint=args.dino_checkpoint,
            cache_path=args.embedding_dir / f"{args.split}_dinov2_vitb14.pt",
            batch_size=args.embedding_batch_size,
            device=args.device,
        )
    no_preprocessing = ModelInterfaceConfig(
        FINGERPRINT_FEATURE=False,
        PREPROCESS_TRANSFORMS=[PreprocessorConfig(name="none")],
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    if args.resume and args.output_csv.is_file():
        rows = pd.read_csv(args.output_csv).to_dict("records")
        print(f"Resuming with {len(rows)} completed checkpoint(s).", flush=True)
    completed = {str(Path(str(row["checkpoint"])).resolve()) for row in rows}
    metric_prefix = "validation" if args.split == "val" else "test"
    accuracy_column = f"{metric_prefix}_accuracy_full"
    error_column = f"{metric_prefix}_error_full"
    count_column = f"n_{metric_prefix}"
    for checkpoint in args.checkpoints:
        checkpoint_text = str(checkpoint.resolve())
        if checkpoint_text in completed:
            print(f"Skipping completed {checkpoint.name}.", flush=True)
            continue
        print(
            f"\nEvaluating {checkpoint.name} on all {len(evaluation.y)} {args.split} rows...",
            flush=True,
        )
        classifier = MMPFNClassifier(
            model_path=checkpoint,
            inference_config=no_preprocessing,
            ignore_pretraining_limits=True,
            device=args.device,
            mixer_type=args.mixer_type,
            mgm_heads=args.mgm_heads,
            cap_heads=args.cap_heads,
            features_per_group=args.features_per_group,
            categorical_features_indices=categorical if use_tabular else None,
            random_state=args.seed,
        ).fit(
            pd.DataFrame(train.x) if use_tabular else None,
            train_embeddings if use_images else None,
            pd.Series(train.y),
        )
        predictions = predict_in_chunks(
            classifier,
            evaluation.x if use_tabular else None,
            evaluation_embeddings if use_images else None,
            batch_size=args.prediction_batch_size,
        )
        accuracy = float(accuracy_score(evaluation.y, predictions))
        row = {
            "step": checkpoint_step(checkpoint),
            "checkpoint": checkpoint_text,
            "mode": args.mode,
            accuracy_column: accuracy,
            error_column: 1.0 - accuracy,
            "n_train": int(len(train.y)),
            count_column: int(len(evaluation.y)),
            "prediction_batch_size": args.prediction_batch_size,
        }
        rows.append(row)
        completed.add(checkpoint_text)
        pd.DataFrame(rows).sort_values(accuracy_column, ascending=False).to_csv(
            args.output_csv, index=False
        )
        print(f"step {row['step']}: {metric_prefix} accuracy = {accuracy:.6f}", flush=True)

    print("\nFull-validation ranking:")
    print(pd.DataFrame(rows).sort_values(accuracy_column, ascending=False).to_string(index=False))
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
