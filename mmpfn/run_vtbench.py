"""Run Full MMPFN on an existing VT-Bench Adoption or Breast split.

Run from the MultiModalPFN repository root with ``python -m mmpfn.run_vtbench``.
The command does not create new VT-Bench splits and never writes into VT-Bench.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

from mmpfn.datasets.vtbench import VTBenchSplitDataset
from mmpfn.models.mmpfn import MMPFNClassifier
from mmpfn.models.mmpfn.constants import ModelInterfaceConfig
from mmpfn.models.mmpfn.preprocessing import PreprocessorConfig
from mmpfn.scripts_finetune_mm.constant_utils import (
    SupportedDevice,
    SupportedValidationMetric,
    TaskType,
)
from mmpfn.vtbench_runner_utils import fixed_subset_indices, predict_in_chunks, subset_embeddings


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Full MMPFN on a fixed VT-Bench split.")
    parser.add_argument("--dataset", choices=("adoption", "breast", "pneumonia", "infarction"), required=True)
    parser.add_argument(
        "--mode",
        choices=("full", "image_only", "tabular_only"),
        default="full",
        help="Keep both modalities, or ablate one modality while preserving the split and seed.",
    )
    parser.add_argument("--data-root", type=Path, required=True, help="VT-Bench exported feature directory")
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "checkpoints" / "vtbench")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument(
        "--max-train-context",
        type=int,
        default=6000,
        help="Maximum fixed training-context rows; 0 keeps all rows.",
    )
    parser.add_argument(
        "--max-val-context",
        type=int,
        default=2000,
        help="Maximum fixed validation rows; 0 keeps all rows.",
    )
    parser.add_argument("--prediction-batch-size", type=int, default=512)
    parser.add_argument(
        "--validation-chunk-size",
        type=int,
        default=0,
        help="Validate in chunks while keeping the complete validation split; 0 uses one chunk.",
    )
    parser.add_argument(
        "--save-all-checkpoints",
        action="store_true",
        help="Save the initial model and every validated step with a checkpoint_metrics.csv table.",
    )
    parser.add_argument("--finetune-steps", type=int, default=100)
    parser.add_argument("--finetune-batch-size", type=int, default=1)
    parser.add_argument("--time-limit", type=int, default=600)
    parser.add_argument("--mixer-type", default="MGM+CAP", choices=("MGM", "MGM+CAP", "MoE"))
    parser.add_argument("--mgm-heads", type=int, default=8)
    parser.add_argument("--cap-heads", type=int, default=8)
    parser.add_argument("--features-per-group", type=int, default=2)
    parser.add_argument(
        "--categorical-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional 0-based categorical column indices. Defaults to MMPFN inference.",
    )
    return parser.parse_args()


def _load_splits(args: argparse.Namespace) -> dict[str, VTBenchSplitDataset]:
    image_encoding = {
        "adoption": "zero_one",
        "breast": "imagenet_normalized",
        "pneumonia": "uint8",
        "infarction": "zero_one",
    }[args.dataset]
    return {
        split: VTBenchSplitDataset(args.data_root, args.dataset, split, image_encoding)
        for split in ("train", "val", "test")
    }


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable.")

    # The original fine-tuning module uses relative log paths.
    os.chdir(PACKAGE_ROOT)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    splits = _load_splits(args)
    output_dir = args.output_dir.expanduser().resolve() / args.dataset / args.mode / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Image embeddings depend only on the fixed split and seed, not on which
    # modality ablation is trained.  Keep one shared cache for fair ablations.
    embedding_dir = args.output_dir.expanduser().resolve() / args.dataset / f"seed_{args.seed}" / "embeddings"
    embeddings = (
        {
            split: dataset.get_embeddings(
                dino_checkpoint=args.dino_checkpoint,
                cache_path=embedding_dir / f"{split}_dinov2_vitb14.pt",
                batch_size=args.embedding_batch_size,
                device=args.device,
            )
            for split, dataset in splits.items()
        }
        if args.mode != "tabular_only"
        else {split: None for split in splits}
    )
    use_tabular = args.mode != "image_only"
    use_images = args.mode != "tabular_only"
    categorical = (
        args.categorical_indices
        if args.categorical_indices is not None
        else splits["train"].categorical_features
    )
    train_indices = fixed_subset_indices(
        splits["train"].y, args.max_train_context, seed=args.seed, task="classification"
    )
    val_indices = fixed_subset_indices(
        splits["val"].y, args.max_val_context, seed=args.seed + 1, task="classification"
    )
    x_train = splits["train"].x[train_indices]
    y_train = splits["train"].y[train_indices]
    x_val = splits["val"].x[val_indices]
    y_val = splits["val"].y[val_indices]
    image_train = subset_embeddings(embeddings["train"], train_indices)
    image_val = subset_embeddings(embeddings["val"], val_indices)

    from mmpfn.scripts_finetune_mm.finetune_mmpfn_main import fine_tune_mmpfn

    fine_tuned_checkpoint = output_dir / "finetuned_mmpfn.ckpt"
    fine_tune_mmpfn(
        mixer_type=args.mixer_type,
        mgm_heads=args.mgm_heads,
        cap_heads=args.cap_heads,
        features_per_group=args.features_per_group,
        save_path_to_fine_tuned_model=fine_tuned_checkpoint,
        time_limit=args.time_limit,
        finetuning_config={
            "learning_rate": 1e-5,
            "batch_size": args.finetune_batch_size,
            "max_steps": args.finetune_steps,
        },
        validation_metric=SupportedValidationMetric.ACCURACY,
        categorical_features_index=categorical if use_tabular else None,
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        device=SupportedDevice(args.device),
        X_train=pd.DataFrame(x_train) if use_tabular else None,
        image_train=image_train if use_images else None,
        y_train=pd.Series(y_train),
        X_val=pd.DataFrame(x_val) if use_tabular else None,
        image_val=image_val if use_images else None,
        y_val=pd.Series(y_val),
        validation_chunk_size=args.validation_chunk_size or None,
        save_all_checkpoints_dir=output_dir / "all_checkpoints" if args.save_all_checkpoints else None,
        random_seed=args.seed,
        logger_level=20,
        freeze_input=True,
    )

    no_preprocessing = ModelInterfaceConfig(
        FINGERPRINT_FEATURE=False,
        PREPROCESS_TRANSFORMS=[PreprocessorConfig(name="none")],
    )
    classifier = MMPFNClassifier(
        model_path=fine_tuned_checkpoint,
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
        pd.DataFrame(x_train) if use_tabular else None,
        image_train if use_images else None,
        pd.Series(y_train),
    )
    predictions = predict_in_chunks(
        classifier,
        splits["test"].x if use_tabular else None,
        embeddings["test"] if use_images else None,
        batch_size=args.prediction_batch_size,
    )
    metrics = {
        "dataset": args.dataset,
        "mode": args.mode,
        "seed": args.seed,
        "accuracy": float(accuracy_score(splits["test"].y, predictions)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(splits["test"].y)),
        "n_train_available": int(len(splits["train"].y)),
        "n_val_available": int(len(splits["val"].y)),
        "max_train_context": args.max_train_context,
        "max_val_context": args.max_val_context,
        "validation_chunk_size": args.validation_chunk_size or None,
        "all_checkpoints_dir": str(output_dir / "all_checkpoints") if args.save_all_checkpoints else None,
        "categorical_features": categorical if use_tabular else None,
        "field_length_categorical_hint": splits["train"].categorical_features,
        "checkpoint": str(fine_tuned_checkpoint),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
