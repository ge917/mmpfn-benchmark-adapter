"""Export the fixed VT-Bench DVM-Car split into user-owned prepared files.

The source ``/mnt/hdd/jiazy/DVM-Car`` is read-only.  Its preprocessing script
defines the VT-Bench protocol: 286-way ``Genmodel_ID`` classification, a
40/10/50 stratified split (seed 2022), and 17 tabular inputs consisting of
13 numeric variables plus four categorical variables.  This exporter uses the
*final* physical-feature CSVs, never the intermediate ``dvm_full_features``
CSVs, which contain IDs and the label column.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


DEFAULT_SOURCE_ROOT = Path("/mnt/hdd/jiazy/DVM-Car/features")
FEATURE_COLUMNS = [
    "Wheelbase", "Height", "Width", "Length",
    "Adv_year", "Adv_month", "Reg_year", "Runned_Miles", "Price",
    "Seat_num", "Door_num", "Entry_price", "Engine_size",
    "Color", "Bodytype", "Gearbox", "Fuel_type",
]
SPLITS = ("train", "val", "test")


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location="cpu")


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required DVM-Car source file is missing: {path}")
    return path


def _source_files(root: Path, split: str) -> tuple[Path, Path, Path]:
    feature_path = _require(
        root / f"dvm_features_{split}_noOH_all_views_physical_jittered_50.csv"
    )
    label_path = _require(root / f"labels_model_all_{split}_all_views.pt")
    paths_path = _require(root / f"{split}_paths_all_views.pt")
    return feature_path, label_path, paths_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Check every referenced JPEG exists before writing output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source = args.source_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    field_lengths_path = _require(source / "tabular_lengths_all_views_physical.pt")
    field_lengths = [int(value) for value in _torch_load(field_lengths_path)]
    if len(field_lengths) != len(FEATURE_COLUMNS):
        raise ValueError(
            "DVM-Car field-length metadata is inconsistent: "
            f"expected {len(FEATURE_COLUMNS)}, got {len(field_lengths)}."
        )
    categorical_indices = [index for index, length in enumerate(field_lengths) if length > 1]
    if categorical_indices != [13, 14, 15, 16]:
        raise ValueError(
            "Unexpected DVM-Car categorical columns. Expected [13, 14, 15, 16], "
            f"got {categorical_indices}."
        )

    # All source checks happen before any user-owned files are created.
    prepared: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    total_rows = 0
    for split in SPLITS:
        feature_path, label_path, paths_path = _source_files(source, split)
        x = pd.read_csv(feature_path, header=None).to_numpy(dtype=np.float32)
        y = np.asarray(_torch_load(label_path), dtype=np.int64).reshape(-1)
        image_paths = np.asarray([str(path) for path in _torch_load(paths_path)], dtype=str)
        if x.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(
                f"DVM-Car {split} has {x.shape[1]} features; expected {len(FEATURE_COLUMNS)}."
            )
        if not (len(x) == len(y) == len(image_paths)):
            raise ValueError(
                f"DVM-Car {split} is misaligned: x={len(x)}, y={len(y)}, images={len(image_paths)}."
            )
        if not np.isfinite(x).all():
            raise ValueError(f"DVM-Car {split} contains non-finite tabular values.")
        if y.min() < 0 or y.max() >= 286:
            raise ValueError(f"DVM-Car {split} labels are outside the expected [0, 285] range.")
        if args.verify_images:
            missing = [path for path in image_paths if not Path(path).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"DVM-Car {split} has {len(missing)} missing images; first: {missing[0]}"
                )
        prepared[split] = (x, y, image_paths)
        total_rows += len(y)

    if total_rows != 176_414:
        raise ValueError(f"DVM-Car has {total_rows} rows; expected VT-Bench total 176414.")

    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 1,
        "dataset": "vt_dvm_car",
        "display_name": "DVM-Car",
        "benchmark": "vtbench",
        "task": "classification",
        "label": "Genmodel_ID (286-way vehicle type)",
        "feature_columns": FEATURE_COLUMNS,
        "categorical_indices": categorical_indices,
        "field_lengths": field_lengths,
        "image_encoding": "image_file",
        "source_root": str(source),
        "split_protocol": "stratified 40/10/50 split, random_state=2022",
        "note": "Source remains read-only; labels, IDs, image names, and predicted viewpoint are excluded from x.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for split, (x, y, image_paths) in prepared.items():
        np.savez_compressed(output / f"{split}.npz", x=x, y=y, image_paths=image_paths)
        print(f"Prepared dvm_car/{split}: rows={len(y)}, features={x.shape[1]}")
    print(f"Prepared user-owned DVM-Car adapter files: {output}")


if __name__ == "__main__":
    main()
