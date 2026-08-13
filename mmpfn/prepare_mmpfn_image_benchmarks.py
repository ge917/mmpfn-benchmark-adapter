"""Prepare original MMPFN image--tabular datasets for the unified runner.

The script intentionally does not download licensed or competition data.  A
caller supplies each source directory explicitly, and the prepared arrays are
written only below a user-owned benchmark-data root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_DATA_ROOT = Path("/mnt/hdd/zhangyg/projects/tab/benchmark_data/mmpfn_paper")
CBIS_IMAGE_COLUMNS = ["image file path", "cropped image file path", "ROI mask file path"]
PETFINDER_CATEGORICAL = [
    "Breed1", "Breed2", "Color1", "Color2", "Color3", "Dewormed", "FurLength",
    "Gender", "Health", "MaturitySize", "State", "Sterilized", "Type", "Vaccinated",
]
PETFINDER_NUMERIC = ["Age", "VideoAmt", "Quantity", "PhotoAmt", "Fee"]


def _parse_source_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--source-root must use DATASET_KEY=/absolute/or/relative/path")
        key, raw_path = value.split("=", 1)
        roots[key.strip()] = Path(raw_path).expanduser().resolve()
    return roots


def _split_train_val(indices: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out 10% of an original training partition for model selection."""
    try:
        return train_test_split(indices, test_size=0.10, random_state=seed, stratify=y[indices])
    except ValueError:
        # Keeps the adapter usable for tiny smoke-test sources with a rare class.
        return train_test_split(indices, test_size=0.10, random_state=seed, stratify=None)


def _split_80_10_10(y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    indices = np.arange(len(y))
    try:
        train_val, test = train_test_split(indices, test_size=0.10, random_state=seed, stratify=y)
        train, val = train_test_split(train_val, test_size=1.0 / 9.0, random_state=seed, stratify=y[train_val])
    except ValueError:
        train_val, test = train_test_split(indices, test_size=0.10, random_state=seed, stratify=None)
        train, val = train_test_split(train_val, test_size=1.0 / 9.0, random_state=seed, stratify=None)
    return {"train": np.asarray(train), "val": np.asarray(val), "test": np.asarray(test)}


def _encode_tabular(
    frame: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
    splits: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[int], list[int]]:
    """Fit imputation, scaling and category maps on train rows only."""
    train = frame.iloc[splits["train"]]
    parts: list[np.ndarray] = []
    field_lengths: list[int] = []
    categorical_indices: list[int] = []
    for column in categorical_columns:
        values = frame[column].fillna("<MISSING>").astype(str)
        categories = sorted(train[column].fillna("<MISSING>").astype(str).unique().tolist())
        mapping = {category: index for index, category in enumerate(categories)}
        encoded = values.map(mapping).fillna(-1).to_numpy(dtype=np.float32)
        categorical_indices.append(len(parts))
        parts.append(encoded.reshape(-1, 1))
        field_lengths.append(max(1, len(categories)))
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        median = float(pd.to_numeric(train[column], errors="coerce").median())
        if not np.isfinite(median):
            median = 0.0
        values = values.fillna(median).to_numpy(dtype=np.float32)
        train_values = values[splits["train"]]
        mean = float(train_values.mean())
        std = float(train_values.std())
        parts.append(((values - mean) / (std if std > 0 else 1.0)).reshape(-1, 1))
        field_lengths.append(1)
    if not parts:
        raise ValueError("No tabular columns were selected.")
    return np.concatenate(parts, axis=1), categorical_indices, field_lengths


def _write_prepared(
    output: Path,
    *,
    dataset: str,
    display_name: str,
    x: np.ndarray,
    y: np.ndarray,
    image_paths: np.ndarray,
    splits: dict[str, np.ndarray],
    categorical_indices: list[int],
    field_lengths: list[int],
    source_root: Path,
    split_protocol: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for split, indices in splits.items():
        np.savez_compressed(
            output / f"{split}.npz",
            x=x[indices].astype(np.float32),
            y=y[indices],
            image_paths=image_paths[indices].astype(str),
        )
    metadata = {
        "dataset": dataset,
        "display_name": display_name,
        "task": "classification",
        "secondary_modality": "image",
        "image_encoding": "image_file",
        "image_base_dir": str(source_root),
        "categorical_indices": categorical_indices,
        "field_lengths": field_lengths,
        "split_counts": {name: int(len(indices)) for name, indices in splits.items()},
        "source_root": str(source_root),
        "split_protocol": split_protocol,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {dataset}: " + ", ".join(f"{name}={len(indices)}" for name, indices in splits.items()))


def _cbis_image_index(source_root: Path) -> dict[str, Path]:
    jpeg_root = source_root / "jpeg"
    if not jpeg_root.is_dir():
        raise FileNotFoundError(f"Expected CBIS image directory: {jpeg_root}")
    indexed: dict[str, Path] = {}
    for directory in jpeg_root.iterdir():
        if directory.is_dir():
            files = sorted(path for path in directory.iterdir() if path.is_file())
            if files:
                indexed[directory.name] = files[0]
    if not indexed:
        raise FileNotFoundError(f"No CBIS image folders found below {jpeg_root}")
    return indexed


def _cbis_frame(source_root: Path, kind: str, split: str, image_index: dict[str, Path]) -> tuple[pd.DataFrame, np.ndarray]:
    csv_path = source_root / "csv" / f"{kind}_case_description_{split}_set.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Expected CBIS description CSV: {csv_path}")
    frame = pd.read_csv(csv_path)
    for column in CBIS_IMAGE_COLUMNS + ["pathology"]:
        if column not in frame.columns:
            raise ValueError(f"{csv_path.name} lacks required column '{column}'")
    resolved: list[list[str]] = []
    keep: list[int] = []
    for index, row in frame.iterrows():
        paths: list[str] = []
        for column in CBIS_IMAGE_COLUMNS:
            raw = row[column]
            if pd.isna(raw):
                paths = []
                break
            folder = Path(str(raw).replace("\\", "/")).parent.name
            path = image_index.get(folder)
            if path is None:
                paths = []
                break
            paths.append(str(path))
        if len(paths) == len(CBIS_IMAGE_COLUMNS):
            keep.append(index)
            resolved.append(paths)
    selected = frame.loc[keep].reset_index(drop=True).copy()
    selected["pathology"] = selected["pathology"].replace("BENIGN_WITHOUT_CALLBACK", "BENIGN")
    return selected, np.asarray(resolved, dtype=str)


def _prepare_cbis(dataset: str, source_root: Path, output: Path, fold: int) -> None:
    kind = "calc" if dataset.endswith("calc") else "mass"
    categorical = (
        ["left or right breast", "image view", "abnormality id", "calc type", "calc distribution"]
        if kind == "calc"
        else ["left or right breast", "image view", "abnormality id", "mass shape", "mass margins"]
    )
    numeric = ["breast density", "assessment", "subtlety"] if kind == "calc" else ["breast_density", "assessment", "subtlety"]
    image_index = _cbis_image_index(source_root)
    train_frame, train_paths = _cbis_frame(source_root, kind, "train", image_index)
    test_frame, test_paths = _cbis_frame(source_root, kind, "test", image_index)
    for column in categorical + numeric:
        if column not in train_frame.columns or column not in test_frame.columns:
            raise ValueError(f"CBIS-{kind} source lacks required field '{column}'")
    frame = pd.concat([train_frame, test_frame], ignore_index=True)
    image_paths = np.concatenate([train_paths, test_paths], axis=0)
    labels = frame["pathology"].astype(str)
    label_map = {label: index for index, label in enumerate(sorted(labels.unique().tolist()))}
    if set(label_map) != {"BENIGN", "MALIGNANT"}:
        raise ValueError(f"Unexpected CBIS-{kind} pathology labels: {sorted(label_map)}")
    y = labels.map(label_map).to_numpy(dtype=np.int64)
    original_train = np.arange(len(train_frame))
    train, val = _split_train_val(original_train, y, seed=42 + fold)
    splits = {"train": train, "val": val, "test": np.arange(len(train_frame), len(frame))}
    x, categorical_indices, field_lengths = _encode_tabular(frame, categorical, numeric, splits)
    _write_prepared(
        output,
        dataset=dataset,
        display_name=f"CBIS-DDSM ({kind.title()})",
        x=x,
        y=y,
        image_paths=image_paths,
        splits=splits,
        categorical_indices=categorical_indices,
        field_lengths=field_lengths,
        source_root=source_root,
        split_protocol="original CBIS train/test; stratified 90/10 train/val split from original train",
    )


def _prepare_petfinder(dataset: str, source_root: Path, output: Path, fold: int) -> None:
    table_path = source_root / "train" / "train.csv"
    image_root = source_root / "train_images"
    if not table_path.is_file() or not image_root.is_dir():
        raise FileNotFoundError(f"Expected {table_path} and {image_root}")
    frame = pd.read_csv(table_path)
    required = PETFINDER_CATEGORICAL + PETFINDER_NUMERIC + ["PetID", "AdoptionSpeed"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"PetFinder source is missing columns: {missing}")
    frame["PetID"] = frame["PetID"].astype(str)
    paths = frame["PetID"].map(lambda pet_id: image_root / f"{pet_id}-1.jpg")
    eligible = paths.map(Path.is_file).to_numpy()
    frame = frame.loc[eligible].reset_index(drop=True)
    image_paths = np.asarray([str(path) for path in paths.loc[eligible]], dtype=str)
    y = pd.to_numeric(frame["AdoptionSpeed"], errors="raise").to_numpy(dtype=np.int64)
    if not np.isin(y, np.arange(5)).all():
        raise ValueError("PetFinder AdoptionSpeed must contain only classes 0..4")
    splits = _split_80_10_10(y, seed=42 + fold)
    x, categorical_indices, field_lengths = _encode_tabular(frame, PETFINDER_CATEGORICAL, PETFINDER_NUMERIC, splits)
    _write_prepared(
        output,
        dataset=dataset,
        display_name="PetFinder-I (T+I)",
        x=x,
        y=y,
        image_paths=image_paths,
        splits=splits,
        categorical_indices=categorical_indices,
        field_lengths=field_lengths,
        source_root=source_root,
        split_protocol="eligible PetFinder train rows; stratified 80/10/10 split, random_state=42+fold",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True, choices=["mmpfn_cbis_ddsm_calc", "mmpfn_cbis_ddsm_mass", "mmpfn_petfinder_i"])
    parser.add_argument("--source-root", action="append", default=[], metavar="DATASET=PATH", help="May be supplied once per selected dataset.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--force", action="store_true", help="Accepted for unified-runner compatibility; outputs are overwritten.")
    args = parser.parse_args()
    source_roots = _parse_source_roots(args.source_root)
    for dataset in args.datasets:
        if dataset not in source_roots:
            raise ValueError(f"Missing --source-root {dataset}=PATH")
        source_root = source_roots[dataset]
        if not source_root.is_dir():
            raise FileNotFoundError(f"Source root does not exist: {source_root}")
        for fold in args.folds:
            output = args.data_root / dataset / f"fold_{fold}"
            if dataset.startswith("mmpfn_cbis_ddsm_"):
                _prepare_cbis(dataset, source_root, output, fold)
            else:
                _prepare_petfinder(dataset, source_root, output, fold)


if __name__ == "__main__":
    main()
