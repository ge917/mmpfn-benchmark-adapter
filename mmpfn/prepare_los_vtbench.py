"""Prepare a user-owned VT-Bench-compatible Length-of-Stay feature folder.

This mirrors the tabular and image processing in VT-Bench's ``dataset/los.py``
without writing next to the original MIMIC data.  It consumes the candidate
table built by :mod:`mmpfn.build_mimic_vtbench_los`, uses only JPEGs already
downloaded into the current project, and writes ``.npy`` images plus the
standard ``*_features.csv``, ``*_labels.pt`` and ``*_paths.pt`` files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


PROJECT_RAW_ROOT = Path("/mnt/hdd/zhangyg/projects/tab/raw/mimic")
CONTINUOUS_COLS = [
    "age", "Temperature", "HeartRate", "RespRate", "SpO2", "SysBP", "DiaBP",
    "WBC", "Hemoglobin", "Platelet", "Glucose", "Creatinine", "Sodium", "Potassium",
]
CATEGORICAL_COLS = ["gender", "ViewPosition", "admission_type", "admission_location"]
LABEL_COL = "target_los_days"
USE_COLS = ["split", "image_path", LABEL_COL, *CONTINUOUS_COLS, *CATEGORICAL_COLS]
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-csv", type=Path,
        default=PROJECT_RAW_ROOT / "los" / "los_dataset_100k_enriched.csv",
    )
    parser.add_argument("--image-root", type=Path, default=PROJECT_RAW_ROOT / "images")
    parser.add_argument("--output", type=Path, default=PROJECT_RAW_ROOT / "los" / "features")
    return parser.parse_args()


def _valid_npy(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        value = np.load(path, mmap_mode="r")
        return value.shape == (224, 224, 3) and value.dtype == np.uint8
    except (OSError, ValueError):
        return False


def _create_npy(jpg_path: Path, npy_path: Path) -> None:
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(jpg_path) as image:
        array = np.asarray(image.convert("RGB").resize((224, 224), resample=RESAMPLE))
    np.save(npy_path, array)


def _encode_and_scale(data: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    data = data.copy()
    dimensions: list[int] = []
    # VT-Bench encodes categories across the complete candidate table before
    # applying its predefined train/valid/test split.
    for column in CATEGORICAL_COLS:
        encoded = data[column].fillna(-1).astype("category").cat.codes
        data[column] = encoded
        dimensions.append(max(int(encoded.max()) + 1, 1))

    train_mask = data["split"].eq("train")
    for column in CONTINUOUS_COLS:
        data[column] = data[column].fillna(0)
    scaler = StandardScaler().fit(data.loc[train_mask, CONTINUOUS_COLS])
    data.loc[:, CONTINUOUS_COLS] = scaler.transform(data[CONTINUOUS_COLS])
    return data, dimensions


def _prepare_split(
    data: pd.DataFrame,
    split: str,
    image_root: Path,
    image_output: Path,
    destination: Path,
) -> tuple[int, list[str]]:
    rows = data[data["split"].eq(split)].copy()
    keep_rows: list[int] = []
    output_paths: list[str] = []
    missing: list[str] = []
    for index, row in tqdm(rows.iterrows(), total=len(rows), desc=f"los {split}", unit="image"):
        relative = Path(str(row.image_path))
        jpg_path = image_root / relative
        npy_path = image_output / relative.with_suffix(".npy")
        if not jpg_path.is_file():
            missing.append(f"{split}\t{relative}\tmissing_jpeg")
            continue
        try:
            if not _valid_npy(npy_path):
                _create_npy(jpg_path, npy_path)
        except (OSError, ValueError) as error:
            missing.append(f"{split}\t{relative}\t{error}")
            continue
        keep_rows.append(index)
        output_paths.append(str(npy_path))

    kept = rows.loc[keep_rows]
    feature_columns = [*CATEGORICAL_COLS, *CONTINUOUS_COLS]
    output_split = "val" if split == "valid" else split
    kept[feature_columns].to_csv(destination / f"{output_split}_features.csv", index=False, header=False)
    torch.save(torch.tensor(kept[LABEL_COL].to_numpy(), dtype=torch.float32), destination / f"{output_split}_labels.pt")
    torch.save(output_paths, destination / f"{output_split}_paths.pt")
    print(f"Prepared los/{output_split}: kept={len(kept)}, skipped={len(rows) - len(kept)}")
    return len(kept), missing


def main() -> None:
    args = _parse_args()
    source_csv = args.source_csv.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    image_output = destination / "images"

    data = pd.read_csv(source_csv, usecols=USE_COLS).dropna(subset=[LABEL_COL])
    unexpected_splits = set(data["split"].dropna()) - {"train", "valid", "test"}
    if unexpected_splits:
        raise ValueError(f"Unexpected split values: {sorted(unexpected_splits)}")
    data, category_dimensions = _encode_and_scale(data)
    torch.save(category_dimensions + [1] * len(CONTINUOUS_COLS), destination / "tabular_lengths.pt")

    all_missing: list[str] = []
    for split in ("train", "valid", "test"):
        _, missing = _prepare_split(
            data, split, image_root, image_output, destination
        )
        all_missing.extend(missing)
    if all_missing:
        report = destination / "missing_images.tsv"
        report.write_text("\n".join(all_missing) + "\n", encoding="utf-8")
        print(f"Skipped {len(all_missing)} unavailable images; details: {report}")
    print(f"Prepared user-owned LOS features: {destination}")


if __name__ == "__main__":
    main()
