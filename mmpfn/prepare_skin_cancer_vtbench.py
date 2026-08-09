"""Prepare a user-owned PAD-UFES-20 export compatible with VT-Bench Skin Cancer.

The feature selection, stratified 80/10/10 split, categorical coding and
train-only continuous scaling reproduce ``VT-Bench/dataset/skin_cancer.ipynb``.
Only derived files below the current user's project directory are written.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


DEFAULT_ROOT = Path("/mnt/hdd/zhangyg/projects/tab/raw/skin_cancer")
CONTINUOUS_COLS = ["age", "diameter_1", "diameter_2"]
CATEGORICAL_COLS = [
    "smoke", "drink", "background_father", "background_mother", "pesticide", "gender",
    "skin_cancer_history", "cancer_history", "has_piped_water", "has_sewage_system",
    "fitspatrick", "region", "itch", "grew", "hurt", "changed", "bleed", "elevation",
    "biopsed",
]
LABEL_COL = "diagnostic"
IMAGE_COL = "img_id"
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_ROOT / "source")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "features")
    return parser.parse_args()


def _image_index(source_root: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in source_root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".png":
            if path.name in images:
                raise ValueError(f"Duplicate image filename in source: {path.name}")
            images[path.name] = path
    if not images:
        raise FileNotFoundError(f"No PNG files found below {source_root}")
    return images


def _valid_npy(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        array = np.load(path, mmap_mode="r")
        return array.shape == (224, 224, 3) and array.dtype == np.uint8
    except (OSError, ValueError):
        return False


def _write_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        array = np.asarray(image.convert("RGB").resize((224, 224), resample=RESAMPLE))
    np.save(destination, array)


def _split_and_transform(data: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], list[int], dict[int, int]]:
    train, remainder = train_test_split(
        data, test_size=0.2, stratify=data[LABEL_COL], random_state=42
    )
    val, test = train_test_split(
        remainder, test_size=0.5, stratify=remainder[LABEL_COL], random_state=42
    )
    splits = {"train": train.copy(), "val": val.copy(), "test": test.copy()}

    continuous = [column for column in CONTINUOUS_COLS if splits["train"][column].nunique() > 1]
    categorical = [column for column in CATEGORICAL_COLS if splits["train"][column].nunique() > 1]
    for column in continuous:
        train_mean = splits["train"][column].mean()
        for frame in splits.values():
            frame.loc[:, column] = frame[column].fillna(train_mean)
    for column in categorical:
        for frame in splits.values():
            frame.loc[:, column] = frame[column].fillna("MISSING")

    # The notebook builds one category map from the concatenated splits.
    combined = pd.concat(splits.values(), axis=0)
    field_lengths: list[int] = []
    for column in categorical:
        dtype = combined[column].astype("category").dtype
        field_lengths.append(len(dtype.categories))
        for frame in splits.values():
            frame.loc[:, column] = frame[column].astype(dtype).cat.codes
    label_dtype = combined[LABEL_COL].astype("category").dtype
    label_map = {int(code): str(label) for code, label in enumerate(label_dtype.categories)}
    for frame in splits.values():
        frame.loc[:, LABEL_COL] = frame[LABEL_COL].astype(label_dtype).cat.codes

    if continuous:
        scaler = StandardScaler().fit(splits["train"][continuous])
        for frame in splits.values():
            frame.loc[:, continuous] = scaler.transform(frame[continuous])

    return splits, [1] * len(continuous) + field_lengths, label_map


def main() -> None:
    args = _parse_args()
    source_root = args.source_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    metadata = source_root / "metadata.csv"
    required = [IMAGE_COL, LABEL_COL, *CONTINUOUS_COLS, *CATEGORICAL_COLS]
    data = pd.read_csv(metadata)
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"metadata.csv is missing VT-Bench columns: {missing}")
    data = data.dropna(subset=[IMAGE_COL, LABEL_COL]).copy()
    images = _image_index(source_root)
    splits, field_lengths, label_map = _split_and_transform(data)

    output.mkdir(parents=True, exist_ok=True)
    image_output = output / "images"
    torch.save(field_lengths, output / "tabular_lengths.pt")
    (output / "label_mapping.txt").write_text(
        "\n".join(f"{key}\t{value}" for key, value in label_map.items()) + "\n", encoding="utf-8"
    )
    missing_rows: list[str] = []
    feature_columns = [column for column in CONTINUOUS_COLS if column in data.columns and column not in []]
    feature_columns = [column for column in CONTINUOUS_COLS if column in splits["train"].columns and splits["train"][column].nunique() > 1]
    feature_columns += [column for column in CATEGORICAL_COLS if column in splits["train"].columns and splits["train"][column].nunique() > 1]

    for split, frame in splits.items():
        keep_indices: list[int] = []
        paths: list[str] = []
        for index, row in tqdm(frame.iterrows(), total=len(frame), desc=f"skin_cancer {split}", unit="image"):
            image_name = Path(str(row[IMAGE_COL])).name
            source = images.get(image_name)
            if source is None:
                missing_rows.append(f"{split}\t{image_name}\tmissing_png")
                continue
            destination = image_output / image_name.replace(".png", ".npy")
            try:
                if not _valid_npy(destination):
                    _write_image(source, destination)
            except (OSError, ValueError) as error:
                missing_rows.append(f"{split}\t{image_name}\t{error}")
                continue
            keep_indices.append(index)
            paths.append(str(destination))
        kept = frame.loc[keep_indices]
        kept[feature_columns].to_csv(output / f"{split}_features.csv", index=False, header=False)
        torch.save(torch.tensor(kept[LABEL_COL].to_numpy(), dtype=torch.long), output / f"{split}_labels.pt")
        torch.save(paths, output / f"{split}_paths.pt")
        print(f"Prepared skin_cancer/{split}: kept={len(kept)}, skipped={len(frame) - len(kept)}")
    if missing_rows:
        report = output / "missing_images.tsv"
        report.write_text("\n".join(missing_rows) + "\n", encoding="utf-8")
        print(f"Skipped {len(missing_rows)} unavailable images; details: {report}")
    print(f"Prepared user-owned Skin Cancer features: {output}")


if __name__ == "__main__":
    main()
