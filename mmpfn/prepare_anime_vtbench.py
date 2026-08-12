"""Prepare a user-owned Anime export compatible with VT-Bench and MMPFN.

This mirrors ``VT-Bench/dataset/anime.ipynb``: it retains the first listed
genre, log-transforms Members/Favorites/Popularity, label-encodes the five
categorical fields before a random 80/10/10 split, then uses train-only
continuous standardisation.  Rows without a downloaded, valid image are
excluded before the split, as in the original notebook.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm


DEFAULT_ROOT = Path("/mnt/hdd/zhangyg/projects/tab/raw/anime")
CONTINUOUS_COLS = ["Members", "Favorites", "Popularity", "Episodes", "Duration", "Scored By"]
CATEGORICAL_COLS = ["Genres", "Type", "Status", "Source", "Rating"]
LABEL_COL = "Score"
ID_COL = "anime_id"
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _valid_npy(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        array = np.load(path, mmap_mode="r")
        return array.shape == (224, 224, 3) and array.dtype == np.uint8
    except (OSError, ValueError):
        return False


def _write_npy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        array = np.asarray(image.convert("RGB").resize((224, 224), resample=RESAMPLE))
    np.save(destination, array)


def main() -> None:
    args = _parse_args()
    root = args.root.expanduser().resolve()
    output = (args.output or root / "features").expanduser().resolve()
    source = root / "source" / "anime-dataset-2023.csv"
    image_root = root / "source" / "images"
    required = [ID_COL, "Image URL", LABEL_COL, *CONTINUOUS_COLS, *CATEGORICAL_COLS]
    if not source.is_file():
        raise FileNotFoundError(f"Source CSV not found: {source}")
    data = pd.read_csv(source)
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Anime source CSV is missing VT-Bench columns: {missing_columns}")

    data[LABEL_COL] = pd.to_numeric(data[LABEL_COL], errors="coerce")
    data = data.dropna(subset=[LABEL_COL, ID_COL]).copy()
    data[ID_COL] = pd.to_numeric(data[ID_COL], errors="raise").astype(int)
    data["image_path"] = data[ID_COL].map(lambda value: image_root / f"{value}.jpg")
    data = data[data["image_path"].map(lambda path: path.is_file())].copy()
    print(f"Rows with valid target and downloaded JPEG: {len(data)}")

    data["Genres"] = data["Genres"].fillna("Unknown").astype(str).map(lambda value: value.split(",")[0].strip())
    for column in ("Members", "Favorites", "Popularity"):
        data[column] = np.log1p(pd.to_numeric(data[column], errors="coerce").fillna(0))
    for column in CONTINUOUS_COLS:
        numeric = pd.to_numeric(data[column], errors="coerce")
        mode = numeric.mode()
        data[column] = numeric.fillna(mode.iloc[0] if not mode.empty else 0).astype(float)
    for column in CATEGORICAL_COLS:
        data[column] = LabelEncoder().fit_transform(data[column].fillna("Unknown").astype(str))
        data[column] = data[column].astype("category").cat.codes

    train, remainder = train_test_split(data, test_size=0.2, random_state=42)
    val, test = train_test_split(remainder, test_size=0.5, random_state=42)
    splits = {"train": train.copy(), "val": val.copy(), "test": test.copy()}
    scaler = StandardScaler().fit(splits["train"][CONTINUOUS_COLS])
    for frame in splits.values():
        frame.loc[:, CONTINUOUS_COLS] = scaler.transform(frame[CONTINUOUS_COLS])

    category_dims = [int(data[column].max()) + 1 for column in CATEGORICAL_COLS]
    output.mkdir(parents=True, exist_ok=True)
    output_images = output / "images"
    torch.save(category_dims + [1] * len(CONTINUOUS_COLS), output / "tabular_lengths.pt")
    missing_rows: list[str] = []
    features = [*CATEGORICAL_COLS, *CONTINUOUS_COLS]
    for split, frame in splits.items():
        kept_rows: list[int] = []
        paths: list[str] = []
        for index, row in tqdm(frame.iterrows(), total=len(frame), desc=f"anime {split}", unit="image"):
            source_image = row["image_path"]
            destination = output_images / f"{row[ID_COL]}.npy"
            try:
                if not _valid_npy(destination):
                    _write_npy(source_image, destination)
            except (OSError, ValueError) as error:
                missing_rows.append(f"{split}\t{row[ID_COL]}\t{error}")
                continue
            kept_rows.append(index)
            paths.append(str(destination))
        kept = frame.loc[kept_rows]
        kept[features].to_csv(output / f"{split}_features.csv", index=False, header=False)
        torch.save(torch.tensor(kept[LABEL_COL].to_numpy(), dtype=torch.float32), output / f"{split}_labels.pt")
        torch.save(paths, output / f"{split}_paths.pt")
        print(f"Prepared anime/{split}: kept={len(kept)}, skipped={len(frame) - len(kept)}")
    if missing_rows:
        report = output / "missing_images.tsv"
        report.write_text("\n".join(missing_rows) + "\n", encoding="utf-8")
        print(f"Skipped {len(missing_rows)} unreadable images; details: {report}")
    print(f"Prepared user-owned Anime features: {output}")


if __name__ == "__main__":
    main()
