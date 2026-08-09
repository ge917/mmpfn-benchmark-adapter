"""Prepare a user-owned VT-Bench-compatible CelebA attribute prediction export.

This reproduces the VT-Bench ``celebA.ipynb`` protocol: predict the
``Attractive`` attribute from the other 39 binary attributes and the aligned
face image, using a stratified 80/10/10 split with random state 42.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DEFAULT_ROOT = Path("/mnt/hdd/zhangyg/projects/tab/raw/celeba")
LABEL_COL = "Attractive"
IMAGE_COL = "image_id"
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path,
        default=DEFAULT_ROOT / "kaggle_cache" / "datasets" / "jessicali9530" / "celeba-dataset" / "versions" / "2",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "features")
    return parser.parse_args()


def _find_images(source_root: Path) -> Path:
    base = source_root / "img_align_celeba"
    for candidate in (base / "img_align_celeba", base):
        if candidate.is_dir() and next(candidate.glob("*.jpg"), None) is not None:
            return candidate
    raise FileNotFoundError(f"Could not find aligned CelebA JPEGs below {base}")


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


def main() -> None:
    args = _parse_args()
    source_root = args.source_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    attr_file = source_root / "list_attr_celeba.csv"
    image_dir = _find_images(source_root)
    data = pd.read_csv(attr_file)
    if IMAGE_COL not in data or LABEL_COL not in data:
        raise ValueError(f"Expected {IMAGE_COL!r} and {LABEL_COL!r} in {attr_file}")
    categorical = [column for column in data.columns if column != IMAGE_COL]
    train, remainder = train_test_split(
        data, test_size=0.2, stratify=data[LABEL_COL], random_state=42
    )
    val, test = train_test_split(
        remainder, test_size=0.5, stratify=remainder[LABEL_COL], random_state=42
    )
    splits = {"train": train.copy(), "val": val.copy(), "test": test.copy()}

    # Mirror the notebook's constant-feature rule, then remove the target.
    features = list(categorical)
    for column in categorical:
        if splits["train"][column].nunique() <= 1 and splits["test"][column].nunique() <= 1:
            features.remove(column)
    features.remove(LABEL_COL)
    for frame in splits.values():
        for column in [*features, LABEL_COL]:
            frame.loc[:, column] = frame[column].replace(-1, 0)
    field_lengths = [max(int(pd.concat([frame[column] for frame in splits.values()]).nunique()), 1) for column in features]

    output.mkdir(parents=True, exist_ok=True)
    image_output = output / "images"
    torch.save(field_lengths, output / "tabular_lengths.pt")
    missing: list[str] = []
    for split, frame in splits.items():
        keep_indices: list[int] = []
        output_paths: list[str] = []
        for index, row in tqdm(frame.iterrows(), total=len(frame), desc=f"celeba {split}", unit="image"):
            image_name = str(row[IMAGE_COL])
            source = image_dir / image_name
            destination = image_output / Path(image_name).with_suffix(".npy")
            if not source.is_file():
                missing.append(f"{split}\t{image_name}\tmissing_jpg")
                continue
            try:
                if not _valid_npy(destination):
                    _write_image(source, destination)
            except (OSError, ValueError) as error:
                missing.append(f"{split}\t{image_name}\t{error}")
                continue
            keep_indices.append(index)
            output_paths.append(str(destination))
        kept = frame.loc[keep_indices]
        kept[features].to_csv(output / f"{split}_features.csv", index=False, header=False)
        torch.save(torch.tensor(kept[LABEL_COL].to_numpy(), dtype=torch.long), output / f"{split}_labels.pt")
        torch.save(output_paths, output / f"{split}_paths.pt")
        print(f"Prepared celeba/{split}: kept={len(kept)}, skipped={len(frame) - len(kept)}")
    if missing:
        report = output / "missing_images.tsv"
        report.write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"Skipped {len(missing)} unavailable images; details: {report}")
    print(f"Prepared user-owned CelebA features: {output}")


if __name__ == "__main__":
    main()
