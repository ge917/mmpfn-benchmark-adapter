"""Create user-owned MIMIC VT-Bench feature folders from a read-only source.

This script reads the existing tabular splits and labels under
``/mnt/hdd/jiazy/mimic`` but never writes below that directory.  It converts
downloaded JPEGs in the current user's project into the exact 224x224 uint8
``.npy`` layout expected by MMPFN, copies the split metadata to a user-owned
directory, and rewrites only the copied ``*_paths.pt`` files.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


SOURCE_ROOT = Path("/mnt/hdd/jiazy/mimic")  # read only
PROJECT_RAW_ROOT = Path("/mnt/hdd/zhangyg/projects/tab/raw/mimic")
TASKS = {
    "pneumonia": {
        "source_features": SOURCE_ROOT / "classification" / "features",
        "splits": ("train", "valid", "test"),
        "resample": Image.Resampling.BILINEAR,
    },
    "rr": {
        "source_features": SOURCE_ROOT / "regression" / "rr" / "features",
        "splits": ("train", "val", "test"),
        "resample": Image.Resampling.LANCZOS,
    },
}


def relative_npy_path(source_path: str) -> Path:
    """Map a legacy jiazy image path to its relative pXX/... .npy location."""
    value = str(source_path).replace("\\", "/")
    marker = "/image/"
    if marker not in value:
        raise ValueError(f"Unexpected legacy image path: {source_path}")
    relative = value.split(marker, 1)[1].lstrip("/")
    if relative.startswith("files/"):
        relative = relative[len("files/") :]
    return Path(relative)


def valid_npy(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        image = np.load(path, mmap_mode="r")
        return image.shape == (224, 224, 3) and image.dtype == np.uint8
    except Exception:
        return False


def make_npy(jpg_path: Path, npy_path: Path, resample: Image.Resampling) -> None:
    if not jpg_path.is_file():
        raise FileNotFoundError(f"Missing downloaded JPEG: {jpg_path}")
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(jpg_path) as image:
        array = np.asarray(image.convert("RGB").resize((224, 224), resample=resample))
    np.save(npy_path, array)


def copy_lengths(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    lengths = source / "tabular_lengths.pt"
    if not lengths.is_file():
        raise FileNotFoundError(lengths)
    shutil.copy2(lengths, destination / lengths.name)


def prepare_task(task: str, image_root: Path, output_root: Path) -> None:
    config = TASKS[task]
    source = config["source_features"]
    destination = output_root / task / "features"
    copy_lengths(source, destination)
    missing_report: list[str] = []
    for split in config["splits"]:
        legacy_paths = torch.load(source / f"{split}_paths.pt", map_location="cpu")
        source_labels = torch.load(source / f"{split}_labels.pt", map_location="cpu")
        source_features = pd.read_csv(source / f"{split}_features.csv", header=None)
        if not (len(legacy_paths) == len(source_labels) == len(source_features)):
            raise ValueError(f"Source {task}/{split} files are not aligned")

        new_paths: list[str] = []
        keep_indices: list[int] = []
        for index, legacy_path in enumerate(tqdm(legacy_paths, desc=f"{task} {split}", unit="image")):
            relative_npy = relative_npy_path(legacy_path)
            new_npy = image_root / relative_npy
            try:
                if not valid_npy(new_npy):
                    make_npy(new_npy.with_suffix(".jpg"), new_npy, config["resample"])
            except (FileNotFoundError, OSError) as error:
                missing_report.append(f"{split}\t{relative_npy}\t{error}")
                continue
            keep_indices.append(index)
            new_paths.append(str(new_npy))

        source_features.iloc[keep_indices].to_csv(
            destination / f"{split}_features.csv", header=False, index=False
        )
        torch.save(source_labels[keep_indices], destination / f"{split}_labels.pt")
        torch.save(new_paths, destination / f"{split}_paths.pt")
        print(f"Prepared {task}/{split}: kept={len(keep_indices)}, skipped={len(legacy_paths) - len(keep_indices)}")

    if missing_report:
        report = destination / "missing_images.tsv"
        report.write_text("\n".join(missing_report) + "\n", encoding="utf-8")
        print(f"Skipped {len(missing_report)} unavailable images; details: {report}")
    print(f"Prepared user-owned {task} features: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), required=True)
    parser.add_argument("--image-root", type=Path, default=PROJECT_RAW_ROOT / "images")
    parser.add_argument("--output-root", type=Path, default=PROJECT_RAW_ROOT)
    args = parser.parse_args()
    for task in args.tasks:
        prepare_task(task, args.image_root, args.output_root)


if __name__ == "__main__":
    main()
