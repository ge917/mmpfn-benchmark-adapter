"""Export the existing VT-Bench UK Biobank Infarction split for MMPFN.

The source directory is treated as read-only.  This script copies only the
tabular feature files, labels, and image-path lists into a user-owned output
directory; the cardiac ``.npy`` images remain in their original location.
The train split intentionally uses VT-Bench's balanced downstream split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch


DEFAULT_SOURCE_ROOT = Path(
    "/mnt/hdd/jiazy/ukbiobank/TIP_OUT/cardiac_segmentations/"
    "projects/SelfSuperBio/18545/final"
)


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Check every referenced image exists before writing the output.",
    )
    args = parser.parse_args()

    source = args.source_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    split_sources = {
        "train": (
            "cardiac_features_train_imputed_noOH_tabular_imaging_Infarction_balanced_reordered.csv",
            "cardiac_labels_Infarction_train_balanced.pt",
            "cardiac_train_paths_imaging_Infarction_balanced.pt",
        ),
        "val": (
            "cardiac_features_val_imputed_noOH_tabular_imaging_reordered.csv",
            "cardiac_labels_Infarction_val.pt",
            "cardiac_val_paths_imaging.pt",
        ),
        "test": (
            "cardiac_features_test_imputed_noOH_tabular_imaging_reordered.csv",
            "cardiac_labels_Infarction_test.pt",
            "cardiac_test_paths_imaging.pt",
        ),
    }
    field_lengths = source / "tabular_lengths_reordered.pt"
    required = [field_lengths] + [source / item for values in split_sources.values() for item in values]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing VT-Bench Infarction source files:\n" + "\n".join(missing))

    output.mkdir(parents=True, exist_ok=True)
    torch.save(_load(field_lengths), output / "tabular_lengths.pt")
    for split, (feature_name, label_name, paths_name) in split_sources.items():
        features = pd.read_csv(source / feature_name, header=None)
        labels = _load(source / label_name)
        image_paths = _load(source / paths_name)
        if not (len(features) == len(labels) == len(image_paths)):
            raise ValueError(
                f"Infarction {split} is misaligned: features={len(features)}, "
                f"labels={len(labels)}, paths={len(image_paths)}"
            )
        if args.verify_images:
            missing_images = [str(path) for path in image_paths if not Path(path).is_file()]
            if missing_images:
                raise FileNotFoundError(
                    f"Infarction {split} has {len(missing_images)} missing images; "
                    f"first: {missing_images[0]}"
                )
        features.to_csv(output / f"{split}_features.csv", header=False, index=False)
        torch.save(labels, output / f"{split}_labels.pt")
        torch.save(image_paths, output / f"{split}_paths.pt")
        print(f"Prepared infarction/{split}: rows={len(features)}, features={features.shape[1]}")
    print(f"Prepared user-owned Infarction adapter files: {output}")


if __name__ == "__main__":
    main()
