"""Export the public Pawpularity raw archive into VT-Bench-style fixed splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


CATEGORICAL_COLUMNS = [
    "Subject Focus", "Eyes", "Face", "Near", "Action", "Accessory", "Group", "Collage",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the VT-Bench Pawpularity regression split.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw_root.resolve()
    output = args.output.resolve()
    df = pd.read_csv(raw / "train.csv")
    required = {"Id", "Pawpularity", *CATEGORICAL_COLUMNS}
    if missing := required.difference(df.columns):
        raise ValueError(f"Missing Pawpularity columns: {sorted(missing)}")
    df["image_path"] = df["Id"].map(lambda value: raw / "train" / f"{value}.jpg")
    df = df[df["image_path"].map(Path.is_file)].copy()
    # This intentionally follows the VT-Bench notebook: categories are encoded
    # before the fixed random 80:10:10 split.
    lengths: list[int] = []
    for column in CATEGORICAL_COLUMNS:
        values = df[column].fillna(-1).astype("category")
        lengths.append(len(values.cat.categories))
        df[column] = values.cat.codes
    train, remaining = train_test_split(df, test_size=0.2, random_state=42)
    valid, test = train_test_split(remaining, test_size=0.5, random_state=42)
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "images"
    for name, split in {"train": train, "val": valid, "test": test}.items():
        split[CATEGORICAL_COLUMNS].to_csv(output / f"{name}_features.csv", index=False, header=False)
        torch.save(torch.tensor(split["Pawpularity"].to_numpy(), dtype=torch.float32), output / f"{name}_labels.pt")
        paths: list[str] = []
        for row in tqdm(split.itertuples(index=False), total=len(split), desc=f"Pawpularity {name} images"):
            source = Path(row.image_path)
            target = image_dir / f"{row.Id}.npy"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file():
                image = Image.open(source).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
                np.save(target, np.asarray(image, dtype=np.float32) / 255.0)
            paths.append(str(target))
        torch.save(paths, output / f"{name}_paths.pt")
    torch.save(lengths, output / "tabular_lengths.pt")
    print(f"Prepared Pawpularity: train={len(train)}, val={len(valid)}, test={len(test)}, features={len(CATEGORICAL_COLUMNS)}")


if __name__ == "__main__":
    main()
