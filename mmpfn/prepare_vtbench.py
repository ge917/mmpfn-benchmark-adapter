"""Build MMPFN-ready, VT-Bench-style fixed splits from the public raw data.

This utility deliberately lives beside the adapter rather than modifying
VT-Bench or MMPFN's model code.  It emits exactly the files consumed by
``mmpfn.run_vtbench``.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def _text_stats(value: object) -> tuple[int, float, int]:
    if not isinstance(value, str) or not value.strip():
        return 0, 0.0, 0
    words = value.strip().split()
    return len(value.strip()), sum(map(len, words)) / len(words), len(words)


def _write_adoption_split(raw_root: Path, output: Path) -> None:
    csv_path = raw_root / "train" / "train.csv"
    if not csv_path.is_file():
        csv_path = raw_root / "train.csv"
    image_dir = raw_root / "train_images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Expected train_images beneath {raw_root}")

    df = pd.read_csv(csv_path)
    required = {"PetID", "Description", "AdoptionSpeed"}
    if missing := required.difference(df.columns):
        raise ValueError(f"Adoption CSV misses required columns: {sorted(missing)}")
    stats = df["Description"].apply(_text_stats)
    df[["desc_length", "average_word_length", "desc_words"]] = pd.DataFrame(stats.tolist(), index=df.index)
    df = df[df["AdoptionSpeed"].isin({0, 1, 2, 3, 4})].copy()
    df = df[df["PetID"].map(lambda x: (image_dir / f"{x}-1.jpg").is_file())].copy()
    df = df.drop(columns=["Name"], errors="ignore")
    train, remainder = train_test_split(df, test_size=0.2, stratify=df["AdoptionSpeed"], random_state=2022)
    valid, test = train_test_split(remainder, test_size=0.5, stratify=remainder["AdoptionSpeed"], random_state=2022)
    feature_columns = [c for c in train.columns if c not in {"AdoptionSpeed", "PetID", "Description"}]
    categorical = train[feature_columns].select_dtypes(include=["object"]).columns.tolist()
    numerical = train[feature_columns].select_dtypes(include=["number"]).columns.tolist()
    category_maps = {c: {v: i for i, v in enumerate(train[c].astype("category").cat.categories)} for c in categorical}
    scaler = StandardScaler().fit(train[numerical]) if numerical else None
    image_output = output / "images"
    output.mkdir(parents=True, exist_ok=True)

    for name, split in {"train": train, "valid": valid, "test": test}.items():
        processed = split.copy()
        for column, mapping in category_maps.items():
            processed[column] = processed[column].map(mapping).fillna(-1)
        if scaler is not None:
            processed[numerical] = scaler.transform(processed[numerical])
        processed[feature_columns] = processed[feature_columns].fillna(0)
        processed[feature_columns].to_csv(output / f"features_{name}.csv", header=False, index=False)
        torch.save(torch.tensor(processed["AdoptionSpeed"].to_numpy(), dtype=torch.long), output / f"labels_{name}.pt")
        split_image_dir = image_output / name
        split_image_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for pet_id in tqdm(processed["PetID"], desc=f"Adoption {name} images"):
            out_path = split_image_dir / f"{pet_id}-1.npy"
            if not out_path.is_file():
                image = Image.open(image_dir / f"{pet_id}-1.jpg").convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
                np.save(out_path, np.asarray(image, dtype=np.float32) / 255.0)
            paths.append(str(out_path.resolve()))
        torch.save(paths, output / f"paths_{name}.pt")
    torch.save([1 if c in numerical else len(category_maps[c]) for c in feature_columns], output / "tabular_lengths.pt")


def _uids(path: object) -> tuple[str | None, str | None]:
    values = re.findall(r"1\.3\.6\.1\.4\.1\.9590\.[\d.]+", str(path))
    return (values[0], values[1]) if len(values) >= 2 else (None, None)


def _pathology(value: object) -> str:
    value = str(value).strip().upper()
    if "MALIGNANT" in value:
        return "MALIGNANT"
    if "BENIGN_WITHOUT_CALLBACK" in value:
        return "BENIGN_WITHOUT_CALLBACK"
    return "BENIGN" if "BENIGN" in value else value


def _prepare_breast_description(frame: pd.DataFrame, lesion: str, dicom: pd.DataFrame) -> pd.DataFrame:
    renames = {"image view": "image_view", "left or right breast": "left_or_right_breast", "breast density": "breast_density", "abnormality type": "abnormality_type"}
    frame = frame.rename(columns=renames).copy()
    for col in ("calc type", "calc distribution", "mass shape", "mass margins"):
        if col in frame:
            frame[col] = frame[col].bfill()
    frame["lesion_type"] = lesion
    frame[["StudyInstanceUID", "SeriesInstanceUID"]] = pd.DataFrame(frame["image file path"].map(_uids).tolist(), index=frame.index)
    frame["pathology"] = frame["pathology"].map(_pathology)
    full = dicom[dicom["SeriesDescription"].str.contains("full", case=False, na=False)].drop_duplicates(["StudyInstanceUID", "SeriesInstanceUID"])
    merged = frame.merge(full, on=["StudyInstanceUID", "SeriesInstanceUID"], how="left", suffixes=("", "_dicom"))
    image_column = "image_path_dicom" if "image_path_dicom" in merged else "image_path"
    return merged.rename(columns={image_column: "image_path_final"}).dropna(subset=["image_path_final"]).reset_index(drop=True)


def _raw_breast_path(raw_root: Path, value: object) -> Path:
    text = str(value).replace("\\", "/")
    marker = "jpeg/"
    start = text.lower().find(marker)
    if start >= 0:
        return raw_root / text[start:]
    return raw_root / text.lstrip("/")


def _write_breast_split(raw_root: Path, output: Path) -> None:
    csv_root = raw_root / "csv"
    dicom = pd.read_csv(csv_root / "dicom_info.csv")
    rows: dict[str, pd.DataFrame] = {}
    for split in ("train", "test"):
        calc = _prepare_breast_description(pd.read_csv(csv_root / f"calc_case_description_{split}_set.csv"), "calcification", dicom)
        mass = _prepare_breast_description(pd.read_csv(csv_root / f"mass_case_description_{split}_set.csv"), "mass", dicom)
        rows[split] = pd.concat([calc, mass], ignore_index=True)
    label_map = {("calcification", "MALIGNANT"): 0, ("calcification", "BENIGN"): 1, ("calcification", "BENIGN_WITHOUT_CALLBACK"): 1, ("mass", "MALIGNANT"): 2, ("mass", "BENIGN"): 3, ("mass", "BENIGN_WITHOUT_CALLBACK"): 3}
    for frame in rows.values():
        frame["label_4c"] = [label_map.get((a, b)) for a, b in zip(frame["lesion_type"], frame["pathology"])]
        frame["resolved_image"] = frame["image_path_final"].map(lambda value: str(_raw_breast_path(raw_root, value)))
        frame.dropna(subset=["label_4c"], inplace=True)
        frame.drop(frame.index[~frame["resolved_image"].map(lambda value: Path(value).is_file())], inplace=True)
    train, valid = train_test_split(rows["train"], test_size=0.2, random_state=42, stratify=rows["train"]["label_4c"])
    test = rows["test"]
    continuous = [c for c in ["BitsAllocated", "BitsStored", "HighBit", "LargestImagePixelValue", "PixelRepresentation", "SamplesPerPixel", "SmallestImagePixelValue"] if c in train]
    categorical = [c for c in ["left_or_right_breast", "image_view", "BodyPartExamined", "ConversionType", "Modality", "PhotometricInterpretation", "SecondaryCaptureDeviceManufacturer", "SecondaryCaptureDeviceManufacturerModelName", "SeriesDescription", "SpecificCharacterSet", "breast_density", "assessment", "subtlety"] if c in train]
    categorical = [c for c in categorical if not (train[c].nunique() <= 1 and test[c].nunique() <= 1)]
    for col in continuous:
        mean = train[col].mean()
        for frame in (train, valid, test): frame.loc[:, col] = frame[col].fillna(mean)
    joined = pd.concat([train, valid, test], ignore_index=True)
    category_sizes: list[int] = []
    for col in categorical:
        joined[col] = joined[col].fillna("MISSING").astype("category")
        category_sizes.append(len(joined[col].cat.categories))
        joined[col] = joined[col].cat.codes
    train, valid, test = joined.iloc[:len(train)].copy(), joined.iloc[len(train):len(train)+len(valid)].copy(), joined.iloc[len(train)+len(valid):].copy()
    if continuous:
        scaler = StandardScaler().fit(train[continuous])
        for frame in (train, valid, test): frame.loc[:, continuous] = scaler.transform(frame[continuous])
    output.mkdir(parents=True, exist_ok=True)
    image_output = output / "images"
    features = continuous + categorical
    for name, split in {"train": train, "val": valid, "test": test}.items():
        split[features].to_csv(output / f"{name}_features.csv", header=False, index=False)
        torch.save(torch.tensor(split["label_4c"].to_numpy(), dtype=torch.long), output / f"{name}_labels.pt")
        paths: list[str] = []
        for source in tqdm(split["resolved_image"], desc=f"Breast {name} images"):
            key = hashlib.sha1(source.encode()).hexdigest()
            target = image_output / f"{key}.npy"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file():
                image = Image.open(source).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
                array = np.asarray(image, dtype=np.float32) / 255.0
                array = (array - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
                np.save(target, array)
            paths.append(str(target.resolve()))
        torch.save(paths, output / f"{name}_paths.pt")
    torch.save([1] * len(continuous) + category_sizes, output / "tabular_lengths.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("adoption", "breast"), required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset == "adoption":
        _write_adoption_split(args.raw_root.resolve(), args.output.resolve())
    else:
        _write_breast_split(args.raw_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
