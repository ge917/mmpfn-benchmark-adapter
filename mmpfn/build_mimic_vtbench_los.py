"""Build the VT-Bench LOS candidate table from user-owned MIMIC source files.

This is a path-safe adaptation of VT-Bench's
``dataset/Constructed_datasets/built_regression.py``.  It reads MIMIC-IV v2.2
and MIMIC-CXR-JPG metadata from the user's project directories, writes only a
LOS candidate CSV below ``/mnt/hdd/zhangyg``, and does *not* download or alter
any images.  A later step downloads only candidate images that are missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_MIMIC_ROOT = Path("/mnt/hdd/zhangyg/projects/tab/raw/mimic_iv_2_2")
DEFAULT_OUTPUT = Path("/mnt/hdd/zhangyg/projects/tab/raw/mimic/los")
TARGET_COUNT = 100_000
CHUNK_SIZE = 5_000_000

VITALS = {
    "Temperature": [223761, 223762],
    "HeartRate": [220045],
    "RespRate": [220210],
    "SpO2": [220277],
    "SysBP": [220179, 220050],
    "DiaBP": [220180, 220051],
}
LABS = {
    "WBC": [51301, 51300],
    "Hemoglobin": [51222],
    "Platelet": [51265],
    "Glucose": [50931],
    "Creatinine": [50912],
    "Sodium": [50983],
    "Potassium": [50971],
}


def _existing(base: Path, stem: str) -> Path:
    for suffix in (".csv.gz", ".csv"):
        candidate = base / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing {stem}.csv[.gz] below {base}")


def _image_path(subject_id: object, study_id: object, dicom_id: object) -> str:
    subject = str(int(subject_id))
    return f"p{subject[:2]}/p{subject}/s{int(study_id)}/{dicom_id}.jpg"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-root", type=Path, default=DEFAULT_MIMIC_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-count", type=int, default=TARGET_COUNT)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    return parser.parse_args()


def _align_cxr_to_admissions(mimic_root: Path) -> pd.DataFrame:
    metadata_path = _existing(mimic_root / "cxr", "mimic-cxr-2.0.0-metadata")
    metadata = pd.read_csv(
        metadata_path,
        usecols=["subject_id", "study_id", "dicom_id", "StudyDate", "StudyTime", "ViewPosition"],
    )
    metadata = metadata[metadata["ViewPosition"].isin(["AP", "PA"])].copy()
    metadata["studydatetime"] = pd.to_datetime(
        metadata["StudyDate"].astype(str)
        + " "
        + metadata["StudyTime"].astype(str).str.split(".").str[0].str.zfill(6),
        format="%Y%m%d %H%M%S",
        errors="coerce",
    )
    metadata = metadata.dropna(subset=["studydatetime"])

    transfers = pd.read_csv(
        _existing(mimic_root / "hosp", "transfers"),
        usecols=["subject_id", "hadm_id", "intime", "outtime"],
    ).dropna(subset=["hadm_id", "intime", "outtime"])
    transfers["intime"] = pd.to_datetime(transfers["intime"])
    transfers["outtime"] = pd.to_datetime(transfers["outtime"])
    transfers = transfers[transfers["subject_id"].isin(metadata["subject_id"].unique())]

    aligned = metadata.merge(transfers, on="subject_id", how="left")
    aligned = aligned[
        (aligned["studydatetime"] >= aligned["intime"])
        & (aligned["studydatetime"] <= aligned["outtime"])
    ].drop_duplicates(subset=["study_id", "dicom_id"])
    return aligned[
        ["subject_id", "study_id", "dicom_id", "hadm_id", "studydatetime", "ViewPosition"]
    ].copy()


def _extract_clinical_features(mimic_root: Path, cohort: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    valid_hadms = set(cohort["hadm_id"].dropna().astype("int64").tolist())
    item_to_name = {
        item_id: name for name, ids in {**VITALS, **LABS}.items() for item_id in ids
    }
    extracted: list[pd.DataFrame] = []

    def scan(path: Path, item_ids: list[int], description: str) -> None:
        reader = pd.read_csv(
            path,
            chunksize=chunk_size,
            usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        )
        for chunk in tqdm(reader, desc=description):
            chunk = chunk[chunk["hadm_id"].isin(valid_hadms) & chunk["itemid"].isin(item_ids)]
            chunk = chunk.dropna(subset=["valuenum"])
            if 223761 in chunk["itemid"].values:
                fahrenheit = chunk["itemid"] == 223761
                chunk.loc[fahrenheit, "valuenum"] = (chunk.loc[fahrenheit, "valuenum"] - 32) * 5 / 9
                chunk.loc[fahrenheit, "itemid"] = 223762
            if not chunk.empty:
                extracted.append(chunk)

    scan(
        _existing(mimic_root / "icu", "chartevents"),
        [item for ids in VITALS.values() for item in ids],
        "Reading MIMIC vital signs",
    )
    scan(
        _existing(mimic_root / "hosp", "labevents"),
        [item for ids in LABS.values() for item in ids],
        "Reading MIMIC lab values",
    )
    if not extracted:
        raise RuntimeError("No requested MIMIC clinical measurements were found.")

    values = pd.concat(extracted, ignore_index=True)
    values["charttime"] = pd.to_datetime(values["charttime"])
    values["feature_name"] = values["itemid"].map(item_to_name)
    timestamps = cohort[["hadm_id", "study_id", "studydatetime"]].drop_duplicates()
    values = values.merge(timestamps, on="hadm_id", how="inner")
    elapsed_hours = (values["charttime"] - values["studydatetime"]).dt.total_seconds() / 3600
    values = values[elapsed_hours.abs() <= 24]
    wide = pd.pivot_table(
        values, values="valuenum", index="study_id", columns="feature_name", aggfunc="mean"
    ).reset_index()
    return cohort.merge(wide, on="study_id", how="left")


def _split_by_subject(data: pd.DataFrame) -> pd.DataFrame:
    subjects = data["subject_id"].drop_duplicates().to_numpy(copy=True)
    rng = np.random.RandomState(42)
    rng.shuffle(subjects)
    train_end = int(len(subjects) * 0.8)
    valid_end = train_end + int(len(subjects) * 0.1)
    train_subjects = set(subjects[:train_end])
    valid_subjects = set(subjects[train_end:valid_end])
    data = data.copy()
    data["split"] = np.where(
        data["subject_id"].isin(train_subjects),
        "train",
        np.where(data["subject_id"].isin(valid_subjects), "valid", "test"),
    )
    return data


def main() -> None:
    args = _parse_args()
    if args.target_count <= 0 or args.chunk_size <= 0:
        raise ValueError("--target-count and --chunk-size must be positive.")
    mimic_root = args.mimic_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    print("Aligning AP/PA chest radiographs to hospital admissions...")
    aligned = _align_cxr_to_admissions(mimic_root)
    print(f"Aligned CXR records: {len(aligned)}")
    admissions = pd.read_csv(
        _existing(mimic_root / "hosp", "admissions"),
        usecols=[
            "hadm_id", "admittime", "dischtime", "admission_type",
            "admission_location", "insurance", "marital_status",
        ],
    )
    patients = pd.read_csv(
        _existing(mimic_root / "hosp", "patients"),
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
    )
    cohort = aligned.merge(admissions, on="hadm_id", how="inner").merge(
        patients, on="subject_id", how="inner"
    )
    cohort["admittime"] = pd.to_datetime(cohort["admittime"])
    cohort["dischtime"] = pd.to_datetime(cohort["dischtime"])
    cohort["target_los_days"] = (
        cohort["dischtime"] - cohort["admittime"]
    ).dt.total_seconds() / (24 * 3600)
    cohort = cohort[
        (cohort["target_los_days"] > 0)
        & (cohort["studydatetime"] >= cohort["admittime"])
        & (cohort["studydatetime"] <= cohort["dischtime"])
    ].copy()
    cohort["admit_year"] = cohort["admittime"].dt.year
    cohort["age"] = cohort["anchor_age"] + (cohort["admit_year"] - cohort["anchor_year"])

    cohort = _extract_clinical_features(mimic_root, cohort, args.chunk_size)
    feature_names = list(VITALS) + list(LABS)
    cohort = cohort.dropna(subset=feature_names, how="any")
    if len(cohort) > args.target_count:
        cohort = cohort.sample(n=args.target_count, random_state=42).reset_index(drop=True)
    cohort["image_path"] = [
        _image_path(row.subject_id, row.study_id, row.dicom_id) for row in cohort.itertuples()
    ]
    cohort = _split_by_subject(cohort)
    output_columns = [
        "split", "subject_id", "study_id", "image_path", "gender", "age", "ViewPosition",
        "admission_type", "admission_location", "target_los_days", *feature_names,
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "los_dataset_100k_enriched.csv"
    cohort[output_columns].to_csv(csv_path, index=False)
    info = {
        "rows": int(len(cohort)),
        "split_counts": {key: int(value) for key, value in cohort["split"].value_counts().items()},
        "target_count_requested": args.target_count,
        "source_mimic_root": str(mimic_root),
        "feature_names": feature_names,
        "split_protocol": "subject-level 80/10/10, random_state=42",
    }
    (output_dir / "build_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    print(f"Wrote user-owned LOS candidate table: {csv_path}")


if __name__ == "__main__":
    main()
