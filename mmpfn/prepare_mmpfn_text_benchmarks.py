"""Prepare the original MMPFN paper's Airbnb, Salary, and Cloth datasets.

The field selection, target construction, and text concatenation follow the
legacy dataset classes in :mod:`mmpfn.datasets`.  The output uses the current
portable ``metadata.json + train/val/test.npz`` benchmark format.  In contrast
to the legacy implementation, category maps and numeric imputers are learned
from the training split only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from mmpfn.benchmarking.registry import DatasetSpec, get_dataset_spec, select_dataset_specs


DEFAULT_ROOT = Path("/mnt/hdd/zhangyg/projects/tab/benchmark_data/mmpfn_paper")


def _parse_mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected DATASET=PATH, got: {value}")
        key, path = value.split("=", 1)
        result[get_dataset_spec(key).key] = Path(path).expanduser().resolve()
    return result


def _required(frame: pd.DataFrame, columns: list[str], dataset: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{dataset} source is missing required columns: {missing}")


def _airbnb(root: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    frame = pd.read_csv(root / "cleansed_listings_dec18.csv")
    categorical = [
        "host_location", "host_since_year", "host_is_superhost", "host_neighborhood",
        "host_has_profile_pic", "host_identity_verified", "neighborhood", "city",
        "smart_location", "suburb", "state", "is_location_exact", "property_type",
        "room_type", "bed_type", "instant_bookable", "cancellation_policy",
        "require_guest_profile_picture", "require_guest_phone_verification",
        "host_response_time", "calendar_updated", "host_verifications", "last_review_year",
    ]
    numerical = [
        "host_response_rate", "latitude", "longitude", "accommodates", "bathrooms",
        "bedrooms", "beds", "security_deposit", "cleaning_fee", "guests_included",
        "extra_people", "minimum_nights", "maximum_nights", "availability_30",
        "availability_60", "availability_90", "availability_365", "number_of_reviews",
        "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
        "review_scores_checkin", "review_scores_communication", "review_scores_location",
        "review_scores_value", "calculated_host_listings_count", "reviews_per_month",
    ]
    _required(frame, ["price", "name", "summary", "description", "host_since", "last_review", *categorical, *numerical], "Airbnb")
    frame = frame.loc[~(frame["summary"].isna() & frame["description"].isna())].copy()
    for column in ("name", "summary", "description"):
        frame[column] = frame[column].fillna("").astype(str)
    frame["text"] = frame["name"] + " " + frame["summary"] + " " + frame["description"]
    frame["host_since_year"] = frame["host_since"].astype(str).str.extract(r"(\d{4})", expand=False)
    frame["last_review_year"] = frame["last_review"].astype(str).str.extract(r"(\d{4})", expand=False)
    frame["host_response_rate"] = frame["host_response_rate"].astype(str).str.replace("%", "", regex=False)
    frame["host_response_rate"] = pd.to_numeric(frame["host_response_rate"], errors="coerce")
    price = pd.to_numeric(frame["price"], errors="coerce")
    edges = np.quantile(price.dropna(), q=np.arange(11) / 10)
    edges[0] = 0
    if len(np.unique(edges)) != len(edges):
        raise ValueError("Airbnb price quantiles are not unique; cannot reproduce the ten-bin target.")
    target = pd.cut(price, bins=edges, labels=np.arange(10), include_lowest=True)
    selected = frame[categorical + numerical + ["text"]].copy()
    keep = selected.notna().all(axis=1) & target.notna()
    return selected.loc[keep].reset_index(drop=True), target.loc[keep].astype(int).reset_index(drop=True), frame.loc[keep, "text"].reset_index(drop=True), categorical, numerical


def _salary(root: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    frame = pd.read_csv(root / "train.csv")
    categorical = ["location", "company_name_encoded", "job_type"]
    numerical = ["experience_int"]
    _required(frame, ["salary", "experience", "job_description", "job_desig", "key_skills", *categorical], "Salary")
    frame["experience_int"] = pd.to_numeric(frame["experience"].astype(str).str.split("-").str[0], errors="coerce")
    for column in ("job_description", "job_desig", "key_skills"):
        frame[column] = frame[column].fillna("").astype(str)
    frame["text"] = frame["job_description"] + " " + frame["job_desig"] + " " + frame["key_skills"]
    selected = frame[categorical + numerical + ["text"]].copy()
    raw_target = frame["salary"]
    keep = selected.notna().all(axis=1) & raw_target.notna()
    labels = LabelEncoder().fit_transform(raw_target.loc[keep].astype(str))
    return selected.loc[keep].reset_index(drop=True), pd.Series(labels), frame.loc[keep, "text"].reset_index(drop=True), categorical, numerical


def _cloth(root: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    frame = pd.read_csv(root / "Womens Clothing E-Commerce Reviews.csv")
    categorical = ["Division Name", "Department Name", "Class Name"]
    numerical = ["Age", "Positive Feedback Count"]
    _required(frame, ["Rating", "Title", "Review Text", *categorical, *numerical], "Cloth")
    frame["Title"] = frame["Title"].fillna("").astype(str)
    frame["Review Text"] = frame["Review Text"].fillna("").astype(str)
    frame["text"] = frame["Title"] + " " + frame["Review Text"]
    selected = frame[categorical + numerical + ["text"]].copy()
    target = pd.to_numeric(frame["Rating"], errors="coerce") - 1
    keep = selected.notna().all(axis=1) & target.notna()
    return selected.loc[keep].reset_index(drop=True), target.loc[keep].astype(int).reset_index(drop=True), frame.loc[keep, "text"].reset_index(drop=True), categorical, numerical


BUILDERS = {
    "mmpfn_airbnb": _airbnb,
    "mmpfn_salary": _salary,
    "mmpfn_cloth": _cloth,
}


def _split_indices(y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    positions = np.arange(len(y))
    try:
        train_val, test = train_test_split(positions, test_size=0.1, random_state=seed, stratify=y)
        train, val = train_test_split(
            train_val, test_size=1 / 9, random_state=seed + 1, stratify=y[train_val]
        )
    except ValueError as error:
        # Salary labels can be very sparse in some public releases.  Preserve
        # determinism and still create the unified 80/10/10 protocol.
        print(f"WARNING: stratified split unavailable ({error}); using random 80/10/10 split.")
        train_val, test = train_test_split(positions, test_size=0.1, random_state=seed)
        train, val = train_test_split(train_val, test_size=1 / 9, random_state=seed + 1)
    return {"train": train, "val": val, "test": test}


def _encode(
    frame: pd.DataFrame,
    splits: dict[str, np.ndarray],
    categorical: list[str],
    numerical: list[str],
) -> tuple[np.ndarray, list[int]]:
    encoded = pd.DataFrame(index=frame.index)
    field_lengths: list[int] = []
    train = frame.iloc[splits["train"]]
    for column in categorical:
        train_values = train[column].fillna("<MISSING>").astype(str)
        categories = sorted(train_values.unique())
        mapping = {value: index for index, value in enumerate(categories)}
        encoded[column] = frame[column].fillna("<MISSING>").astype(str).map(mapping).fillna(-1).astype(np.float32)
        field_lengths.append(max(len(categories), 1))
    if numerical:
        numeric = frame[numerical].apply(pd.to_numeric, errors="coerce")
        train_numeric = numeric.iloc[splits["train"]]
        medians = train_numeric.median().fillna(0.0)
        numeric = numeric.fillna(medians)
        scaler = StandardScaler().fit(numeric.iloc[splits["train"]])
        encoded[numerical] = scaler.transform(numeric)
    return encoded[categorical + numerical].to_numpy(dtype=np.float32), field_lengths + [1] * len(numerical)


def prepare_one(spec: DatasetSpec, source_root: Path, output: Path, seed: int, force: bool) -> None:
    expected = [output / "metadata.json", *(output / f"{split}.npz" for split in ("train", "val", "test"))]
    if not force and all(path.is_file() for path in expected):
        print(f"Prepared: {spec.key} -> {output}")
        return
    frame, target, text, categorical, numerical = BUILDERS[spec.key](source_root)
    y = target.to_numpy(dtype=np.int64)
    splits = _split_indices(y, seed)
    x, field_lengths = _encode(frame, splits, categorical, numerical)
    output.mkdir(parents=True, exist_ok=True)
    for name, indices in splits.items():
        np.savez_compressed(
            output / f"{name}.npz",
            x=x[indices],
            y=y[indices],
            texts=text.iloc[indices].astype(str).to_numpy(dtype=str),
        )
    metadata = {
        "dataset": spec.key,
        "secondary_modality": "text",
        "text_model_id": spec.text_model_id,
        "categorical_indices": list(range(len(categorical))),
        "field_lengths": field_lengths,
        "split_protocol": "stratified 80/10/10, random_state=" + str(seed),
        "source_format": "original MMPFN paper dataset fields and text concatenation",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared {spec.key}: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["mmpfn_paper"])
    parser.add_argument("--source-root", action="append", default=[], metavar="DATASET=PATH", required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    source_roots = _parse_mapping(args.source_root)
    specs = select_dataset_specs(args.datasets)
    for spec in specs:
        if spec.benchmark != "mmpfn_paper":
            raise ValueError(f"{spec.key} is not an original-MMPFN text dataset.")
        if spec.key not in source_roots:
            raise ValueError(f"Missing --source-root {spec.key}=PATH")
        for fold in args.folds:
            prepare_one(spec, source_roots[spec.key], args.data_root / spec.key / f"fold_{fold}", seed=42 + fold, force=args.force)


if __name__ == "__main__":
    main()
