"""Aggregate per-run metrics and compute MCR-style modality contributions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def summarize_results(output_root: str | Path) -> tuple[Path, Path]:
    output_root = Path(output_root).expanduser().resolve()
    rows = []
    for metrics_path in output_root.rglob("metrics.json"):
        if "summary" in metrics_path.parts:
            continue
        try:
            row = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        row["metrics_path"] = str(metrics_path)
        rows.append(row)

    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    results_path = summary_dir / "results.csv"
    transfer_path = summary_dir / "negative_transfer.csv"
    results = pd.DataFrame(rows)
    if results.empty:
        results.to_csv(results_path, index=False)
        pd.DataFrame().to_csv(transfer_path, index=False)
        return results_path, transfer_path

    preferred = [
        "benchmark",
        "dataset",
        "display_name",
        "task",
        "fold",
        "seed",
        "mode",
        "primary_metric",
        "primary_value",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "mae",
        "rmse",
        "r2",
        "n_train",
        "n_val",
        "n_test",
        "checkpoint",
        "metrics_path",
    ]
    ordered = [column for column in preferred if column in results] + [
        column for column in results.columns if column not in preferred
    ]
    results[ordered].sort_values(
        [column for column in ("benchmark", "dataset", "fold", "seed", "mode") if column in results]
    ).to_csv(results_path, index=False)

    transfer_rows = []
    group_columns = ["benchmark", "dataset", "display_name", "task", "fold", "seed", "primary_metric", "higher_is_better"]
    for keys, group in results.groupby(group_columns, dropna=False):
        mode_values = group.groupby("mode")["primary_value"].last().to_dict()
        secondary_mode = "image_only" if "image_only" in mode_values else "text_only"
        if not {"full", secondary_mode, "tabular_only"}.issubset(mode_values):
            continue
        info = dict(zip(group_columns, keys))
        full = float(mode_values["full"])
        secondary = float(mode_values[secondary_mode])
        tabular = float(mode_values["tabular_only"])
        if bool(info["higher_is_better"]):
            raw_image = full - tabular
            raw_tabular = full - secondary
            strongest_unimodal = max(secondary, tabular)
            full_vs_best = full - strongest_unimodal
        else:
            raw_image = tabular - full
            raw_tabular = secondary - full
            strongest_unimodal = min(secondary, tabular)
            full_vs_best = strongest_unimodal - full
        denominator = abs(raw_image) + abs(raw_tabular)
        image_pct = 0.0 if denominator == 0 else 100.0 * raw_image / denominator
        tabular_pct = 0.0 if denominator == 0 else 100.0 * raw_tabular / denominator
        row = {
                **info,
                "full": full,
                "tabular_only": tabular,
                "strongest_unimodal": strongest_unimodal,
                "full_vs_best_unimodal": full_vs_best,
                "tabular_contribution_pct": tabular_pct,
                "tabular_negative_transfer": raw_tabular < 0,
                "any_negative_transfer": raw_image < 0 or raw_tabular < 0,
                "secondary_modality": secondary_mode.removesuffix("_only"),
                "note": "MCR-style comparison from independently trained modality modes",
            }
        # Keep the existing image column names for backward-compatible VT- and
        # MulTaBench result tables; text datasets receive parallel text names.
        modality = secondary_mode.removesuffix("_only")
        row[secondary_mode] = secondary
        row[f"{modality}_contribution_pct"] = image_pct
        row[f"{modality}_negative_transfer"] = raw_image < 0
        transfer_rows.append(row)
    pd.DataFrame(transfer_rows).to_csv(transfer_path, index=False)
    return results_path, transfer_path
