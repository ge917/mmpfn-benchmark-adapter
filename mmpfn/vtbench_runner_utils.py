"""Shared, deterministic size controls for VT-Bench MMPFN runners."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def fixed_subset_indices(
    y: np.ndarray,
    maximum: int | None,
    *,
    seed: int,
    task: str,
) -> np.ndarray:
    """Pick a reproducible subset while preserving class/target coverage."""
    indices = np.arange(len(y))
    if maximum is None or maximum <= 0 or len(indices) <= maximum:
        return indices
    if maximum < 2:
        raise ValueError("The MMPFN context subset must contain at least two rows.")

    if task == "classification":
        strata: np.ndarray | None = y
    elif task == "regression":
        # Rank first so equal values never make qcut collapse bins.  This keeps
        # the long-tailed RR target represented without changing its values.
        n_bins = min(10, maximum, len(indices))
        strata = pd.qcut(pd.Series(y).rank(method="first"), q=n_bins, labels=False).to_numpy()
    else:
        raise ValueError(f"Unsupported task: {task}")

    try:
        chosen, _ = train_test_split(
            indices, train_size=maximum, random_state=seed, stratify=strata
        )
    except ValueError:
        # Fallback remains deterministic for unusual label distributions.
        chosen = np.random.default_rng(seed).choice(indices, size=maximum, replace=False)
    return np.sort(chosen)


def subset_embeddings(embeddings: torch.Tensor | None, indices: np.ndarray) -> torch.Tensor | None:
    if embeddings is None:
        return None
    return embeddings.index_select(0, torch.as_tensor(indices, dtype=torch.long))


def predict_in_chunks(
    model: Any,
    x: np.ndarray | None,
    images: torch.Tensor | None,
    *,
    batch_size: int,
) -> np.ndarray:
    """Predict test rows in chunks while retaining the model's fitted context."""
    n_rows = len(images) if images is not None else len(x) if x is not None else 0
    if n_rows == 0:
        raise ValueError("At least one modality is required for prediction.")
    outputs = []
    for start in range(0, n_rows, batch_size):
        stop = min(start + batch_size, n_rows)
        outputs.append(
            model.predict(
                pd.DataFrame(x[start:stop]) if x is not None else None,
                images[start:stop] if images is not None else None,
            )
        )
    return np.concatenate(outputs)


def predict_proba_in_chunks(
    model: Any,
    x: np.ndarray | None,
    images: torch.Tensor | None,
    *,
    batch_size: int,
) -> np.ndarray:
    """Predict class probabilities in chunks while retaining fitted context."""
    n_rows = len(images) if images is not None else len(x) if x is not None else 0
    if n_rows == 0:
        raise ValueError("At least one modality is required for prediction.")
    outputs = []
    for start in range(0, n_rows, batch_size):
        stop = min(start + batch_size, n_rows)
        outputs.append(
            model.predict_proba(
                pd.DataFrame(x[start:stop]) if x is not None else None,
                images[start:stop] if images is not None else None,
            )
        )
    return np.concatenate(outputs)
