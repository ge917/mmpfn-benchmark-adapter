from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from mmpfn.scripts_finetune_mm.constant_utils import SupportedDevice, TaskType
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from mmpfn.scripts_finetune_mm.metric_utils.ag_metrics import Scorer
    from mmpfn.models.mmpfn.model.transformer import PerFeatureTransformer


def create_val_data(
    *,
    X_train: pd.DataFrame | np.ndarray,
    image_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    rng: np.random.RandomState,
    n_samples: int,
    is_classification: bool,
) -> tuple[
    pd.DataFrame | np.ndarray,
    pd.DataFrame | np.ndarray,
    pd.DataFrame | np.ndarray,
    pd.DataFrame | np.ndarray,
    pd.Series | np.ndarray,
    pd.Series | np.ndarray,
]:
    # Split data ourselves
    if n_samples < 10000:
        test_size = 0.2#0.33
    elif n_samples < 500000:
        test_size = 0.2
    elif n_samples < 1000000:
        test_size = 0.1
    else:
        test_size = 0.05
        
    if image_train is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=test_size,
            random_state=rng,
            stratify=y_train if is_classification else None,
        )
        return X_train, X_val, None, None, y_train, y_val
    elif X_train is None:
        image_train, image_val, y_train, y_val = train_test_split(
            image_train,
            y_train,
            test_size=test_size,
            random_state=rng,
            stratify=y_train if is_classification else None,
        )
        return None, None, image_train, image_val, y_train, y_val    
    X_train, X_val, image_train, image_val, y_train, y_val = train_test_split(
        X_train,
        image_train,
        y_train,
        test_size=test_size,
        random_state=rng,
        stratify=y_train if is_classification else None,
    )
    return X_train, X_val, image_train, image_val, y_train, y_val


def validate_tabpfn(
    *,
    X_train: torch.Tensor,  # (n_samples, batch_size, n_features)
    image_train: torch.Tensor,  # (n_samples, batch_size, n_features)
    y_train: torch.Tensor,  # (n_samples, batch_size, 1)
    X_val: torch.Tensor,  # (n_samples, batch_size, 1)
    image_val: torch.Tensor,  # (n_samples, batch_size, n_features)
    y_val: torch.Tensor,  # (n_samples, batch_size, 1)
    validation_metric: Scorer,
    model: PerFeatureTransformer,
    model_forward_fn: Callable,
    task_type: TaskType,
    device: SupportedDevice,
    validation_chunk_size: int | None = None,
) -> float:
    """Validate the TabPFN model and return a loss (lower is better).

    This code assumes that batch_size for validation is 1. Otherwise,
    need to write a loop, I guess?

    ``validation_chunk_size`` splits only validation rows. Every chunk keeps the
    complete training context, and predictions are concatenated before the metric
    is computed.
    """
    if X_train is not None:
        X_train = X_train.to(device)

    y_train = y_train.to(device)

    if image_train is not None:
        image_train = image_train.to(device)

    n_validation_rows = len(y_val)
    chunk_size = validation_chunk_size or n_validation_rows
    if chunk_size < 1:
        raise ValueError("validation_chunk_size must be positive when provided.")

    # MMPFN conditions every validation row on the same training context.  Splitting
    # only the validation rows therefore preserves the predictions and final metric,
    # while avoiding a train+validation sequence that is too long for CUDA attention.
    prediction_chunks = []
    for start in range(0, n_validation_rows, chunk_size):
        stop = min(start + chunk_size, n_validation_rows)
        X_val_chunk = X_val[start:stop].to(device) if X_val is not None else None
        image_val_chunk = image_val[start:stop].to(device) if image_val is not None else None
        pred_logits = model_forward_fn(
            model=model,
            X_train=X_train,
            image_train=image_train,
            y_train=y_train,
            X_test=X_val_chunk,
            image_test=image_val_chunk,
            forward_for_validation=True,
        )
        prediction_chunks.append(pred_logits.float().cpu())

    pred_logits = torch.cat(prediction_chunks, dim=0)

    match task_type:
        case TaskType.REGRESSION:
            y_pred = pred_logits.float().flatten().cpu().detach().numpy()
            y_true = y_val.float().flatten().cpu().detach().numpy()
        case TaskType.BINARY_CLASSIFICATION:
            # TODO: check that this works / is exhaustive.
            if validation_metric.needs_threshold or validation_metric.needs_proba:
                y_pred = (
                    torch.nn.functional.sigmoid(pred_logits[:, 0, 1])
                    .cpu()
                    .detach()
                    .numpy()
                )
            else:
                # Required to get the correct classes for the metrics
                y_pred = (
                    torch.nn.functional.softmax(pred_logits[:, 0, :], dim=-1)
                    .cpu()
                    .detach()
                    .numpy()
                )
            y_true = y_val.long().flatten().cpu().detach().numpy()
        case TaskType.MULTICLASS_CLASSIFICATION:
            y_pred = (
                torch.nn.functional.softmax(pred_logits[:, 0, :], dim=-1)
                .cpu()
                .detach()
                .numpy()
            )
            y_true = y_val.long().flatten().cpu().detach().numpy()
        case _:
            raise ValueError(f"Task type {task_type} not supported.")

    score = validation_metric(y_true=y_true, y_pred=y_pred)

    if X_train is not None:
        X_train.cpu()
    
    y_train.cpu()
    
    if image_train is not None:
        image_train.cpu()
    
    return validation_metric.convert_score_to_error(score=score)
