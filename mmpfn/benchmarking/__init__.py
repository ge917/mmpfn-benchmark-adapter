"""Extensible benchmark adapters and orchestration for MMPFN."""

from .registry import DATASET_REGISTRY, DatasetSpec, get_dataset_spec, select_dataset_specs

__all__ = [
    "DATASET_REGISTRY",
    "DatasetSpec",
    "get_dataset_spec",
    "select_dataset_specs",
]
