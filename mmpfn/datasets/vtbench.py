"""Adapters for VT-Bench discriminative prediction splits.

The adapter deliberately consumes VT-Bench's exported files instead of its raw
datasets.  This keeps the split, tabular preprocessing, and image eligibility
identical to the benchmark while leaving the VT-Bench code untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image


SplitName = Literal["train", "val", "test"]
ImageEncoding = Literal["zero_one", "uint8", "imagenet_normalized"]


@dataclass(frozen=True)
class _SplitFiles:
    features: str
    labels: str
    paths: str


_FILES = {
    "adoption": {
        "train": _SplitFiles("features_train.csv", "labels_train.pt", "paths_train.pt"),
        "val": _SplitFiles("features_valid.csv", "labels_valid.pt", "paths_valid.pt"),
        "test": _SplitFiles("features_test.csv", "labels_test.pt", "paths_test.pt"),
    },
    "breast": {
        "train": _SplitFiles("train_features.csv", "train_labels.pt", "train_paths.pt"),
        "val": _SplitFiles("val_features.csv", "val_labels.pt", "val_paths.pt"),
        "test": _SplitFiles("test_features.csv", "test_labels.pt", "test_paths.pt"),
    },
    "pawpularity": {
        "train": _SplitFiles("train_features.csv", "train_labels.pt", "train_paths.pt"),
        "val": _SplitFiles("val_features.csv", "val_labels.pt", "val_paths.pt"),
        "test": _SplitFiles("test_features.csv", "test_labels.pt", "test_paths.pt"),
    },
    "pneumonia": {
        "train": _SplitFiles("train_features.csv", "train_labels.pt", "train_paths.pt"),
        "val": _SplitFiles("valid_features.csv", "valid_labels.pt", "valid_paths.pt"),
        "test": _SplitFiles("test_features.csv", "test_labels.pt", "test_paths.pt"),
    },
    "rr": {
        "train": _SplitFiles("train_features.csv", "train_labels.pt", "train_paths.pt"),
        "val": _SplitFiles("val_features.csv", "val_labels.pt", "val_paths.pt"),
        "test": _SplitFiles("test_features.csv", "test_labels.pt", "test_paths.pt"),
    },
    "infarction": {
        "train": _SplitFiles("train_features.csv", "train_labels.pt", "train_paths.pt"),
        "val": _SplitFiles("val_features.csv", "val_labels.pt", "val_paths.pt"),
        "test": _SplitFiles("test_features.csv", "test_labels.pt", "test_paths.pt"),
    },
}


def _torch_load(path: Path):
    """Load local VT-Bench tensors on both pre- and post-2.6 PyTorch."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location="cpu")


class VTBenchSplitDataset:
    """One already-preprocessed VT-Bench split in MMPFN-ready form.

    Parameters
    ----------
    root:
        The directory containing the exported VT-Bench feature files for one
        dataset, not the repository root.
    dataset:
        ``"adoption"`` or ``"breast"``.
    split:
        One of VT-Bench's ``train``, ``val``, or ``test`` splits.
    image_encoding:
        How VT-Bench stored the ``.npy`` images.  Adoption writes [0, 1]
        arrays; the Breast preprocessing notebook writes ImageNet-normalized
        arrays, which are converted back to [0, 1] before DINOv2 embedding.
    """

    def __init__(
        self,
        root: str | Path,
        dataset: Literal["adoption", "breast", "pawpularity", "pneumonia", "rr", "infarction"],
        split: SplitName,
        image_encoding: ImageEncoding,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.dataset = dataset
        self.split = split
        self.image_encoding = image_encoding

        try:
            files = _FILES[dataset][split]
        except KeyError as error:
            raise ValueError(f"Unsupported VT-Bench dataset/split: {dataset}/{split}") from error

        self.x = pd.read_csv(self.root / files.features, header=None).to_numpy(dtype=np.float32)
        label_dtype = np.float32 if dataset in ("pawpularity", "rr") else np.int64
        self.y = np.asarray(_torch_load(self.root / files.labels), dtype=label_dtype).reshape(-1)
        self.image_paths = [Path(path) for path in _torch_load(self.root / files.paths)]

        sample_count = len(self.x)
        if len(self.y) != sample_count or len(self.image_paths) != sample_count:
            raise ValueError(
                f"VT-Bench {dataset}/{split} is misaligned: "
                f"{sample_count} feature rows, {len(self.y)} labels, "
                f"{len(self.image_paths)} image paths."
            )
        missing_paths = [str(path) for path in self.image_paths if not path.is_file()]
        if missing_paths:
            example = missing_paths[0]
            raise FileNotFoundError(
                f"VT-Bench {dataset}/{split} references {len(missing_paths)} missing images; "
                f"first missing path: {example}"
            )

        field_lengths_path = self.root / "tabular_lengths.pt"
        self.field_lengths = list(_torch_load(field_lengths_path)) if field_lengths_path.is_file() else None
        self.categorical_features = self._categorical_features()
        self.embeddings: torch.Tensor | None = None

    def _categorical_features(self) -> list[int]:
        if self.field_lengths is None:
            return []
        if len(self.field_lengths) != self.x.shape[1]:
            raise ValueError(
                f"tabular_lengths.pt has {len(self.field_lengths)} entries but "
                f"{self.split} has {self.x.shape[1]} features."
            )
        return [index for index, cardinality in enumerate(self.field_lengths) if cardinality > 1]

    def _load_image(self, path: Path, image_size: int) -> np.ndarray:
        image = np.asarray(np.load(path), dtype=np.float32)
        if image.ndim == 2:
            image = image[:, :, None]
        if image.ndim != 3:
            raise ValueError(f"Expected a 2D or 3D image array, received {image.shape} at {path}")
        if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.moveaxis(image, 0, -1)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.shape[-1] != 3:
            raise ValueError(f"Expected one or three image channels, received {image.shape} at {path}")

        if self.image_encoding == "imagenet_normalized":
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            image = image * std + mean
        elif self.image_encoding == "uint8":
            image = image / 255.0
        image = np.clip(image, 0.0, 1.0)
        image_uint8 = np.rint(image * 255.0).astype(np.uint8)
        return np.asarray(
            Image.fromarray(image_uint8).resize(
                (image_size, image_size),
                getattr(Image, "Resampling", Image).BILINEAR,
            ),
            dtype=np.float32,
        ) / 255.0

    def get_images(self, image_size: int = 336) -> torch.Tensor:
        """Return DINOv2-ready RGB images with shape ``[N, 1, 3, H, W]``."""
        images = np.stack([self._load_image(path, image_size) for path in self.image_paths])
        self.images = torch.from_numpy(np.moveaxis(images, -1, 1)).unsqueeze(1)
        return self.images

    def get_embeddings(
        self,
        dino_checkpoint: str | Path,
        cache_path: str | Path,
        batch_size: int = 16,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Load or create cached DINOv2 CLS embeddings for this exact split."""
        cache_path = Path(cache_path)
        if cache_path.is_file():
            self.embeddings = _torch_load(cache_path)
            if len(self.embeddings) != len(self.x):
                raise ValueError(f"Embedding cache does not match {self.dataset}/{self.split}: {cache_path}")
            return self.embeddings

        if not torch.cuda.is_available() and device.startswith("cuda"):
            raise RuntimeError("DINOv2 embedding extraction requires CUDA, but no CUDA device is available.")

        from mmpfn.models.dino_v2.models.vision_transformer import vit_base

        encoder = vit_base(
            patch_size=14,
            img_size=518,
            init_values=1.0,
            num_register_tokens=0,
            block_chunks=0,
        )
        state_dict = _torch_load(Path(dino_checkpoint))
        encoder.load_state_dict(state_dict)
        encoder = encoder.to(device).eval()

        all_embeddings = []
        with torch.no_grad():
            for start in range(0, len(self.image_paths), batch_size):
                batch_paths = self.image_paths[start : start + batch_size]
                batch_images = np.stack([self._load_image(path, image_size=336) for path in batch_paths])
                batch = torch.from_numpy(np.moveaxis(batch_images, -1, 1)).unsqueeze(1).to(device, non_blocking=True)
                batch = batch.flatten(0, 1)
                features = encoder.forward_features(batch)["x_norm_clstoken"]
                all_embeddings.append(features.reshape(-1, 1, features.shape[-1]).cpu())
        self.embeddings = torch.cat(all_embeddings, dim=0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.embeddings, cache_path)
        return self.embeddings
