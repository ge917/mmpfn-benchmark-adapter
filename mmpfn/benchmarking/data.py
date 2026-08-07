"""Unified prepared-split loader used by the benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from mmpfn.benchmarking.registry import DatasetSpec
from mmpfn.datasets.vtbench import VTBenchSplitDataset, _torch_load


class PreparedBenchmarkSplit:
    """A portable ``train/val/test.npz`` image-tabular split."""

    def __init__(self, root: str | Path, spec: DatasetSpec, split: str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.spec = spec
        self.dataset = spec.key
        self.split = split
        metadata_path = self.root / "metadata.json"
        split_path = self.root / f"{split}.npz"
        if not metadata_path.is_file() or not split_path.is_file():
            raise FileNotFoundError(
                f"Prepared dataset is incomplete at {self.root}; expected metadata.json and {split}.npz."
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(split_path, allow_pickle=False) as payload:
            self.x = np.asarray(payload["x"], dtype=np.float32)
            label_dtype = np.int64 if spec.task == "classification" else np.float32
            self.y = np.asarray(payload["y"], dtype=label_dtype).reshape(-1)
            raw_paths = np.asarray(payload["image_paths"]).astype(str).tolist()

        image_base = Path(self.metadata.get("image_base_dir", self.root)).expanduser()
        if not image_base.is_absolute():
            image_base = (self.root / image_base).resolve()
        self.image_paths = [
            Path(path) if Path(path).is_absolute() else (image_base / path).resolve()
            for path in raw_paths
        ]
        self.image_encoding = self.metadata.get("image_encoding", spec.image_encoding)
        self.categorical_features = [int(index) for index in self.metadata.get("categorical_indices", [])]
        self.field_lengths = self.metadata.get("field_lengths")
        self.embeddings: torch.Tensor | None = None

        if len(self.x) != len(self.y) or len(self.x) != len(self.image_paths):
            raise ValueError(
                f"Misaligned {spec.key}/{split}: x={len(self.x)}, y={len(self.y)}, "
                f"images={len(self.image_paths)}"
            )
        missing = [path for path in self.image_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{spec.key}/{split} references {len(missing)} missing images; first: {missing[0]}"
            )

    def _load_image(self, path: Path, image_size: int) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            image = np.asarray(np.load(path), dtype=np.float32)
            if image.ndim == 2:
                image = image[:, :, None]
            if image.ndim != 3:
                raise ValueError(f"Expected 2D/3D image array, got {image.shape}: {path}")
            if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
                image = np.moveaxis(image, 0, -1)
            if image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)
            if image.shape[-1] != 3:
                raise ValueError(f"Expected one or three channels, got {image.shape}: {path}")
            if self.image_encoding == "imagenet_normalized":
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                image = image * std + mean
            elif self.image_encoding == "uint8":
                image = image / 255.0
            image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
            pil_image = Image.fromarray(image)
        else:
            with Image.open(path) as opened:
                pil_image = ImageOps.exif_transpose(opened).convert("RGB")
        resized = pil_image.resize(
            (image_size, image_size),
            getattr(Image, "Resampling", Image).BILINEAR,
        )
        return np.asarray(resized, dtype=np.float32) / 255.0

    def get_embeddings(
        self,
        dino_checkpoint: str | Path,
        cache_path: str | Path,
        batch_size: int = 16,
        device: str = "cuda",
    ) -> torch.Tensor:
        cache_path = Path(cache_path)
        if cache_path.is_file():
            self.embeddings = _torch_load(cache_path)
            if len(self.embeddings) != len(self.x):
                raise ValueError(f"Embedding cache does not match {self.dataset}/{self.split}: {cache_path}")
            return self.embeddings
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("DINOv2 embedding extraction requires an available CUDA device.")

        from mmpfn.models.dino_v2.models.vision_transformer import vit_base

        encoder = vit_base(
            patch_size=14,
            img_size=518,
            init_values=1.0,
            num_register_tokens=0,
            block_chunks=0,
        )
        encoder.load_state_dict(_torch_load(Path(dino_checkpoint)))
        encoder = encoder.to(device).eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(self.image_paths), batch_size):
                paths = self.image_paths[start : start + batch_size]
                images = np.stack([self._load_image(path, 336) for path in paths])
                batch = torch.from_numpy(np.moveaxis(images, -1, 1)).unsqueeze(1)
                batch = batch.flatten(0, 1).to(device, non_blocking=True)
                features = encoder.forward_features(batch)["x_norm_clstoken"]
                outputs.append(features.reshape(-1, 1, features.shape[-1]).cpu())
        self.embeddings = torch.cat(outputs, dim=0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.embeddings, cache_path)
        return self.embeddings


def load_benchmark_splits(spec: DatasetSpec, root: str | Path) -> dict[str, Any]:
    """Load portable prepared files, with a compatibility path for current VT exports."""
    root = Path(root).expanduser().resolve()
    if (root / "metadata.json").is_file() and (root / "train.npz").is_file():
        return {split: PreparedBenchmarkSplit(root, spec, split) for split in ("train", "val", "test")}
    if spec.benchmark == "vtbench" and spec.legacy_vtbench_name:
        return {
            split: VTBenchSplitDataset(
                root,
                spec.legacy_vtbench_name,
                split,
                spec.image_encoding,
            )
            for split in ("train", "val", "test")
        }
    raise FileNotFoundError(
        f"No prepared data found for {spec.key} at {root}. "
        "Expected metadata.json plus train.npz/val.npz/test.npz."
    )
