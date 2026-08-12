"""Frozen visual encoders shared by benchmark and encoder-comparison runs.

The original MMPFN implementation consumes one 768-dimensional visual token
per example.  This module keeps encoder extraction separate from MMPFN: it
can cache features from several frozen encoders and, when necessary, projects
them to the 768 dimensions expected by MMPFN.  The projection is fit on the
training split only and then reused unchanged for validation and test data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA


ImageEncoderName = Literal["dino_v2", "dino_v3", "clip_vitl14", "resnet50", "vit_b16"]
IMAGE_ENCODERS: tuple[ImageEncoderName, ...] = (
    "dino_v2",
    "dino_v3",
    "clip_vitl14",
    "resnet50",
    "vit_b16",
)

# The selected CLIP model, DINOv2-B and DINOv3-B all produce 768-dimensional
# vectors.  ResNet-50 produces 2048 dimensions and is reduced below.
DEFAULT_IMAGE_MODEL_IDS = {
    "dino_v3": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "clip_vitl14": "openai/clip-vit-large-patch14",
}
MMPFN_IMAGE_DIM = 768


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location="cpu")


def _as_pil(images: Sequence[np.ndarray]) -> list[Image.Image]:
    converted: list[Image.Image] = []
    for image in images:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
        converted.append(Image.fromarray(array).convert("RGB"))
    return converted


def _extract_dino_v2(
    load_image: Callable[[Path, int], np.ndarray],
    paths: Sequence[Path],
    checkpoint: Path,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    from mmpfn.models.dino_v2.models.vision_transformer import vit_base

    encoder = vit_base(
        patch_size=14,
        img_size=518,
        init_values=1.0,
        num_register_tokens=0,
        block_chunks=0,
    )
    encoder.load_state_dict(_torch_load(checkpoint))
    encoder = encoder.to(device).eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_images = np.stack([load_image(path, 336) for path in paths[start : start + batch_size]])
            batch = torch.from_numpy(np.moveaxis(batch_images, -1, 1)).to(device, non_blocking=True)
            features = encoder.forward_features(batch)["x_norm_clstoken"]
            outputs.append(features.cpu())
    return torch.cat(outputs, dim=0)


def _extract_transformers(
    encoder_name: ImageEncoderName,
    load_image: Callable[[Path, int], np.ndarray],
    paths: Sequence[Path],
    model_id: str,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

    if encoder_name == "clip_vitl14":
        processor = CLIPProcessor.from_pretrained(model_id)
        encoder = CLIPModel.from_pretrained(model_id).to(device).eval()
    else:
        processor = AutoImageProcessor.from_pretrained(model_id)
        encoder = AutoModel.from_pretrained(model_id).to(device).eval()

    outputs = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            # The processor owns the model-specific resize, crop and
            # normalization.  Loading at 518 avoids an unnecessary loss of
            # detail before the processor applies its own transform.
            images = _as_pil([load_image(path, 518) for path in paths[start : start + batch_size]])
            pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(
                device, non_blocking=True
            )
            if encoder_name == "clip_vitl14":
                features = encoder.get_image_features(pixel_values=pixel_values)
            else:
                model_output = encoder(pixel_values=pixel_values)
                features = model_output.last_hidden_state[:, 0, :]
            outputs.append(features.cpu())
    return torch.cat(outputs, dim=0)


def _extract_torchvision(
    encoder_name: ImageEncoderName,
    load_image: Callable[[Path, int], np.ndarray],
    paths: Sequence[Path],
    batch_size: int,
    device: str,
) -> torch.Tensor:
    from torchvision.models import (
        ResNet50_Weights,
        ViT_B_16_Weights,
        resnet50,
        vit_b_16,
    )

    if encoder_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        encoder = resnet50(weights=weights)
        encoder.fc = torch.nn.Identity()
    else:
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        encoder = vit_b_16(weights=weights)
        encoder.heads = torch.nn.Identity()
    transform = weights.transforms()
    encoder = encoder.to(device).eval()

    outputs = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            images = np.stack([load_image(path, 518) for path in paths[start : start + batch_size]])
            batch = torch.from_numpy(np.moveaxis(images, -1, 1))
            # torchvision's official weight transform applies resize/crop and
            # ImageNet normalization to the whole tensor batch.
            features = encoder(transform(batch).to(device, non_blocking=True))
            outputs.append(features.cpu())
    return torch.cat(outputs, dim=0)


def extract_image_embeddings(
    *,
    encoder_name: ImageEncoderName,
    image_paths: Sequence[Path],
    load_image: Callable[[Path, int], np.ndarray],
    cache_path: str | Path,
    dino_checkpoint: str | Path | None,
    image_model_id: str | None,
    batch_size: int = 16,
    device: str = "cuda",
) -> torch.Tensor:
    """Return frozen encoder features with shape ``[n_examples, 1, dim]``."""
    cache_path = Path(cache_path)
    if cache_path.is_file():
        embeddings = _torch_load(cache_path)
        if len(embeddings) != len(image_paths):
            raise ValueError(f"Embedding cache does not match {cache_path}: {len(embeddings)} != {len(image_paths)}")
        return embeddings.float()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Visual embedding extraction requires CUDA, but CUDA is unavailable.")
    if not image_paths:
        raise ValueError("Cannot extract image embeddings from an empty split.")

    if encoder_name == "dino_v2":
        if dino_checkpoint is None or not Path(dino_checkpoint).is_file():
            raise FileNotFoundError("dino_v2 requires --dino-checkpoint pointing to dinov2_vitb14_pretrain.pth.")
        raw = _extract_dino_v2(load_image, image_paths, Path(dino_checkpoint), batch_size, device)
    elif encoder_name in {"dino_v3", "clip_vitl14"}:
        model_id = image_model_id or DEFAULT_IMAGE_MODEL_IDS[encoder_name]
        raw = _extract_transformers(encoder_name, load_image, image_paths, model_id, batch_size, device)
    elif encoder_name in {"resnet50", "vit_b16"}:
        raw = _extract_torchvision(encoder_name, load_image, image_paths, batch_size, device)
    else:  # defensive guard for CLI values passed programmatically
        raise ValueError(f"Unsupported image encoder: {encoder_name}")

    embeddings = raw.float().unsqueeze(1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, cache_path)
    return embeddings


def project_for_mmpfn(
    embeddings: dict[str, torch.Tensor],
    projection_path: str | Path,
    *,
    output_dim: int = MMPFN_IMAGE_DIM,
    max_fit_rows: int = 10_000,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Project raw frozen features to MMPFN's 768-dimensional token space.

    The projector is fitted from ``embeddings['train']`` only.  ``identity``
    preserves the historical DINOv2 path exactly; ``pca`` is used for wider
    encoders such as ResNet-50; and ``pad`` is a deterministic fallback for a
    narrower model.
    """
    projection_path = Path(projection_path)
    train = embeddings["train"].detach().cpu().float()
    if train.ndim != 3 or train.shape[1] != 1:
        raise ValueError(f"Expected [N, 1, D] embeddings, got {tuple(train.shape)}")
    input_dim = int(train.shape[-1])

    if projection_path.is_file():
        state = _torch_load(projection_path)
        if int(state["input_dim"]) != input_dim or int(state["output_dim"]) != output_dim:
            raise ValueError(f"Projection cache is incompatible with current embeddings: {projection_path}")
    elif input_dim == output_dim:
        state = {"method": "identity", "input_dim": input_dim, "output_dim": output_dim}
        projection_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, projection_path)
    elif input_dim < output_dim:
        state = {"method": "pad", "input_dim": input_dim, "output_dim": output_dim}
        projection_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, projection_path)
    else:
        values = train[:, 0, :].numpy()
        if len(values) > max_fit_rows:
            rng = np.random.default_rng(seed)
            values = values[rng.choice(len(values), size=max_fit_rows, replace=False)]
        # A tiny smoke-test split can contain fewer than 768 rows.  PCA then
        # has fewer usable axes; retain all it has and zero-pad afterwards.
        pca_dim = min(output_dim, input_dim, len(values))
        pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=seed)
        pca.fit(values)
        state = {
            "method": "pca",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "mean": torch.from_numpy(pca.mean_.astype(np.float32)),
            "components": torch.from_numpy(pca.components_.astype(np.float32)),
            "fit_rows": int(len(values)),
            "pca_dim": int(pca_dim),
        }
        projection_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, projection_path)

    method = state["method"]
    projected: dict[str, torch.Tensor] = {}
    for split, value in embeddings.items():
        matrix = value.detach().cpu().float()[:, 0, :]
        if method == "identity":
            result = matrix
        elif method == "pad":
            result = torch.nn.functional.pad(matrix, (0, output_dim - input_dim))
        elif method == "pca":
            result = (matrix - state["mean"]) @ state["components"].T
            if result.shape[1] < output_dim:
                result = torch.nn.functional.pad(result, (0, output_dim - result.shape[1]))
        else:
            raise ValueError(f"Unknown projection method in {projection_path}: {method}")
        projected[split] = result.unsqueeze(1).contiguous()
    return projected
