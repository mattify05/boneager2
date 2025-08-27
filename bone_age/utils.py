# bone_age/utils.py
from __future__ import annotations
from pathlib import Path
import os
import hashlib
from typing import Iterable, Tuple, Optional, List, Dict

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------- Files & integrity ----------

def file_exists(path: str | os.PathLike) -> bool:
    return Path(path).exists()

def sha256_of(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    """Return SHA256 hex digest of a file (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------- Image I/O & preprocessing ----------

def read_image_rgb(image_path: str) -> np.ndarray:
    """Read from disk as grayscale and convert to RGB; raise on failure."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def make_preprocessor(
    img_size: int = 384,
    use_clahe: bool = True,
    mean: Iterable[float] = (0.485, 0.456, 0.406),
    std: Iterable[float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Albumentations preprocessing matching training."""
    ops = [A.Resize(img_size, img_size)]
    if use_clahe:
        ops.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=1.0))
    ops.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])
    return A.Compose(ops)

def preprocess_image_path(image_path: str, preproc: A.Compose) -> torch.Tensor:
    """Load image from path and apply Albumentations; returns (1, C, H, W) tensor."""
    rgb = read_image_rgb(image_path)
    t = preproc(image=rgb)["image"].unsqueeze(0)
    return t

def preprocess_image_array(image: np.ndarray, preproc: A.Compose) -> torch.Tensor:
    """Apply Albumentations to an already-loaded NumPy image (RGB or grayscale)."""
    if image.ndim == 2:  # grayscale → RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    t = preproc(image=image)["image"].unsqueeze(0)
    return t


# ---------- Checkpoint utilities ----------

def sanitize_state_dict(
    state_dict: Dict[str, torch.Tensor],
    drop_prefixes: Tuple[str, ...] = ("gender_head",),
    strip_module: bool = True,
) -> Dict[str, torch.Tensor]:
    """Normalize keys: strip 'module.' and drop unwanted heads."""
    sd = state_dict
    if strip_module and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    if drop_prefixes:
        keep = {}
        for k, v in sd.items():
            if not any(k.startswith(pref) for pref in drop_prefixes):
                keep[k] = v
        sd = keep
    return sd


# ---------- Monte-Carlo (noise) uncertainty ----------

@torch.no_grad()
def mc_noise_predict(
    model,
    image_tensor: torch.Tensor,
    samples: int = 10,
    noise_std: float = 0.01,
    device: Optional[str] = None,
) -> Tuple[List[float], Optional[List[float]]]:
    """
    Run multiple forwards while adding small Gaussian noise to the input.
    Returns (predictions, model_uncertainties_if_present_or_None).
    Expects model(image_tensor) -> dict with 'age' (and optionally 'uncertainty').
    """
    if device:
        image_tensor = image_tensor.to(device)

    preds: List[float] = []
    model_unct: Optional[List[float]] = []

    # Ensure eval mode so BN/Dropout behave consistently unless you intentionally enable MC-dropout.
    was_training = model.training
    model.eval()

    x = image_tensor.clone()
    for i in range(samples):
        out = model(x)
        preds.append(float(out["age"].detach().cpu().item()))
        if "uncertainty" in out:
            model_unct.append(float(out["uncertainty"].detach().cpu().item()))

        if i < samples - 1:
            x = x + torch.randn_like(x) * noise_std

    if was_training:
        model.train()

    if len(model_unct) == 0:
        model_unct = None
    return preds, model_unct


# ---------- Prediction summarization ----------

def summarize_predictions(
    preds: List[float],
    model_unc: Optional[List[float]] = None,
    min_unc_months: float = 3.0,
    conf_cap: float = 0.95,
) -> dict:
    """Combine MC variance + (optional) model-reported aleatoric uncertainty."""
    preds_arr = np.array(preds, dtype=np.float32)
    mean = float(preds_arr.mean())
    std = float(preds_arr.std()) if len(preds_arr) > 1 else 0.0

    if model_unc is not None and len(model_unc) > 0:
        unc_mean = float(np.mean(model_unc))
    else:
        unc_mean = 0.0

    total_unc = max(float(np.sqrt(std**2 + unc_mean**2)), min_unc_months)
    conf = min(1.0 / (1.0 + total_unc / 12.0), conf_cap)

    ci_low = max(0.0, mean - 1.96 * total_unc)
    ci_high = mean + 1.96 * total_unc

    return {
        "age_months": mean,
        "age_years": mean / 12.0,
        "total_uncertainty": total_unc,
        "confidence": conf,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }

def developmental_stage(age_months: float) -> str:
    """Map age in months to a coarse stage label."""
    if age_months < 24:
        return "Infant/Toddler"
    if age_months < 72:
        return "Early Childhood"
    if age_months < 144:
        return "Middle Childhood"
    if age_months < 192:
        return "Adolescence"
    return "Young Adult"
