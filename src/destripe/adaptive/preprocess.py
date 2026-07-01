import numpy as np
import torch
import torch.nn.functional as F

from . import constants


def make_analysis_tensor(gray: np.ndarray) -> torch.Tensor:
    normalized = _normalize_gray(gray)
    tensor = torch.as_tensor(normalized, dtype=torch.float32)
    height, width = tensor.shape
    max_side = max(height, width)
    if max_side <= 512:
        return tensor
    scale = 512 / max_side
    size = (max(8, round(height * scale)), max(8, round(width * scale)))
    return F.interpolate(
        tensor.unsqueeze(0).unsqueeze(0),
        size=size,
        mode="area",
    ).squeeze(0).squeeze(0)


def extract_high_pass(tensor: torch.Tensor) -> torch.Tensor:
    height, width = tensor.shape
    if min(height, width) < 4:
        return tensor - tensor.mean()
    kernel = int(round(min(height, width) * 0.015))
    kernel = max(7, min(31, kernel | 1))
    pad = kernel // 2
    padded = F.pad(
        tensor.unsqueeze(0).unsqueeze(0),
        pad=(pad, pad, pad, pad),
        mode="reflect",
    )
    blur = F.avg_pool2d(padded, kernel_size=kernel, stride=1)
    return tensor - blur.squeeze(0).squeeze(0)


def _normalize_gray(gray: np.ndarray) -> np.ndarray:
    arr = np.asarray(gray, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("gray must have shape (H, W).")
    low = float(np.min(arr))
    high = float(np.max(arr))
    scale = high - low
    if scale <= constants.EPS:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - low) / scale
