import numpy as np
import torch
import torch.nn.functional as F

from . import constants


def analysis_tensor(gray: np.ndarray) -> torch.Tensor:
    evidence = _normalized_gray(gray)
    t = torch.as_tensor(evidence, dtype=torch.float32)
    if t.dim() != 2:
        raise ValueError("gray must have shape (H, W).")
    h, w = t.shape
    max_side = max(h, w)
    if max_side <= 512:
        return t
    scale = 512 / max_side
    size = (max(8, round(h * scale)), max(8, round(w * scale)))
    return F.interpolate(
        t.unsqueeze(0).unsqueeze(0),
        size=size,
        mode="area",
    ).squeeze(0).squeeze(0)


def high_pass(t: torch.Tensor) -> torch.Tensor:
    h, w = t.shape
    if min(h, w) < 4:
        return t - t.mean()
    kernel = int(round(min(h, w) * 0.015))
    kernel = max(7, min(31, kernel | 1))
    pad = kernel // 2
    padded = F.pad(
        t.unsqueeze(0).unsqueeze(0),
        pad=(pad, pad, pad, pad),
        mode="reflect",
    )
    blur = F.avg_pool2d(padded, kernel_size=kernel, stride=1)
    return t - blur.squeeze(0).squeeze(0)


def _normalized_gray(gray: np.ndarray) -> np.ndarray:
    arr = np.asarray(gray, dtype=np.float64)
    if arr.ndim != 2:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    scale = hi - lo
    if scale <= constants.EPS:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / scale
