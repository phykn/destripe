from dataclasses import dataclass
import math

import cv2
import numpy as np
import torch

from .constants import EPS, NORMAL_MAD_SCALE, PARALLEL_OFFSETS
from .preprocess import extract_high_pass
from .stripe import measure_shrinkage, project_robust


@dataclass(frozen=True)
class DirectionEvidence:
    protection: torch.Tensor
    reliability: float
    input_profile: torch.Tensor


def make_direction_evidence(
    gray: np.ndarray,
    directions: tuple[int, ...],
) -> dict[int, DirectionEvidence]:
    image = np.asarray(gray, dtype=np.float32)
    tensor = torch.as_tensor(image, dtype=torch.float32)
    high_pass = extract_high_pass(tensor)
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    blurred_high_pass = extract_high_pass(torch.as_tensor(blurred))

    evidence = {}
    for mode in directions:
        protection = _make_protection(high_pass, mode)
        safe = 1.0 - protection
        input_profile = project_robust(high_pass, mode, safe)
        blurred_profile = project_robust(blurred_high_pass, mode, safe)
        scale_repeatability = _positive_centered_cosine(
            input_profile,
            blurred_profile,
        )
        split_repeatability = measure_shrinkage(
            high_pass,
            mode,
            weights=safe,
        )
        reliability = math.sqrt(split_repeatability * scale_repeatability)
        evidence[mode] = DirectionEvidence(
            protection=protection,
            reliability=float(np.clip(np.nan_to_num(reliability), 0.0, 1.0)),
            input_profile=torch.nan_to_num(input_profile),
        )
    return evidence


def _make_protection(high_pass: torch.Tensor, mode: int) -> torch.Tensor:
    activity = _parallel_activity(high_pass, mode)
    median = torch.median(activity)
    mad = torch.median((activity - median).abs()) / NORMAL_MAD_SCALE
    scale = mad
    if not math.isfinite(float(scale.item())) or float(scale.item()) <= EPS:
        scale = activity.std(unbiased=False)
    normalized = ((activity - median) / (3.0 * scale + EPS)).clamp(0.0, 1.0)

    array = normalized.numpy()
    array = cv2.dilate(array, np.ones((3, 3), dtype=np.uint8))
    array = cv2.GaussianBlur(array, (5, 5), sigmaX=0)
    return torch.from_numpy(np.nan_to_num(array)).clamp(0.0, 1.0)


def _parallel_activity(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    row_step, col_step = PARALLEL_OFFSETS[mode]
    source_rows, target_rows = _valid_slices(tensor.shape[0], row_step)
    source_cols, target_cols = _valid_slices(tensor.shape[1], col_step)
    activity = torch.zeros_like(tensor)
    activity[target_rows, target_cols] = (
        tensor[target_rows, target_cols] - tensor[source_rows, source_cols]
    ).abs()
    return activity


def _valid_slices(size: int, step: int) -> tuple[slice, slice]:
    if step >= 0:
        return slice(0, size - step), slice(step, size)
    return slice(-step, size), slice(0, size + step)


def _positive_centered_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    centered_first = first - first.mean()
    centered_second = second - second.mean()
    denominator = torch.linalg.vector_norm(centered_first) * torch.linalg.vector_norm(
        centered_second
    )
    if not math.isfinite(float(denominator.item())) or float(denominator.item()) <= EPS:
        return 0.0
    correlation = torch.sum(centered_first * centered_second) / denominator
    return float(torch.nan_to_num(correlation).clamp(0.0, 1.0).item())
