from dataclasses import dataclass
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F


ALL_DIRECTIONS = (0, 1, 2, 3, 4)
PARALLEL_OFFSETS = {
    0: (1, 0),
    1: (2, 1),
    2: (1, 1),
    3: (2, -1),
    4: (1, -1),
}

_EPS = 1e-9
_NORMAL_MAD_SCALE = 0.6744897501960817


@dataclass(frozen=True)
class AutomaticResult:
    clean: np.ndarray
    direction: int
    alpha: float


def automatic_clean(gray: np.ndarray, *, proj: bool) -> AutomaticResult:
    gray_array = np.asarray(gray, dtype=np.float64)
    tensor = torch.as_tensor(gray_array, dtype=torch.float32)
    high_pass = _extract_high_pass(tensor)
    blurred = cv2.GaussianBlur(
        np.asarray(gray_array, dtype=np.float32),
        (0, 0),
        sigmaX=1.0,
    )
    blurred_high_pass = _extract_high_pass(torch.as_tensor(blurred))

    profiles: dict[int, torch.Tensor] = {}
    protections: dict[int, torch.Tensor] = {}
    reliabilities: dict[int, float] = {}
    for mode in ALL_DIRECTIONS:
        protection = _make_protection(high_pass, mode)
        weights = 1.0 - protection
        profile = _project_robust(high_pass, mode, weights)
        blurred_profile = _project_robust(blurred_high_pass, mode, weights)
        scale_repeatability = _positive_centered_cosine(profile, blurred_profile)
        blocked_repeatability = _blocked_repeatability(
            high_pass,
            mode=mode,
            weights=weights,
        )
        reliability = math.sqrt(blocked_repeatability * scale_repeatability)
        profiles[mode] = torch.nan_to_num(profile)
        protections[mode] = protection
        reliabilities[mode] = float(np.clip(np.nan_to_num(reliability), 0.0, 1.0))

    selected = max(
        ALL_DIRECTIONS,
        key=lambda mode: (reliabilities[mode], -mode),
    )
    selected_profile = profiles[selected]
    selected_protection = protections[selected]
    selected_weights = 1.0 - selected_protection
    proposal_profile = _project_robust(
        _extract_high_pass(selected_profile),
        selected,
        selected_weights,
    )
    curvature = _second_parallel_diff(selected_profile, selected)
    profile_power = float(torch.sum(proposal_profile.square()).item())
    protected_curvature = float(
        torch.sum((selected_protection * curvature).square()).item()
    )
    leakage = protected_curvature / (profile_power + _EPS)
    alpha = _choose_alpha(
        input_profile=selected_profile,
        proposal_profile=proposal_profile,
        reliability=reliabilities[selected],
        leakage=leakage,
    )

    profile_array = selected_profile.numpy().astype(np.float64, copy=False)
    clean = gray_array - alpha * profile_array
    if proj:
        clean = np.clip(clean, 0.0, 1.0)
    return AutomaticResult(clean=clean, direction=selected, alpha=alpha)


def _extract_high_pass(tensor: torch.Tensor) -> torch.Tensor:
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


def _make_protection(high_pass: torch.Tensor, mode: int) -> torch.Tensor:
    activity = _parallel_diff(high_pass, mode).abs()
    median = torch.median(activity)
    mad = torch.median((activity - median).abs()) / _NORMAL_MAD_SCALE
    scale = mad
    if not math.isfinite(float(scale.item())) or float(scale.item()) <= _EPS:
        scale = activity.std(unbiased=False)
    normalized = ((activity - median) / (3.0 * scale + _EPS)).clamp(0.0, 1.0)

    array = normalized.numpy()
    array = cv2.dilate(array, np.ones((3, 3), dtype=np.uint8))
    array = cv2.GaussianBlur(array, (5, 5), sigmaX=0)
    return torch.from_numpy(np.nan_to_num(array)).clamp(0.0, 1.0)


def _parallel_diff(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    row_step, col_step = PARALLEL_OFFSETS[mode]
    source_rows, target_rows = _valid_slices(tensor.shape[0], row_step)
    source_cols, target_cols = _valid_slices(tensor.shape[1], col_step)
    difference = torch.zeros_like(tensor)
    difference[target_rows, target_cols] = (
        tensor[target_rows, target_cols] - tensor[source_rows, source_cols]
    )
    return difference


def _second_parallel_diff(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    row_step, col_step = PARALLEL_OFFSETS[mode]
    source_rows, middle_rows, target_rows = _second_valid_slices(
        tensor.shape[0], row_step
    )
    source_cols, middle_cols, target_cols = _second_valid_slices(
        tensor.shape[1], col_step
    )
    difference = torch.zeros_like(tensor)
    difference[target_rows, target_cols] = (
        tensor[target_rows, target_cols]
        - 2.0 * tensor[middle_rows, middle_cols]
        + tensor[source_rows, source_cols]
    )
    return difference


def _valid_slices(size: int, step: int) -> tuple[slice, slice]:
    if step >= 0:
        span = max(0, size - step)
        start = min(step, size)
        return slice(0, span), slice(start, start + span)
    span = max(0, size + step)
    start = min(-step, size)
    return slice(start, start + span), slice(0, span)


def _second_valid_slices(size: int, step: int) -> tuple[slice, slice, slice]:
    span = max(0, size - 2 * abs(step))
    if step >= 0:
        middle = min(step, size)
        target = min(2 * step, size)
        return (
            slice(0, span),
            slice(middle, middle + span),
            slice(target, target + span),
        )
    source = min(-2 * step, size)
    middle = min(-step, size)
    return (
        slice(source, source + span),
        slice(middle, middle + span),
        slice(0, span),
    )


def _project_robust(
    tensor: torch.Tensor,
    mode: int,
    weights: torch.Tensor,
) -> torch.Tensor:
    safe = weights.to(dtype=tensor.dtype, device=tensor.device).clamp(0.0, 1.0)
    first = _project_weighted(tensor=tensor, mode=mode, weights=safe)
    residual = (tensor - first).abs()
    scale = _project_weighted(tensor=residual, mode=mode, weights=safe)
    cutoff = 1.345 * scale
    huber = torch.where(
        residual <= _EPS,
        torch.ones_like(residual),
        torch.minimum(
            torch.ones_like(residual),
            cutoff / residual.clamp(min=_EPS),
        ),
    )
    return _project_weighted(tensor=tensor, mode=mode, weights=safe * huber)


def _project_weighted(
    *,
    tensor: torch.Tensor,
    mode: int,
    weights: torch.Tensor,
) -> torch.Tensor:
    line_ids = _make_line_ids(tensor.shape, mode, tensor.device).reshape(-1)
    values = tensor.reshape(-1)
    flat_weights = weights.reshape(-1)

    unique, inverse = torch.unique(line_ids, sorted=True, return_inverse=True)
    sums = torch.zeros(
        unique.numel(),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, inverse, flat_weights * values)
    counts.scatter_add_(0, inverse, flat_weights)
    denominator = torch.where(counts == 0, torch.ones_like(counts), counts)
    return (sums / denominator)[inverse].reshape(tensor.shape)


def _make_line_ids(
    shape: torch.Size | tuple[int, int],
    mode: int,
    device: torch.device,
) -> torch.Tensor:
    row_step, col_step = PARALLEL_OFFSETS[mode]
    height, width = shape
    rows = torch.arange(height, device=device)[:, None]
    cols = torch.arange(width, device=device)[None, :]
    return col_step * rows - row_step * cols


def _blocked_repeatability(
    tensor: torch.Tensor,
    mode: int,
    weights: torch.Tensor,
) -> float:
    values = tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    safe = weights.detach().cpu().numpy().astype(np.float64, copy=False)
    height, width = values.shape
    rows, cols = np.indices((height, width), dtype=np.int64)
    row_step, col_step = PARALLEL_OFFSETS[mode]
    line_ids = col_step * rows - row_step * cols
    along = row_step * rows + col_step * cols

    flat_lines = line_ids.reshape(-1)
    unique, inverse = np.unique(flat_lines, return_inverse=True)
    flat_along = along.reshape(-1).astype(np.float64)
    minima = np.full(unique.size, np.inf, dtype=np.float64)
    maxima = np.full(unique.size, -np.inf, dtype=np.float64)
    np.minimum.at(minima, inverse, flat_along)
    np.maximum.at(maxima, inverse, flat_along)
    span = maxima[inverse] - minima[inverse]
    position = np.divide(
        flat_along - minima[inverse],
        span,
        out=np.full_like(flat_along, 0.5),
        where=span > 0,
    )

    flat_values = values.reshape(-1)
    flat_weights = np.clip(safe.reshape(-1), 0.0, 1.0)
    profiles: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for selected in (position <= 0.25, position >= 0.75):
        sums = np.zeros(unique.size, dtype=np.float64)
        counts = np.zeros(unique.size, dtype=np.float64)
        selected_inverse = inverse[selected]
        selected_weights = flat_weights[selected]
        np.add.at(sums, selected_inverse, selected_weights * flat_values[selected])
        np.add.at(counts, selected_inverse, selected_weights)
        profiles.append(np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0))
        masks.append(counts > 0)

    usable = masks[0] & masks[1]
    if int(np.count_nonzero(usable)) < 2:
        return 0.0
    first = profiles[0][usable]
    second = profiles[1][usable]
    first = first - float(first.mean())
    second = second - float(second.mean())
    full = (first + second) / 2.0
    variance = float(np.mean(full * full))
    if variance <= _EPS:
        return 0.0
    covariance = float(np.mean(first * second))
    return float(np.clip(covariance / variance, 0.0, 1.0))


def _positive_centered_cosine(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float:
    centered_first = first - first.mean()
    centered_second = second - second.mean()
    denominator = torch.linalg.vector_norm(centered_first) * torch.linalg.vector_norm(
        centered_second
    )
    if (
        not math.isfinite(float(denominator.item()))
        or float(denominator.item()) <= _EPS
    ):
        return 0.0
    correlation = torch.sum(centered_first * centered_second) / denominator
    return float(torch.nan_to_num(correlation).clamp(0.0, 1.0).item())


def _choose_alpha(
    *,
    input_profile: torch.Tensor,
    proposal_profile: torch.Tensor,
    reliability: float,
    leakage: float,
) -> float:
    normalizer = float(torch.sum(input_profile.square()).item()) + _EPS
    a_value = float(torch.sum(proposal_profile.square()).item()) / normalizer
    a_value += max(0.0, float(leakage))
    b_value = float(reliability) * float(
        torch.sum(input_profile * proposal_profile).item()
    ) / normalizer
    return float(np.clip(b_value / (a_value + _EPS), 0.0, 1.0))
