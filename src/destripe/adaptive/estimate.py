"""Image-driven direction and parameter estimation."""

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F

_ALL_DIRECTIONS = (0, 1, 2, 3, 4)
_PARALLEL_OFFSETS = {
    0: (1, 0),
    1: (2, 1),
    2: (1, 1),
    3: (2, -1),
    4: (1, -1),
}
_CROSS_OFFSETS = {
    0: (0, 1),
    1: (1, -2),
    2: (1, -1),
    3: (1, 2),
    4: (1, 1),
}

MU1_MIN = 0.10
MU1_MAX = 0.50
MU2_MIN = 0.0017
MU2_MAX = 0.017
_EPS = 1e-9


@dataclass(frozen=True)
class AdaptiveParams:
    directions: tuple[int, ...]
    mu1: float
    mu2: float
    confidence: float


def estimate_adaptive_params(
    gray: np.ndarray,
    *,
    fixed_directions: tuple[int, ...] | None = None,
) -> AdaptiveParams:
    fixed = (
        None
        if fixed_directions is None
        else _validate_fixed_directions(fixed_directions)
    )
    analysis = _analysis_tensor(gray)
    high_pass = _high_pass(analysis)
    contrast = _robust_contrast(high_pass)

    scores = {
        mode: _direction_score(high_pass, mode=mode, contrast=contrast)
        for mode in _ALL_DIRECTIONS
    }
    score_values = _score_values(scores)
    evidence_weights = _stripe_evidence_weights(score_values)
    support_weights = _direction_support_weights(score_values)
    directions = _directions_from_weights(support_weights) if fixed is None else fixed

    evidence_strength = _distribution_concentration(evidence_weights)
    support_strength = _distribution_concentration(support_weights)
    strength = _adaptive_strength(
        evidence_strength=evidence_strength,
        support_strength=support_strength,
    )
    ambiguity = _distribution_entropy(evidence_weights)
    mu1 = _estimate_mu1(strength)
    mu2 = _estimate_mu2(strength=strength, ambiguity=ambiguity)
    confidence = strength * (1.0 - ambiguity)
    return AdaptiveParams(
        directions=tuple(directions),
        mu1=mu1,
        mu2=mu2,
        confidence=confidence,
    )


def _validate_fixed_directions(directions: object) -> tuple[int, ...]:
    if not isinstance(directions, (tuple, list)):
        raise ValueError("directions must be a non-empty sequence of unique modes 0..4.")

    normalized: list[int] = []
    seen: set[int] = set()
    for mode in directions:
        if isinstance(mode, bool) or not isinstance(mode, int):
            raise ValueError("directions must be a non-empty sequence of unique modes 0..4.")
        if mode not in _ALL_DIRECTIONS or mode in seen:
            raise ValueError("directions must be a non-empty sequence of unique modes 0..4.")
        normalized.append(mode)
        seen.add(mode)

    if not normalized:
        raise ValueError("directions must be a non-empty sequence of unique modes 0..4.")
    return tuple(normalized)


def _analysis_tensor(gray: np.ndarray) -> torch.Tensor:
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


def _normalized_gray(gray: np.ndarray) -> np.ndarray:
    arr = np.asarray(gray, dtype=np.float64)
    if arr.ndim != 2:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    scale = hi - lo
    if scale <= _EPS:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / scale


def _high_pass(t: torch.Tensor) -> torch.Tensor:
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


def _offset_diff(t: torch.Tensor, row_step: int, col_step: int) -> torch.Tensor:
    row_start_a = max(0, -row_step)
    row_start_b = max(0, row_step)
    col_start_a = max(0, -col_step)
    col_start_b = max(0, col_step)
    rows = t.shape[0] - abs(row_step)
    cols = t.shape[1] - abs(col_step)
    if rows <= 0 or cols <= 0:
        return torch.zeros(1, dtype=t.dtype, device=t.device)
    a = t[row_start_a : row_start_a + rows, col_start_a : col_start_a + cols]
    b = t[row_start_b : row_start_b + rows, col_start_b : col_start_b + cols]
    return b - a


def _robust_contrast(t: torch.Tensor) -> float:
    return float(torch.quantile(t.abs().reshape(-1), 0.90).item()) + _EPS


def _direction_score(t: torch.Tensor, *, mode: int, contrast: float) -> float:
    parallel = _offset_diff(t, *_PARALLEL_OFFSETS[mode]).abs().reshape(-1)
    cross_offset = _CROSS_OFFSETS[mode]
    cross = _offset_diff(t, *cross_offset).abs().reshape(-1)
    parallel_q = float(torch.quantile(parallel, 0.75).item()) + _EPS
    cross_q = float(torch.quantile(cross, 0.90).item()) + _EPS
    power_q = float(torch.quantile(t.abs().reshape(-1), 0.90).item()) + _EPS
    cross_length = math.hypot(*cross_offset)
    return ((cross_q / cross_length) / parallel_q) * (power_q / contrast)


def _select_directions(scores: dict[int, float]) -> tuple[int, ...]:
    values = _score_values(scores)
    return _directions_from_weights(_direction_support_weights(values))


def _score_values(scores: dict[int, float]) -> np.ndarray:
    return np.array([scores[mode] for mode in _ALL_DIRECTIONS], dtype=np.float64)


def _standardized_scores(values: np.ndarray) -> np.ndarray:
    scale = float(values.std())
    if scale <= _EPS:
        return np.zeros_like(values)
    return (values - float(values.mean())) / scale


def _stripe_evidence_weights(values: np.ndarray) -> np.ndarray:
    return _sparsemax(values)


def _direction_support_weights(values: np.ndarray) -> np.ndarray:
    if float(values.max() - values.min()) <= _EPS:
        weights = np.zeros_like(values)
        weights[int(np.argmax(values))] = 1.0
        return weights
    return _sparsemax(_standardized_scores(values))


def _directions_from_weights(weights: np.ndarray) -> tuple[int, ...]:
    support = [
        mode
        for mode, weight in zip(_ALL_DIRECTIONS, weights)
        if weight > _EPS
    ]
    if not support:
        return (int(np.argmax(weights)),)
    return tuple(
        sorted(support, key=lambda mode: (-weights[mode], mode))
    )


def _sparsemax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(values.mean())
    sorted_values = np.sort(shifted)[::-1]
    cumulative = np.cumsum(sorted_values)
    ranks = np.arange(1, len(sorted_values) + 1, dtype=np.float64)
    support = sorted_values + (1.0 - cumulative) / ranks > 0.0
    if not np.any(support):
        out = np.zeros_like(shifted)
        out[int(np.argmax(shifted))] = 1.0
        return out
    support_size = int(np.nonzero(support)[0][-1]) + 1
    threshold = (cumulative[support_size - 1] - 1.0) / support_size
    return np.maximum(shifted - threshold, 0.0)


def _distribution_concentration(weights: np.ndarray) -> float:
    uniform_power = 1.0 / len(weights)
    power = float(np.sum(weights * weights))
    return min(1.0, max(0.0, (power - uniform_power) / (1.0 - uniform_power)))


def _distribution_entropy(weights: np.ndarray) -> float:
    positive = weights[weights > 0.0]
    if positive.size <= 1:
        return 0.0
    entropy = -float(np.sum(positive * np.log(positive)))
    return min(1.0, max(0.0, entropy / math.log(len(weights))))


def _adaptive_strength(*, evidence_strength: float, support_strength: float) -> float:
    evidence = min(1.0, max(0.0, evidence_strength))
    support = min(1.0, max(0.0, support_strength))
    return math.sqrt(evidence * support)


def _estimate_mu1(strength: float) -> float:
    return _linear_interp(MU1_MIN, MU1_MAX, strength)


def _estimate_mu2(*, strength: float, ambiguity: float) -> float:
    return _log_interp(MU2_MIN, MU2_MAX, strength * ambiguity)


def _linear_interp(lo: float, hi: float, t: float) -> float:
    t = min(1.0, max(0.0, t))
    return float(lo * (1.0 - t) + hi * t)


def _log_interp(lo: float, hi: float, t: float) -> float:
    t = min(1.0, max(0.0, t))
    return float(math.exp(math.log(lo) * (1.0 - t) + math.log(hi) * t))
