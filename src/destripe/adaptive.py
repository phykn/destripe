"""Adaptive direction and parameter estimation for destriping."""

from __future__ import annotations

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

_MU1_MIN = 0.10
_MU1_ANCHOR = 0.33
_MU1_MAX = 0.50
_MU2_MIN = 0.0017
_MU2_ANCHOR = 0.003
_MU2_MAX = 0.017
_STRENGTH_ANCHOR_SCORE = 1.9
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
    directions = _select_directions(scores) if fixed is None else fixed
    selected_scores = [scores[mode] for mode in directions]
    top_score = max(selected_scores)
    selected_ranked = sorted(selected_scores, reverse=True)
    second_score = selected_ranked[1] if len(selected_ranked) > 1 else 0.0
    ambiguity = _ambiguity_score(selected_scores=selected_scores)

    strength = _stripe_strength(top_score)
    mu1 = _estimate_mu1(strength)
    mu2 = _estimate_mu2(strength=strength, ambiguity=ambiguity)
    confidence = _confidence(
        top_score=top_score,
        second_score=second_score,
        ambiguity=ambiguity,
    )
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


def smooth_tile_mus(mus: np.ndarray) -> np.ndarray:
    if mus.ndim != 3 or mus.shape[-1] != 2:
        raise ValueError("mus must have shape (rows, cols, 2).")

    clipped = np.empty_like(mus, dtype=np.float64)
    clipped[..., 0] = np.clip(mus[..., 0], _MU1_MIN, _MU1_MAX)
    clipped[..., 1] = np.clip(mus[..., 1], _MU2_MIN, _MU2_MAX)

    log_mus = np.log(clipped)
    padded = np.pad(log_mus, ((1, 1), (1, 1), (0, 0)), mode="edge")
    out = np.zeros_like(log_mus)
    for row_offset in range(3):
        for col_offset in range(3):
            out += padded[
                row_offset : row_offset + log_mus.shape[0],
                col_offset : col_offset + log_mus.shape[1],
                :,
            ]
    out /= 9.0
    smoothed = np.exp(out)
    smoothed[..., 0] = np.clip(smoothed[..., 0], _MU1_MIN, _MU1_MAX)
    smoothed[..., 1] = np.clip(smoothed[..., 1], _MU2_MIN, _MU2_MAX)
    return smoothed


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
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_mode, top_score = ranked[0]
    directions = [top_mode]
    if ranked[1][1] >= 0.85 * top_score and ranked[1][1] >= 1.15:
        directions.append(ranked[1][0])
    return tuple(directions)


def _stripe_strength(score: float) -> float:
    anchor_span = _STRENGTH_ANCHOR_SCORE - 1.0
    return min(1.0, max(0.0, (score - 1.0) / (2.0 * anchor_span)))


def _estimate_mu1(strength: float) -> float:
    if strength <= 0.5:
        return _log_interp(_MU1_MIN, _MU1_ANCHOR, strength / 0.5)
    return _log_interp(_MU1_ANCHOR, _MU1_MAX, (strength - 0.5) / 0.5)


def _estimate_mu2(*, strength: float, ambiguity: float) -> float:
    base = _log_interp(_MU2_MIN, _MU2_ANCHOR, min(1.0, strength / 0.5))
    if ambiguity <= 0.5:
        return base
    return _log_interp(base, _MU2_MAX, (ambiguity - 0.5) / 0.5)


def _ambiguity_score(*, selected_scores: list[float]) -> float:
    ranked = sorted(selected_scores, reverse=True)
    top = ranked[0] + _EPS
    second = ranked[1] if len(ranked) > 1 else 0.0
    direction_confusion = second / top
    multi_direction_penalty = 0.25 if len(selected_scores) > 1 else 0.0
    return min(1.0, max(0.0, direction_confusion + multi_direction_penalty))


def _confidence(*, top_score: float, second_score: float, ambiguity: float) -> float:
    dominance = 1.0 - min(1.0, second_score / (top_score + _EPS))
    strength = min(1.0, max(0.0, (top_score - 1.0) / 3.0))
    return min(1.0, max(0.0, 0.5 * dominance + 0.5 * strength - 0.25 * ambiguity))


def _log_interp(lo: float, hi: float, t: float) -> float:
    t = min(1.0, max(0.0, t))
    return float(math.exp(math.log(lo) * (1.0 - t) + math.log(hi) * t))
