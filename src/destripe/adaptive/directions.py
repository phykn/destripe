import math

import numpy as np
import torch

from . import constants


def score_directions(t: torch.Tensor) -> dict[int, float]:
    contrast = _robust_contrast(t)
    return {
        mode: _direction_score(t, mode=mode, contrast=contrast)
        for mode in constants.ALL_DIRECTIONS
    }


def evidence_weights(scores: dict[int, float]) -> np.ndarray:
    values = np.array(
        [scores[mode] for mode in constants.ALL_DIRECTIONS], dtype=np.float64
    )
    return _sparsemax(values)


def support_weights(scores: dict[int, float]) -> np.ndarray:
    values = np.array(
        [scores[mode] for mode in constants.ALL_DIRECTIONS], dtype=np.float64
    )
    if float(values.max() - values.min()) <= constants.EPS:
        weights = np.zeros_like(values)
        weights[int(np.argmax(values))] = 1.0
        return weights
    return _sparsemax(_standardized_scores(values))


def select_directions(scores: dict[int, float]) -> tuple[int, ...]:
    return select_from_weights(support_weights(scores))


def select_from_weights(weights: np.ndarray) -> tuple[int, ...]:
    support = [
        mode
        for mode, weight in zip(constants.ALL_DIRECTIONS, weights)
        if weight > constants.EPS
    ]
    if not support:
        return (int(np.argmax(weights)),)
    return tuple(sorted(support, key=lambda mode: (-weights[mode], mode)))


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
    return float(torch.quantile(t.abs().reshape(-1), 0.90).item()) + constants.EPS


def _direction_score(t: torch.Tensor, *, mode: int, contrast: float) -> float:
    parallel = _offset_diff(t, *constants.PARALLEL_OFFSETS[mode]).abs().reshape(-1)
    cross_offset = constants.CROSS_OFFSETS[mode]
    cross = _offset_diff(t, *cross_offset).abs().reshape(-1)
    parallel_q = float(torch.quantile(parallel, 0.75).item()) + constants.EPS
    cross_q = float(torch.quantile(cross, 0.90).item()) + constants.EPS
    power_q = float(torch.quantile(t.abs().reshape(-1), 0.90).item()) + constants.EPS
    cross_length = math.hypot(*cross_offset)
    return ((cross_q / cross_length) / parallel_q) * (power_q / contrast)


def _standardized_scores(values: np.ndarray) -> np.ndarray:
    scale = float(values.std())
    if scale <= constants.EPS:
        return np.zeros_like(values)
    return (values - float(values.mean())) / scale


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
