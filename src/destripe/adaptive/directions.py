import math

import numpy as np
import torch

from . import constants


def score_directions(high_pass: torch.Tensor) -> dict[int, float]:
    return {
        mode: _score_direction(high_pass, mode=mode)
        for mode in constants.ALL_DIRECTIONS
    }


def make_score_weights(scores: dict[int, float]) -> np.ndarray:
    return _sparsemax(_make_score_array(scores))


def make_selection_weights(scores: dict[int, float]) -> np.ndarray:
    values = _make_score_array(scores)
    if float(values.max() - values.min()) <= constants.EPS:
        weights = np.zeros_like(values)
        weights[int(np.argmax(values))] = 1.0
        return weights
    return _sparsemax(_standardize_scores(values))


def select_directions(weights: np.ndarray) -> tuple[int, ...]:
    selected_modes = [
        mode
        for mode, weight in zip(constants.ALL_DIRECTIONS, weights)
        if weight > constants.EPS
    ]
    if not selected_modes:
        return (int(np.argmax(weights)),)
    return tuple(sorted(selected_modes, key=lambda mode: (-weights[mode], mode)))


def _make_score_array(scores: dict[int, float]) -> np.ndarray:
    return np.array(
        [scores[mode] for mode in constants.ALL_DIRECTIONS], dtype=np.float64
    )


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


def _score_direction(t: torch.Tensor, *, mode: int) -> float:
    parallel = _offset_diff(t, *constants.PARALLEL_OFFSETS[mode]).abs().reshape(-1)
    cross_offset = constants.CROSS_OFFSETS[mode]
    cross = _offset_diff(t, *cross_offset).abs().reshape(-1)
    parallel_q = float(torch.quantile(parallel, 0.75).item()) + constants.EPS
    cross_q = float(torch.quantile(cross, 0.90).item()) + constants.EPS
    cross_length = math.hypot(*cross_offset)
    return (cross_q / cross_length) / parallel_q


def _standardize_scores(values: np.ndarray) -> np.ndarray:
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
