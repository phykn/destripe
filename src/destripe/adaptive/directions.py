import numpy as np
import torch

from .constants import ALL_DIRECTIONS, CROSS_OFFSETS, EPS, PARALLEL_OFFSETS
from .analysis import extract_high_pass
from .profiles import measure_shrinkage


DIRECTION_COVERAGE_BANDS = 4
MIN_DIRECTION_BAND_SCORE_RATIO = 0.05


def score_directions(high_pass: torch.Tensor) -> dict[int, float]:
    return {
        mode: _score_direction(high_pass, mode=mode)
        for mode in supported_directions(high_pass.shape)
    }


def measure_direction_coverage(tensor: torch.Tensor, mode: int) -> float:
    minimum_band_height = (
        max(
            abs(PARALLEL_OFFSETS[mode][0]),
            abs(CROSS_OFFSETS[mode][0]),
        )
        + 1
    )
    band_count = min(DIRECTION_COVERAGE_BANDS, tensor.shape[0] // minimum_band_height)
    if band_count < 2:
        return 1.0

    band_scores = [
        _score_direction(extract_high_pass(band), mode=mode)
        for band in torch.tensor_split(tensor, band_count, dim=0)
    ]
    median_score = float(np.median(band_scores))
    if median_score <= EPS:
        return 0.0
    score_ratio = min(band_scores) / median_score
    return min(1.0, max(0.0, score_ratio / MIN_DIRECTION_BAND_SCORE_RATIO))


def make_score_weights(scores: dict[int, float]) -> np.ndarray:
    modes, values = _make_score_values(scores)
    return _expand_weights(modes, _sparsemax(values))


def make_selection_weights(scores: dict[int, float]) -> np.ndarray:
    modes, values = _make_score_values(scores)
    if float(values.max() - values.min()) <= EPS:
        weights = np.zeros_like(values)
        weights[int(np.argmax(values))] = 1.0
    else:
        weights = _sparsemax(_standardize_scores(values))
    return _expand_weights(modes, weights)


def select_directions(weights: np.ndarray) -> tuple[int, ...]:
    selected_modes = [
        mode for mode, weight in zip(ALL_DIRECTIONS, weights) if weight > EPS
    ]
    if not selected_modes:
        return (int(np.argmax(weights)),)
    return tuple(sorted(selected_modes, key=lambda mode: (-weights[mode], mode)))


def supported_directions(
    shape: torch.Size | tuple[int, int],
) -> tuple[int, ...]:
    height, width = shape
    return tuple(
        mode
        for mode in ALL_DIRECTIONS
        if _offset_fits(height, width, PARALLEL_OFFSETS[mode])
        and _offset_fits(height, width, CROSS_OFFSETS[mode])
    )


def _make_score_values(
    scores: dict[int, float],
) -> tuple[tuple[int, ...], np.ndarray]:
    modes = tuple(mode for mode in ALL_DIRECTIONS if mode in scores)
    if not modes:
        raise ValueError("scores must contain at least one supported direction.")
    return modes, np.array([scores[mode] for mode in modes], dtype=np.float64)


def _expand_weights(modes: tuple[int, ...], values: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(ALL_DIRECTIONS), dtype=np.float64)
    weights[list(modes)] = values
    return weights


def _offset_fits(
    height: int,
    width: int,
    offset: tuple[int, int],
) -> bool:
    row_step, col_step = offset
    return height > abs(row_step) and width > abs(col_step)


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
    parallel = _offset_diff(t, *PARALLEL_OFFSETS[mode]).abs().reshape(-1)
    cross_offset = CROSS_OFFSETS[mode]
    cross = _offset_diff(t, *cross_offset).abs().reshape(-1)
    parallel_q = float(torch.quantile(parallel, 0.75).item()) + EPS
    cross_q = float(torch.quantile(cross, 0.90).item()) + EPS
    reliability = measure_shrinkage(t, mode)
    return (cross_q / parallel_q) * reliability


def _standardize_scores(values: np.ndarray) -> np.ndarray:
    scale = float(values.std())
    if scale <= EPS:
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
