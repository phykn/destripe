"""Tile-local adaptive parameter estimation."""

import numpy as np

from .estimate import (
    _MU1_MAX,
    _MU1_MIN,
    _MU2_MAX,
    _MU2_MIN,
    estimate_adaptive_params,
)


def estimate_tile_mus(
    gray: np.ndarray,
    *,
    tiles: int,
    directions: tuple[int, ...],
) -> list[tuple[float, float]]:
    if tiles <= 1:
        return []
    h, w = gray.shape
    pad_h = (tiles - h % tiles) % tiles
    pad_w = (tiles - w % tiles) % tiles
    pad_mode = (
        "edge"
        if ((pad_h > 0 and h <= 1) or (pad_w > 0 and w <= 1))
        else "reflect"
    )
    padded = np.pad(gray, ((0, pad_h), (0, pad_w)), mode=pad_mode)
    core_h = padded.shape[0] // tiles
    core_w = padded.shape[1] // tiles
    mus = np.empty((tiles, tiles, 2), dtype=np.float64)
    for row in range(tiles):
        for col in range(tiles):
            tile = padded[
                row * core_h : (row + 1) * core_h,
                col * core_w : (col + 1) * core_w,
            ]
            params = estimate_adaptive_params(tile, fixed_directions=directions)
            mus[row, col, 0] = params.mu1
            mus[row, col, 1] = params.mu2
    smoothed = smooth_tile_mus(mus)
    return [
        (float(smoothed[row, col, 0]), float(smoothed[row, col, 1]))
        for row in range(tiles)
        for col in range(tiles)
    ]


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
