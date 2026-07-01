import cv2
import numpy as np

from . import constants
from .estimate import estimate_adaptive_params


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
    tile_h = padded.shape[0] // tiles
    tile_w = padded.shape[1] // tiles
    tile_grid = padded.reshape(tiles, tile_h, tiles, tile_w).swapaxes(1, 2)

    mus = np.empty((tiles * tiles, 2), dtype=np.float64)
    for index, tile in enumerate(tile_grid.reshape(-1, tile_h, tile_w)):
        params = estimate_adaptive_params(tile, fixed_directions=directions)
        mus[index] = (params.mu1, params.mu2)

    mus = mus.reshape(tiles, tiles, 2)
    smoothed = smooth_tile_mus(mus)
    return [
        (float(mu1), float(mu2))
        for mu1, mu2 in smoothed.reshape(-1, 2)
    ]


def smooth_tile_mus(mus: np.ndarray) -> np.ndarray:
    if mus.ndim != 3 or mus.shape[-1] != 2:
        raise ValueError("mus must have shape (rows, cols, 2).")

    clipped = np.asarray(mus, dtype=np.float64).copy()
    clipped[..., 0] = np.clip(mus[..., 0], constants.MU1_MIN, constants.MU1_MAX)
    clipped[..., 1] = np.clip(mus[..., 1], constants.MU2_MIN, constants.MU2_MAX)

    log_mus = np.log(clipped)
    smoothed = np.exp(
        cv2.blur(log_mus, ksize=(3, 3), borderType=cv2.BORDER_REPLICATE)
    )
    smoothed[..., 0] = np.clip(
        smoothed[..., 0], constants.MU1_MIN, constants.MU1_MAX
    )
    smoothed[..., 1] = np.clip(
        smoothed[..., 1], constants.MU2_MIN, constants.MU2_MAX
    )
    return smoothed
