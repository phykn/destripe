from collections.abc import Sequence
import numbers
import warnings

import cv2
import numpy as np
import torch

from .adaptive import estimate_adaptive_params, smooth_tile_mus
from .core import UniversalStripeRemover

# Rec. 601 luma coefficients (standard for NTSC/JPEG grayscale conversion)
_LUMA_R = 0.2989
_LUMA_G = 0.5870
_LUMA_B = 0.1140


def _estimate_tile_mus(
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


def destripe(
    image: np.ndarray,
    mu1: float | None = None,
    mu2: float | None = None,
    iterations: int = 500,
    tol: float = 1e-5,
    tiles: int = 1,
    overlap: int = 64,
    process_size: int | None = None,
    proj: bool = True,
    device: torch.device | str | None = None,
    verbose: bool = False,
    adaptive: bool = False,
    directions: Sequence[int] | None = None,
) -> np.ndarray:
    """Remove stripe noise from a NumPy image.

    Supports grayscale ``(H, W)`` and color ``(H, W, C)`` images where
    ``C in {1, 3}``. For RGB inputs, stripe estimates are computed on the
    luminance channel and then subtracted from each color channel.

    Args:
        image: Input image array.
        mu1: TV regularization weight. Uses the manual default when ``None``.
            Ignored when ``adaptive`` is true.
        mu2: L2 stripe penalty weight. Uses the manual default when ``None``.
            Ignored when ``adaptive`` is true.
        iterations: Maximum PDHG iterations. Must be positive.
        tol: Relative convergence tolerance. Must be non-negative.
        tiles: Number of tiles per image side. Must be positive.
        overlap: Overlap width between neighboring tiles. Must be non-negative.
        process_size: Optional long-edge size for the solver image. ``None``
            keeps the original resolution. Integers smaller than the input
            long edge resize the grayscale/luminance solver image while
            preserving aspect ratio, then upsample only the estimated stripe
            component and subtract it from the original-resolution image.
        proj: Whether to project the clean component onto ``[0, 1]``.
        device: Computation device for the underlying torch solver.
        verbose: Whether to print iteration progress.
        adaptive: Whether to use adaptive parameter estimation.
        directions: Stripe direction modes to use. ``None`` uses all modes.
            Ignored when ``adaptive`` is true.

    Returns:
        Destriped image with the same shape and dtype as
        ``np.asarray(image)``.

    Raises:
        ValueError: If image rank/channels are unsupported, the input contains
            non-finite values, or solver/tile parameters are invalid.
    """
    input_array = np.asarray(image)
    if not np.issubdtype(input_array.dtype, np.number):
        raise ValueError("image must contain numeric values.")
    if not np.isfinite(input_array).all():
        raise ValueError("image must not contain NaN or Inf values.")

    process_size_value = _validate_process_size(process_size)
    manual_mu1 = mu1 is not None
    manual_mu2 = mu2 is not None
    manual_directions = directions is not None

    if adaptive and (manual_mu1 or manual_mu2 or manual_directions):
        warnings.warn(
            "adaptive=True ignores manual directions, mu1, and mu2 values.",
            UserWarning,
            stacklevel=2,
        )

    effective_mu1 = 0.33 if adaptive or mu1 is None else mu1
    effective_mu2 = 0.003 if adaptive or mu2 is None else mu2
    effective_directions = None if adaptive else directions

    orig_dtype = input_array.dtype
    normalized = input_array.astype(np.float64)

    min_value, max_value = normalized.min(), normalized.max()
    intensity_scale = max_value - min_value
    if intensity_scale < 1e-12:
        return input_array.copy()
    normalized = (normalized - min_value) / intensity_scale

    if normalized.ndim == 2:
        clean = _destripe_grayscale(
            gray=normalized,
            adaptive=adaptive,
            mu1=effective_mu1,
            mu2=effective_mu2,
            directions=effective_directions,
            device=device,
            iterations=iterations,
            tol=tol,
            tiles=tiles,
            overlap=overlap,
            proj=proj,
            verbose=verbose,
            process_size=process_size_value,
        )
    elif normalized.ndim == 3 and normalized.shape[2] in {1, 3}:
        if normalized.shape[2] == 3:
            gray = (
                _LUMA_R * normalized[..., 0]
                + _LUMA_G * normalized[..., 1]
                + _LUMA_B * normalized[..., 2]
            )
        else:
            gray = normalized[..., 0]

        clean_gray = _destripe_grayscale(
            gray=gray,
            adaptive=adaptive,
            mu1=effective_mu1,
            mu2=effective_mu2,
            directions=effective_directions,
            device=device,
            iterations=iterations,
            tol=tol,
            tiles=tiles,
            overlap=overlap,
            proj=proj,
            verbose=verbose,
            process_size=process_size_value,
        )
        stripe = gray - clean_gray
        clean = normalized - stripe[..., np.newaxis]
        if proj:
            clean = np.clip(clean, 0.0, 1.0)
    else:
        raise ValueError(
            f"image must have shape (H, W) or (H, W, C) with C in {{1, 3}}, "
            f"got {input_array.shape}."
        )

    result = clean * intensity_scale + min_value

    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        result = np.clip(result, info.min, info.max)

    return result.astype(orig_dtype)


def _destripe_grayscale(
    *,
    gray: np.ndarray,
    adaptive: bool,
    mu1: float,
    mu2: float,
    directions: Sequence[int] | None,
    device: torch.device | str | None,
    iterations: int,
    tol: float,
    tiles: int,
    overlap: int,
    proj: bool,
    verbose: bool,
    process_size: int | None,
) -> np.ndarray:
    solver_gray = _solver_gray(gray=gray, process_size=process_size)
    remover = _make_remover(
        gray=solver_gray,
        adaptive=adaptive,
        mu1=mu1,
        mu2=mu2,
        directions=directions,
        device=device,
    )
    tile_mus = None
    if adaptive and tiles > 1:
        tile_mus = _estimate_tile_mus(
            gray=solver_gray,
            tiles=tiles,
            directions=remover.directions,
        )
    solver_clean = _run_grayscale(
        remover=remover,
        gray=solver_gray,
        iterations=iterations,
        tol=tol,
        tiles=tiles,
        overlap=overlap,
        proj=proj,
        verbose=verbose,
        tile_mus=tile_mus,
    )

    if solver_gray.shape == gray.shape:
        return solver_clean

    stripe = solver_gray - solver_clean
    full_stripe = _resize_2d(stripe, size=gray.shape, mode="lanczos")
    clean = gray - full_stripe
    if proj:
        clean = np.clip(clean, 0.0, 1.0)
    return clean


def _make_remover(
    *,
    gray: np.ndarray,
    adaptive: bool,
    mu1: float,
    mu2: float,
    directions: Sequence[int] | None,
    device: torch.device | str | None,
) -> UniversalStripeRemover:
    if adaptive:
        params = estimate_adaptive_params(gray, fixed_directions=None)
        return UniversalStripeRemover(
            mu1=params.mu1,
            mu2=params.mu2,
            device=device,
            directions=params.directions,
        )
    return UniversalStripeRemover(
        mu1=mu1,
        mu2=mu2,
        device=device,
        directions=directions,
    )


def _run_grayscale(
    remover: UniversalStripeRemover,
    gray: np.ndarray,
    iterations: int,
    tol: float,
    tiles: int,
    overlap: int,
    proj: bool,
    verbose: bool,
    tile_mus: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    out = remover.process_tiled(
        image=gray,
        tiles=tiles,
        iterations=iterations,
        tol=tol,
        overlap=overlap,
        proj=proj,
        verbose=verbose,
        tile_mus=tile_mus,
    )
    return out.numpy()


def _validate_process_size(process_size: int | None) -> int | None:
    if process_size is None:
        return None
    if isinstance(process_size, bool) or not isinstance(process_size, numbers.Integral):
        raise ValueError("process_size must be None or an integer greater than 1.")
    value = int(process_size)
    if value <= 1:
        raise ValueError("process_size must be None or an integer greater than 1.")
    return value


def _solver_gray(*, gray: np.ndarray, process_size: int | None) -> np.ndarray:
    size = _process_shape(gray.shape, process_size)
    if size == gray.shape:
        return gray
    return np.clip(_resize_2d(gray, size=size, mode="lanczos"), 0.0, 1.0)


def _process_shape(
    shape: tuple[int, int], process_size: int | None
) -> tuple[int, int]:
    if process_size is None:
        return shape
    h, w = shape
    long_edge = max(h, w)
    if process_size >= long_edge:
        return shape
    scale = process_size / long_edge
    if h >= w:
        return process_size, _scaled_dim(w, scale)
    return _scaled_dim(h, scale), process_size


def _scaled_dim(dim: int, scale: float) -> int:
    if dim <= 1:
        return dim
    return max(2, int(np.floor(dim * scale + 0.5)))


def _resize_2d(
    image: np.ndarray,
    *,
    size: tuple[int, int],
    mode: str,
) -> np.ndarray:
    if image.shape == size:
        return np.asarray(image, dtype=np.float64).copy()

    if mode != "lanczos":
        raise ValueError(f"unsupported resize mode: {mode}")

    array = np.asarray(image, dtype=np.float64)
    resized = cv2.resize(
        array,
        dsize=(size[1], size[0]),
        interpolation=cv2.INTER_LANCZOS4,
    )
    return np.asarray(resized, dtype=np.float64).reshape(size)
