from collections.abc import Sequence
import warnings

import numpy as np
import torch

from .adaptive import estimate_adaptive_params, estimate_tile_mus
from .adaptive.constants import ADAPTIVE_LEVELS
from .adaptive.refine import refine_clean
from .core import UniversalStripeRemover
from .preprocess import (
    prepare_solver_gray,
    resize_to_shape,
    rgb_to_luma,
    validate_process_size,
)


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
    adaptive: int | None = None,
    directions: Sequence[int] | None = None,
) -> np.ndarray:
    """Remove stripe noise from a NumPy image.

    Supports grayscale, single-channel, and RGB arrays. RGB stripes are
    estimated from luma and subtracted from every channel.

    Args:
        image: Input image.
        mu1: Manual TV weight; ignored when adaptive is a level.
        mu2: Manual stripe sparsity weight; ignored when adaptive is a level.
        iterations: Maximum PDHG iterations.
        tol: Relative convergence tolerance.
        tiles: Tiles per image side.
        overlap: Tile overlap in pixels.
        process_size: Long-edge solver size; None keeps original resolution.
        proj: Project output to [0, 1].
        device: Torch device.
        verbose: Print solver progress.
        adaptive: None for manual settings, or level 0..3 for adaptive mode.
        directions: Manual stripe modes; ignored when adaptive is a level.

    Returns:
        Destriped image with the same shape and dtype.
    """
    input_array = np.asarray(image)
    if not np.issubdtype(input_array.dtype, np.number):
        raise ValueError("image must contain numeric values.")
    if not np.isfinite(input_array).all():
        raise ValueError("image must not contain NaN or Inf values.")

    process_size_value = validate_process_size(process_size)
    adaptive_level = _validate_adaptive(adaptive)
    manual_args_used = (
        mu1 is not None
        or mu2 is not None
        or directions is not None
    )

    if adaptive_level is not None and manual_args_used:
        warnings.warn(
            "adaptive level ignores manual directions, mu1, and mu2 values.",
            UserWarning,
            stacklevel=2,
        )

    effective_mu1 = 1 / 3 if adaptive_level is not None or mu1 is None else mu1
    effective_mu2 = 1 / 300 if adaptive_level is not None or mu2 is None else mu2
    effective_directions = None if adaptive_level is not None else directions

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
            adaptive_level=adaptive_level,
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
            gray = rgb_to_luma(normalized)
        else:
            gray = normalized[..., 0]

        clean_gray = _destripe_grayscale(
            gray=gray,
            adaptive_level=adaptive_level,
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
    adaptive_level: int | None,
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
    processed_gray = prepare_solver_gray(gray=gray, process_size=process_size)
    if adaptive_level is not None:
        params = estimate_adaptive_params(
            processed_gray,
            level=adaptive_level,
            fixed_directions=None,
        )
        resolved_mu1 = params.mu1
        resolved_mu2 = params.mu2
        resolved_directions = params.directions
    else:
        resolved_mu1 = mu1
        resolved_mu2 = mu2
        resolved_directions = directions

    remover = UniversalStripeRemover(
        mu1=resolved_mu1,
        mu2=resolved_mu2,
        device=device,
        directions=resolved_directions,
    )
    tile_mus = None
    if adaptive_level is not None and tiles > 1:
        tile_mus = estimate_tile_mus(
            gray=processed_gray,
            tiles=tiles,
            level=adaptive_level,
            directions=resolved_directions,
        )
    solver_input = np.asarray(processed_gray, dtype=np.float32)
    solver_clean = remover.process_tiled(
        image=solver_input,
        tiles=tiles,
        iterations=iterations,
        tol=tol,
        overlap=overlap,
        proj=proj,
        verbose=verbose,
        tile_mus=tile_mus,
    ).numpy()
    if adaptive_level is not None:
        solver_clean = refine_clean(
            gray=processed_gray,
            clean=solver_clean,
            directions=resolved_directions,
            proj=proj,
        )

    if processed_gray.shape == gray.shape:
        return solver_clean

    stripe = processed_gray - solver_clean
    full_stripe = resize_to_shape(stripe, shape=gray.shape)
    clean = gray - full_stripe
    if proj:
        clean = np.clip(clean, 0.0, 1.0)
    return clean


def _validate_adaptive(adaptive: object) -> int | None:
    if adaptive is None:
        return None
    if isinstance(adaptive, bool) or not isinstance(adaptive, int):
        raise ValueError("adaptive must be None or an integer level 0..3.")
    if adaptive not in ADAPTIVE_LEVELS:
        raise ValueError("adaptive must be None or an integer level 0..3.")
    return adaptive
