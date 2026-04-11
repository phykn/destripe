import numpy as np
import torch

from .core import UniversalStripeRemover

# Rec. 601 luma coefficients (standard for NTSC/JPEG grayscale conversion)
_LUMA_R = 0.2989
_LUMA_G = 0.5870
_LUMA_B = 0.1140


def destripe(
    image: np.ndarray,
    mu1: float = 0.33,
    mu2: float = 0.003,
    iterations: int = 500,
    tol: float = 1e-5,
    tiles: int = 1,
    overlap: int = 64,
    proj: bool = True,
    device: torch.device | str | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Remove stripe noise from a NumPy image.

    Supports grayscale ``(H, W)`` and color ``(H, W, C)`` images where
    ``C in {1, 3}``. For RGB inputs, stripe estimates are computed on the
    luminance channel and then subtracted from each color channel.

    Args:
        image: Input image array.
        mu1: TV regularization weight.
        mu2: L2 stripe penalty weight.
        iterations: Maximum PDHG iterations. Must be positive.
        tol: Relative convergence tolerance. Must be non-negative.
        tiles: Number of tiles per image side. Must be positive.
        overlap: Overlap width between neighboring tiles. Must be non-negative.
        proj: Whether to project the clean component onto ``[0, 1]``.
        device: Computation device for the underlying torch solver.
        verbose: Whether to print iteration progress.

    Returns:
        Destriped image with the same shape and dtype as ``image``.

    Raises:
        ValueError: If image rank/channels are unsupported, the input contains
            non-finite values, or solver/tile parameters are invalid.
    """
    input_array = np.asarray(image)
    if not np.issubdtype(input_array.dtype, np.number):
        raise ValueError("image must contain numeric values.")
    if not np.isfinite(input_array).all():
        raise ValueError("image must not contain NaN or Inf values.")

    orig_dtype = input_array.dtype
    normalized = input_array.astype(np.float64)

    min_value, max_value = normalized.min(), normalized.max()
    scale = max_value - min_value
    if scale < 1e-12:
        return input_array.copy()
    normalized = (normalized - min_value) / scale

    remover = UniversalStripeRemover(mu1=mu1, mu2=mu2, device=device)

    if normalized.ndim == 2:
        clean = _run(remover, normalized, iterations, tol, tiles, overlap, proj, verbose)
    elif normalized.ndim == 3 and normalized.shape[2] in {1, 3}:
        if normalized.shape[2] == 3:
            gray = (
                _LUMA_R * normalized[..., 0]
                + _LUMA_G * normalized[..., 1]
                + _LUMA_B * normalized[..., 2]
            )
        else:
            gray = normalized[..., 0]

        clean_gray = _run(remover, gray, iterations, tol, tiles, overlap, proj, verbose)
        stripe = gray - clean_gray
        clean = np.clip(normalized - stripe[..., np.newaxis], 0.0, 1.0)
    else:
        raise ValueError(
            f"image must have shape (H, W) or (H, W, C) with C in {{1, 3}}, "
            f"got {input_array.shape}."
        )

    result = clean * scale + min_value

    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        result = np.clip(result, info.min, info.max)

    return result.astype(orig_dtype)


def _run(
    remover: UniversalStripeRemover,
    gray: np.ndarray,
    iterations: int,
    tol: float,
    tiles: int,
    overlap: int,
    proj: bool,
    verbose: bool,
) -> np.ndarray:
    out = remover.process_tiled(
        image=gray,
        tiles=tiles,
        iterations=iterations,
        tol=tol,
        overlap=overlap,
        proj=proj,
        verbose=verbose,
    )
    return out.numpy().astype(np.float64)
