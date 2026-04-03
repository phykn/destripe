import numpy as np
import torch

from .core import UniversalStripeRemover

_LUMA_R = 0.2989
_LUMA_G = 0.5870
_LUMA_B = 0.1140


def destripe(
    image: np.ndarray,
    mu1: float = 0.1,
    mu2: float = 0.001,
    iterations: int = 500,
    tol: float = 1e-5,
    tiles: int = 1,
    overlap: int = 64,
    proj: bool = True,
    device: torch.device | str | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Removes stripe noise from a numpy image.

    Supports grayscale (H, W) and color (H, W, C) images. For color images,
    stripes are estimated from the luminance channel and subtracted from
    each color channel to preserve hue.

    Args:
        image: Input image. Integer or float dtype accepted.
        mu1: TV regularization weight.
        mu2: L2 stripe penalty weight.
        iterations: Maximum PDHG iterations.
        tol: Relative convergence tolerance.
        tiles: Number of tiles per side for tiled processing.
        overlap: Overlap pixels between tiles.
        proj: Whether to project u onto [0, 1].
        device: Computation device.
        verbose: Print iteration progress.

    Returns:
        Destriped image with the same shape and dtype as input.
    """
    orig_dtype = image.dtype
    img = image.astype(np.float64)

    vmin, vmax = img.min(), img.max()
    scale = vmax - vmin
    if scale < 1e-12:
        return image.copy()
    img = (img - vmin) / scale

    remover = UniversalStripeRemover(mu1=mu1, mu2=mu2, device=device)

    if img.ndim == 2:
        clean = _run(remover, img, iterations, tol, tiles, overlap, proj, verbose)
    elif img.ndim == 3 and img.shape[2] in {1, 3}:
        channels = img.shape[2]
        if channels == 3:
            gray = _LUMA_R * img[..., 0] + _LUMA_G * img[..., 1] + _LUMA_B * img[..., 2]
        else:
            gray = img[..., 0]

        clean_gray = _run(remover, gray, iterations, tol, tiles, overlap, proj, verbose)
        stripe = gray - clean_gray
        clean = np.clip(img - stripe[..., np.newaxis], 0.0, 1.0)
    else:
        raise ValueError(
            f"image must have shape (H, W) or (H, W, C) with C in {{1, 3}}, "
            f"got {image.shape}."
        )

    result = clean * scale + vmin

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

    if tiles > 1:
        out = remover.process_tiled(
            image=gray,
            tiles=tiles,
            iterations=iterations,
            tol=tol,
            overlap=overlap,
            proj=proj,
            verbose=verbose,
        )
    else:
        out = remover.process(
            image=gray,
            iterations=iterations,
            tol=tol,
            proj=proj,
            verbose=verbose,
        )
    return out.numpy().astype(np.float64)
