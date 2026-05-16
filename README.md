# destripe

PDHG-based stripe-noise removal for NumPy images, backed by PyTorch. The solver decomposes an image into a TV-regularized clean component and directional ℓ²-penalized stripe components.

## Features
- Removes vertical and diagonal stripe patterns with five directional components.
- Accepts grayscale `(H, W)`, single-channel `(H, W, 1)`, and RGB `(H, W, 3)` arrays.
- Preserves input shape and dtype; integer outputs are clipped to their dtype range.
- For RGB inputs, estimates stripes on Rec. 601 luminance and subtracts them from each channel.
- Supports `n x n` tiled processing with cosine-blended overlap for large images.
- Uses CUDA when available if `device=None`; otherwise falls back to CPU.

## Install
```bash
pip install destripe
```

## Quick Start
```python
from destripe import destripe

image = ...  # numpy.ndarray, shape [H, W] or [H, W, 3]

clean = destripe(
    image,
    mu1=0.33,
    mu2=0.003,
    iterations=500,
    tiles=1,       # >1 for n x n tiled processing
    device="cpu",  # "cpu", "cuda", or None to auto-select
)
```

## Parameters
- `image`: numeric NumPy-compatible array with shape `(H, W)`, `(H, W, 1)`, or `(H, W, 3)`.
- `mu1` (default `0.33`): TV weight. Higher smooths more and removes stronger stripes; loses fine detail.
- `mu2` (default `0.003`): ℓ² stripe penalty. Higher extracts stripes more eagerly; can leak real structure.
- `iterations` (default `500`): maximum PDHG iterations.
- `tol` (default `1e-5`): relative-change tolerance for early stopping, checked every 20 iterations.
- `tiles` (default `1`): number of tiles per side. Use values greater than `1` when the image does not fit in memory or stripes are locally non-stationary.
- `overlap` (default `64`): requested blend width in pixels. The solver clamps it to at most one quarter of each tile dimension.
- `device` (default `None`): `"cpu"`, `"cuda"`, a `torch.device`, or `None` to auto-select CUDA when available.
- `proj` (default `True`): project the normalized clean component onto `[0, 1]`.
- `verbose` (default `False`): print iteration progress.

## Suggested `mu` Pairs
- Conservative / subtle stripes: `[0.1, 0.001]`, `[0.1, 0.0017]`
- Light, thin stripes: `[0.17, 0.003]`, `[0.23, 0.003]`
- Typical to strong stripes: `[0.33, 0.003]`, `[0.4, 0.007]`
- Severe corruption / short stripes: `[0.5, 0.017]`

Starting points, not universal optima.

## Reference
- https://github.com/NiklasRottmayer/General-Stripe-Removal
