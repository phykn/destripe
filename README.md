# destripe

Universal stripe-noise removal for NumPy images. Decomposes an image into a TV-regularized clean component and directional ℓ²-penalized stripe components via a PDHG (primal-dual hybrid gradient) solver, backed by PyTorch.

## Features
- Removes vertical and diagonal stripe patterns (5 directions).
- Preserves dtype and value range: `uint8`, `float32`, `float64` round-trip unchanged.
- Supports grayscale, single-channel, and RGB; color is preserved by estimating stripes on the luminance channel.
- Tiled processing with cosine blending for large images that do not fit in memory.
- Auto-selects CUDA when available; falls back to CPU.

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
    tiles=1,        # >1 for n x n tiled processing
    device="cpu",   # "cpu", "cuda", or None to auto-select
)
```

Integer inputs are clipped to their dtype range on output.

## Parameters
- `mu1` (default `0.33`): TV weight. Higher smooths more and removes stronger stripes; loses fine detail.
- `mu2` (default `0.003`): ℓ² stripe penalty. Higher extracts stripes more eagerly; can leak real structure.
- `iterations` (default `500`): max PDHG iterations. Early-stops on relative change below `tol` (default `1e-5`), checked every 20 iterations.
- `tiles` (default `1`): `n × n` tiling with cosine-blended overlap. Use when the image does not fit in memory or stripes are locally non-stationary. `overlap` (default `64`) sets blending width.
- `device` (default `None`): `"cpu"`, `"cuda"`, a `torch.device`, or `None` to auto-select CUDA when available.
- `proj` (default `True`): project the clean component onto `[0, 1]`.

## Suggested `mu` Pairs
- Conservative / subtle stripes: `[0.1, 0.001]`, `[0.1, 0.0017]`
- Light, thin stripes: `[0.17, 0.003]`, `[0.23, 0.003]`
- Typical to strong stripes: `[0.33, 0.003]`, `[0.4, 0.007]`
- Severe corruption / short stripes: `[0.5, 0.017]`

Starting points, not universal optima.

## Reference
- https://github.com/NiklasRottmayer/General-Stripe-Removal
