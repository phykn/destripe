# destripe

Universal stripe-noise removal for NumPy images (`[H, W]` or `[H, W, 3]`). Decomposes an image into a TV-regularized clean component and directional ℓ²-penalized stripe components via a PDHG (primal-dual hybrid gradient) solver, implemented in PyTorch with CPU/CUDA support.

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

For development:
```bash
git clone https://github.com/phykn/destripe.git
cd destripe
pip install -e .
```

## Quick Start
```python
import numpy as np
from destripe import destripe

image = ...  # numpy.ndarray, shape [H, W] or [H, W, 3], any numeric dtype

clean = destripe(
    image,
    mu1=0.33,       # TV smoothing / stripe-removal strength
    mu2=0.003,      # stripe penalty (structure-protection tradeoff)
    iterations=500, # max PDHG iterations
    tiles=1,        # set >1 to process in an n x n grid for large inputs
    device="cpu",   # "cpu", "cuda", or None to auto-select
)
```

The output has the same shape and dtype as the input. Integer inputs are clipped to their dtype range.

## Parameters
- `mu1` (default `0.33`): TV regularization weight. Larger values smooth more aggressively and remove stronger stripes, at the cost of fine detail.
- `mu2` (default `0.003`): ℓ² penalty on stripe components. Larger values extract stripes more eagerly but can leak real structure into the stripe estimate.
- `iterations` (default `500`): maximum PDHG iterations. The solver checks a relative-change convergence criterion (`tol`, default `1e-5`) every 20 iterations and stops early when reached.
- `tiles` (default `1`): splits the image into an `n × n` grid, runs the solver per tile as a single batch, and blends with cosine windows. Use when the full image does not fit in memory or when stripes are locally non-stationary. `overlap` (default `64`) controls blending width.
- `device` (default `None`): `"cpu"`, `"cuda"`, a `torch.device`, or `None` to auto-select CUDA when available.
- `proj` (default `True`): projects the clean component onto `[0, 1]` during normalized solving.

See the docstring of `destripe()` for the full signature.

## Suggested `mu` Pairs
- Conservative / subtle stripes: `[0.1, 0.001]`, `[0.1, 0.0017]`
- Light, thin stripes: `[0.17, 0.003]`, `[0.23, 0.003]`
- Typical to strong stripes: `[0.33, 0.003]`, `[0.4, 0.007]`
- Severe corruption / short stripes: `[0.5, 0.017]`

These are starting points, not universal optima. Stronger `mu1` at fixed `mu2` smooths more; stronger `mu2` extracts stripes more aggressively.

## Test
```bash
pip install -e .
pytest -q
```

## Reference
- https://github.com/NiklasRottmayer/General-Stripe-Removal
