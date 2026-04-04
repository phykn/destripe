# destripe

PDHG-based universal stripe-noise removal for NumPy images (`[H, W]` or `[H, W, 3]`), backed by PyTorch (`cpu`/`cuda`).

## Features
- Removes vertical and diagonal stripe patterns.
- Supports tiled processing for large images with smooth blending.
- Preserves color by estimating noise from luminance.
- Runs on CPU or CUDA devices.

## Install
```bash
pip install .
```

For development:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start
```python
from destripe import destripe

image = ...  # numpy.ndarray, shape [H, W] or [H, W, 3]

clean = destripe(
    image,
    mu1=0.33,    # smoothing / removal strength
    mu2=0.003,   # stripe penalty (structure protection tradeoff)
    tiles=2,     # n x n grid for tiled processing
    device="cpu" # or "cuda"
)
```

## Parameters
- `mu1` (default `0.33`): stronger smoothing and stripe suppression as it increases.
- `mu2` (default `0.003`): larger values enforce stronger stripe extraction but may affect fine structures.
- `tiles` (default `1`): splits image into `tiles x tiles`; useful for very large inputs.
- `device` (`"cpu"` or `"cuda"`): compute target.

## Suggested `mu` Pairs
- Light, thin stripes: `[0.17, 0.003]`, `[0.23, 0.003]`
- Typical to strong stripes: `[0.33, 0.003]`, `[0.4, 0.007]`
- Severe corruption / short stripes: `[0.5, 0.017]`
- Conservative option: `[0.1, 0.0017]`

These are starting points, not universal optima.

## Test
```bash
pytest -q
```

## Reference
- https://github.com/NiklasRottmayer/General-Stripe-Removal
