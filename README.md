# destripe

PDHG-based stripe-noise removal for NumPy images, backed by PyTorch. The solver decomposes an image into a TV-regularized clean component and directional ℓ²-penalized stripe components.

## Features
- Removes vertical and diagonal stripe patterns with five directional components.
- Optional adaptive mode selects sparse stripe directions and `mu` strengths from the image.
- Manual mode can restrict solver directions with `directions=[...]`; `directions=None` keeps all five modes.
- Accepts grayscale `(H, W)`, single-channel `(H, W, 1)`, and RGB `(H, W, 3)` arrays.
- Preserves input shape and dtype; integer outputs are clipped to their dtype range.
- For RGB inputs, estimates stripes on Rec. 601 luminance and subtracts them from each channel.
- Supports `n x n` tiled processing with cosine-blended overlap for large images.
- Optional `process_scale` runs the solver at lower resolution and subtracts only the upsampled stripe component from the original image.
- Uses CUDA when available if `device=None`; otherwise falls back to CPU.

## Quick Start
```python
from destripe import destripe

image = ...  # numpy.ndarray, shape [H, W] or [H, W, 3]

clean = destripe(
    image,
    adaptive=True,  # automatically chooses directions and mu strengths
    iterations=500,
    tiles=1,       # >1 for n x n tiled processing
    device="cpu",  # "cpu", "cuda", or None to auto-select
)
```

## Adaptive Mode
Adaptive mode is the recommended default for exploratory use. It estimates a
sparse set of stripe directions and chooses `mu1` / `mu2` from the image, then
runs the solver with only the selected directions.

```python
clean = destripe(
    image,
    adaptive=True,
    iterations=500,
)
```

If `adaptive=True`, any explicit `directions`, `mu1`, or `mu2` values are
ignored with a warning.

## Tiled Processing
Use tiled processing for large images or locally varying stripe strength.
Adaptive mode keeps a global direction set and estimates smoothed tile-local
`mu` values.

```python
clean = destripe(
    image,
    adaptive=True,
    tiles=3,
    overlap=64,
)
```

Tile-local `mu` values are processed as a batch, so `tiles > 1` does not force a
separate solver run per tile.

## Coarse Processing
Use `process_scale < 1` when the stripe field is broad enough to estimate at a
lower resolution. The solver runs on the resized luminance/grayscale image,
then only the estimated stripe component is upsampled and subtracted from the
original-resolution input.

```python
clean = destripe(
    image,
    adaptive=True,
    process_scale=0.5,  # 1.0 keeps the original resolution
    iterations=500,
)
```

This is most useful for large images with smooth curtain-like artifacts. Very
thin one-pixel stripes can be weakened by downsampling; keep
`process_scale=1.0` for those cases.

## Manual Mode
Manual mode is useful when the stripe direction and regularization strength are
known.

```python
clean = destripe(
    image,
    directions=[0],  # 0=vertical, 1..4=diagonal modes
    mu1=0.33,
    mu2=0.003,
    iterations=500,
)
```

When `adaptive=False`, `directions=None` keeps all five modes active.

## Parameters
- `image`: numeric NumPy-compatible array with shape `(H, W)`, `(H, W, 1)`, or `(H, W, 3)`.
- `adaptive` (default `False`): estimate directions and `mu` automatically. Explicit `directions`, `mu1`, and `mu2` are ignored with a warning.
- `directions` (default `None`): manual solver modes to use when `adaptive=False`. `None` uses all five modes; pass a non-empty list containing integers `0..4` to restrict the solver.
- `mu1` (default `0.33`): manual TV weight used when `adaptive=False`. Higher smooths more and removes stronger stripes; loses fine detail.
- `mu2` (default `0.003`): manual ℓ² stripe penalty used when `adaptive=False`. Higher extracts stripes more eagerly; can leak real structure.
- `iterations` (default `500`): maximum PDHG iterations.
- `tol` (default `1e-5`): relative-change tolerance for early stopping, checked every 20 iterations.
- `tiles` (default `1`): number of tiles per side. Use values greater than `1` when the image does not fit in memory or stripes are locally non-stationary.
- `overlap` (default `64`): requested blend width in pixels. The solver clamps it to at most one quarter of each tile dimension.
- `process_scale` (default `1.0`): solver resolution scale in `(0, 1]`. Values below `1` estimate the stripe field at lower resolution and subtract the upsampled stripe from the original image.
- `device` (default `None`): `"cpu"`, `"cuda"`, a `torch.device`, or `None` to auto-select CUDA when available.
- `proj` (default `True`): project the normalized clean component onto `[0, 1]`.
- `verbose` (default `False`): print iteration progress.

## Suggested `mu` Pairs
- Conservative / subtle stripes: `[0.1, 0.001]`, `[0.1, 0.0017]`
- Light, thin stripes: `[0.17, 0.003]`, `[0.23, 0.003]`
- Typical to strong stripes: `[0.33, 0.003]`, `[0.4, 0.007]`
- Severe corruption / short stripes: `[0.5, 0.017]`

Adaptive mode uses sparse direction support plus score concentration and entropy
to map each image into the documented literature range. The pairs above are
manual-mode examples, not adaptive tuning anchors.

## Reference
- https://github.com/NiklasRottmayer/General-Stripe-Removal
