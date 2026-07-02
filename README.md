# destripe

PDHG-based stripe-noise removal for NumPy images, backed by PyTorch.

The solver decomposes an image into a TV-regularized clean component and sparse
directional stripe components. In most cases, use an adaptive level.

## Usage

```python
from destripe import destripe

clean = destripe(
    image,
    adaptive=2,
    iterations=500,
)
```

`image` can be `(H, W)`, `(H, W, 1)`, or `(H, W, 3)`. Output shape and dtype are
preserved.

## Adaptive Mode

Adaptive mode automatically:

- selects the stripe direction set from five candidate modes,
- sets `mu1` from the requested level,
- estimates `mu2` from stripe-candidate soft-threshold risk,
- applies a residual line-projection shrinkage step for weak curtain artifacts.

Levels control how strongly the clean image is regularized:

```text
adaptive=0 -> mu1 = 1 / 6
adaptive=1 -> mu1 = 1 / 5
adaptive=2 -> mu1 = 1 / 4
adaptive=3 -> mu1 = 1 / 3
```

`mu2` is still selected from the image, using line-reliability-adjusted
soft-threshold risk over `1 / 300` through `1 / 60`.

The residual step projects the high-pass clean image onto the selected
line-constant stripe subspace and scales it by split-half cross-covariance:

```text
alpha = clip(Cov(p1, p2) / Var((p1 + p2) / 2), 0, 1)
```

This keeps line structure that is reproducible across the split and suppresses
non-reproducible texture.

Manual `directions`, `mu1`, and `mu2` are ignored when `adaptive` is a level.

## Large Images

Use `process_size` to estimate broad curtain fields at lower resolution:

```python
clean = destripe(image, adaptive=2, process_size=512)
```

Use `tiles` when stripe strength varies locally:

```python
clean = destripe(image, adaptive=2, tiles=3, overlap=64)
```

## Manual Mode

```python
clean = destripe(
    image,
    adaptive=None,
    directions=[0],
    mu1=1 / 3,
    mu2=1 / 300,
)
```

`directions=None` uses all five modes. Mode `0` targets vertical stripes;
`1..4` target diagonal modes.

## Key Parameters

- `adaptive`: `None` for manual mode, or `0..3` for adaptive mode.
- `process_size`: optional solver long-edge size; `None` keeps original resolution.
- `tiles`: number of tiles per side.
- `mu1`: manual TV weight for `adaptive=None`.
- `mu2`: manual stripe sparsity weight for `adaptive=None`.
- `device`: `"cpu"`, `"cuda"`, `torch.device`, or `None`.

## Reference

- https://github.com/NiklasRottmayer/General-Stripe-Removal
