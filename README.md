# destripe

PDHG-based stripe-noise removal for NumPy images, backed by PyTorch.

The solver decomposes an image into a TV-regularized clean component and sparse
directional stripe components. In most cases, use `adaptive=True`.

## Usage

```python
from destripe import destripe

clean = destripe(
    image,
    adaptive=True,
    iterations=500,
)
```

`image` can be `(H, W)`, `(H, W, 1)`, or `(H, W, 3)`. Output shape and dtype are
preserved.

## Adaptive Mode

Adaptive mode automatically:

- selects the stripe direction set from five candidate modes,
- estimates `mu1` and `mu2` from directional stripe evidence,
- applies a residual line-projection shrinkage step for weak curtain artifacts.

The residual step projects the high-pass clean image onto the selected
line-constant stripe subspace and scales it by split-half cross-covariance:

```text
alpha = clip(Cov(p1, p2) / Var((p1 + p2) / 2), 0, 1)
```

This keeps line structure that is reproducible across the split and suppresses
non-reproducible texture.

Manual `directions`, `mu1`, and `mu2` are ignored when `adaptive=True`.

## Large Images

Use `process_size` to estimate broad curtain fields at lower resolution:

```python
clean = destripe(image, adaptive=True, process_size=512)
```

Use `tiles` when stripe strength varies locally:

```python
clean = destripe(image, adaptive=True, tiles=3, overlap=64)
```

## Manual Mode

```python
clean = destripe(
    image,
    adaptive=False,
    directions=[0],
    mu1=0.33,
    mu2=0.003,
)
```

`directions=None` uses all five modes. Mode `0` targets vertical stripes;
`1..4` target diagonal modes.

## Key Parameters

- `adaptive`: estimate directions and `mu` automatically.
- `process_size`: optional solver long-edge size; `None` keeps original resolution.
- `tiles`: number of tiles per side.
- `mu1`: manual TV weight for `adaptive=False`.
- `mu2`: manual stripe sparsity weight for `adaptive=False`.
- `device`: `"cpu"`, `"cuda"`, `torch.device`, or `None`.

## Reference

- https://github.com/NiklasRottmayer/General-Stripe-Removal
