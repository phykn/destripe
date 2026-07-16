# destripe

An automatic adaptive PDHG pipeline removes stripe noise from NumPy images.
For advanced control, the manual PDHG core is available through PyTorch.

## Automatic Usage

In most cases, call `destripe` directly:

```python
from destripe import destripe

clean = destripe(image, process_size=256)
```

`image` can have shape `(H, W)`, `(H, W, 1)`, or `(H, W, 3)`. The output keeps
the input shape and dtype. Set `process_size=None` to analyze the original
resolution, or use a positive long-edge size to estimate broad curtain fields
at lower resolution and resize the correction back. Set `proj=False` only when
unclipped floating output is required.

Native analysis scales with the image pixel count. For large images, use
`process_size=256` or `process_size=512` to bound analysis time and memory unless
native-resolution detection of very narrow stripes is required.

The automatic path estimates supported stripe directions and `mu2` from the
image, then runs the multi-direction PDHG solver with local 2-by-2 tile weights.
If multiple directions coincide with a sparse scene structure, it conservatively
leaves the input unchanged to avoid subtracting overlapping structure projections.
When repeated directional evidence is too weak relative to the image high-pass
energy, it returns the input unchanged
instead of forcing a stripe direction. It finishes detected stripes with a small
residual refinement. The public API intentionally has no adaptive tuning level;
the quality-oriented solver configuration is kept internal and deterministic.

## Manual PDHG Core

Use `UniversalStripeRemover` when the direction and regularization weights must
be controlled explicitly:

```python
from destripe import UniversalStripeRemover

remover = UniversalStripeRemover(
    mu1=1 / 3,
    mu2=1 / 300,
    directions=[0],
    device="cpu",
)
clean = remover.process(image, iterations=500)
```

Mode `0` targets vertical stripes; modes `1` through `4` target the supported
diagonal directions. Passing `directions=None` uses all five modes.

## Validation Scope

The bundled `asset/sample.jpeg` is the visual reference for the automatic path.
Because no paired ground truth exists for this real striped image, the executed
notebook and visual inspection remain the quality check.

## Reference

- https://github.com/NiklasRottmayer/General-Stripe-Removal
