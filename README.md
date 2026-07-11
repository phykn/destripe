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

The automatic path estimates supported stripe directions and `mu2` from the
image, then runs the multi-direction PDHG solver with local 2-by-2 tile weights.
It finishes with a small residual refinement. The public API intentionally has
no adaptive tuning level; the quality-oriented solver configuration is kept
internal and deterministic.

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

The automatic path is validated primarily for battery-SEM curtaining. The
bundled real striped image is the visual baseline; clean SEM sources are used
with synthetic curtains for repeatable checks. Because no paired ground truth
exists for the real striped image, visual inspection remains part of validation.

## Reference

- https://github.com/NiklasRottmayer/General-Stripe-Removal
