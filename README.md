# destripe

Automatic robust directional profiles remove stripe noise from NumPy images.
For advanced control, an optional manual PDHG core is available through PyTorch.

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

The automatic H3 path evaluates five stripe directions, estimates robust line
profiles from direction-consistent pixels, checks distant-block and multi-scale
repeatability, and subtracts the strongest supported profile with an analytic
coefficient. It does not require ground truth, a strength level, or a per-image
threshold.

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

The automatic path is validated primarily for vertical battery-SEM curtaining.
Medium and strong oblique stripes perform well, but weak oblique direction
coverage remains incomplete. That limitation needs a separate future design
rather than hidden image-specific heuristics.

## Reference

- https://github.com/NiklasRottmayer/General-Stripe-Removal
