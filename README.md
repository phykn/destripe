# destripe

An automatic H3-guided PDHG hybrid removes stripe noise from NumPy images.
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

The parameter-free automatic path uses H3 only to detect one supported stripe
direction and to apply safety gates. Four along-line regions must agree, the
signal must not be concentrated in one broad structural band, and ambiguous
mirrored directions are rejected. If those checks fail, the image is returned
unchanged.

When detection is supported, a compact image-derived `mu1`/`mu2` search runs the
existing PDHG solver in the selected direction. The candidate that best explains
the H3 target with the least protected-structure leakage is scaled
conservatively and applied. This keeps correction spatially local instead of
subtracting one profile along an entire line. Weak supported curtains can be
removed, while ambiguous weak signals safely remain unchanged.

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
On the bundled SEM sources it preserves clean images exactly, improves medium
and strong continuous synthetic curtains, and returns interrupted curtains
unchanged so clean gaps do not acquire full-line noise. Highly textured 1%
synthetic curtains may safely no-op when the evidence is ambiguous. Weak oblique
direction coverage remains incomplete; it is not handled with image-specific
branches.

## Reference

- https://github.com/NiklasRottmayer/General-Stripe-Removal
