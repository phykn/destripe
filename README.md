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
the input shape and dtype. Set `process_size=None` to run the solver at the
original resolution, or use a positive long-edge target for a smaller working
solver. Set `proj=False` only when unclipped floating output is required.

Stripe directions and global regularization are always estimated from the
original resolution. The working solver correction is resized back and refined
against the original image. If downsampling loses a detected direction, the
pipeline falls back to the native solver. Resized results are also checked at
the original resolution and retried natively when they fail to remove at least
half of the detected directional energy. This avoids silently returning an
aliased or under-corrected image. Consequently, `process_size` is a best-effort
speed and memory hint rather than a hard limit. Targets that would save fewer
than 20% of pixels are treated as native to avoid paying resize, validation,
and refinement costs without a meaningful solver reduction.

The automatic path runs a multi-direction PDHG solver with local 2-by-2 tile
weights. Small working images stay on CPU to avoid GPU launch overhead. If
multiple directions coincide with a sparse scene structure, the pipeline
conservatively leaves the input unchanged to avoid subtracting overlapping
structure projections. When repeated directional evidence is too weak relative
to the image high-pass energy, it returns the input unchanged instead of forcing
a stripe direction. The public API intentionally has no adaptive tuning level;
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

## Package Responsibilities

- `ops.py` owns public NumPy validation, normalization, channel handling, and
  dtype restoration.
- `preprocess.py` owns luma conversion and image/solver resizing primitives.
- `automatic.py` owns detection-to-solver orchestration, working-resolution
  fallback, tiling policy, and native residual refinement.
- `adaptive/analysis.py`, `profiles.py`, and `structure.py` separately own
  high-pass analysis, directional profile statistics, and sparse scene
  protection.
- `core/remover.py`, `solver.py`, and `operators.py` own the public manual
  facade, PDHG state updates, and tensor stencils respectively.

## Validation Scope

The bundled `asset/sample.jpeg` is the visual reference for the automatic path.
Because no paired ground truth exists, tests compare detected directions and
residual directional energy across native and working resolutions. The notebook
and visual inspection remain the final qualitative check.

## Reference

- https://github.com/NiklasRottmayer/General-Stripe-Removal
