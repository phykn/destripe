# Simple Automatic Destriping Design

## Decision

Replace the adaptive-level controller with the frozen H3 algorithm and remove
compatibility code for unused levels. `destripe()` becomes one automatic path.
Advanced manual PDHG use remains available through `UniversalStripeRemover`.

This supersedes the adaptive-controller architecture in
`2026-07-10-adaptive-quality-preserving-destriping-design.md`.

## Evidence

The component-plus-residual controller failed identically on seeds `1234` and
`20260710`: canonical weak PSNR fell by about `1.6 dB`, clean sample 02 reached
only `37.2 dB`, and robustness worst cases lost about `11 dB`.

Systematic diagnostics found two causes:

- adjacent alternating splits treated smooth SEM structure as repeatable stripe;
- solver components and residual projections contained real content and received
  coefficients much larger than GT-diagnostic oracle coefficients.

The frozen H3 diagnostic uses one direct robust profile, distant blocked
repeatability, one selected direction, and the existing analytic coefficient. It
has no fitted constant, confidence blend, pattern-specific branch, or top-K grid.

One-shot blind seed `8675309` was run after freezing H3:

- clean minimum: `50.58 dB`, SSIM `0.999357`;
- vertical weak curtain: `+1.84 dB`, projection left `63.75%`, all 4 cases pass;
- canonical medium/strong: `+3.37/+3.95 dB`;
- canonical weak across all five orientations: `+1.11 dB`, but projection left
  `79.52%` and coverage `70%`;
- robustness pooled: `+0.84 dB`, worst `-1.50 dB`.

The blind result was stable relative to development seed `1234`. Weak oblique
coverage remains a documented limitation rather than a reason to add more
scoring stages.

## Public API

The NumPy wrapper becomes:

```text
destripe(image, *, process_size=None, proj=True) -> np.ndarray
```

There is no `adaptive`, level, `mu1`, `mu2`, direction, iteration, tile,
overlap, tolerance, device, or verbosity argument on this wrapper.

Manual PDHG remains explicit:

```python
remover = UniversalStripeRemover(
    mu1=1 / 3,
    mu2=1 / 300,
    directions=[0],
    device="cpu",
)
clean = remover.process(image, iterations=500)
```

No deprecated aliases or ignored arguments remain. Passing an old wrapper
argument raises normal Python `TypeError`.

## Automatic Algorithm

At processed-image resolution:

1. Normalize the grayscale input and extract the existing local high-pass image.
2. For each of the five fixed directions, build a direction-aware protection map.
3. Estimate one robust line profile from unprotected pixels.
4. Measure distant blocked repeatability along the stripe direction and combine
   it with the existing sigma-1 scale repeatability by geometric mean.
5. Select the single direction with maximum combined reliability, breaking ties
   by the lowest mode number.
6. Calculate the existing analytic `choose_alpha()` coefficient for that direct
   profile and subtract `alpha * profile` from the input.
7. Resize only the correction field to the original resolution and apply it to
   grayscale or all RGB channels.

The accepted correction is never spatially masked at SEM edges. Protection is
used only for profile estimation and leakage measurement.

## Code Structure

Create `src/destripe/automatic.py` as the single automatic-algorithm module. It
owns direction offsets, high-pass extraction, protection, robust line projection,
blocked repeatability, scale repeatability, top-1 selection, and coefficient
application.

Keep `src/destripe/core/` unchanged as the manual PDHG implementation. Simplify
`src/destripe/ops.py` to normalization, shape/dtype/RGB handling, process-size
handling, and one call into `automatic.py`.

Delete the old `src/destripe/adaptive/` package and its estimator, strength,
tile-mu, level, component-safety, and refinement code. Delete tests that only
assert those removed contracts.

## Benchmark and Acceptance

Remove the benchmark `level` column and `--levels` argument. Each case is emitted
once. Update uniqueness/completeness identities accordingly.

Primary acceptance reflects the battery-SEM scope:

- clean sample 02 through 05: at least `40 dB` PSNR and `0.99` SSIM;
- vertical mode-0 weak curtains: mean PSNR gain at least `0.10 dB`, mean SSIM
  gain at least `0.001`, mean projection left at most `70%`, all four cases gain
  at least `0.05 dB` and `0.0001` SSIM, and no case loses more than `1 dB`;
- canonical medium and strong curtains across all five modes retain the existing
  non-regression gates;
- all outputs are finite and projected when requested.

Weak oblique and robustness results remain mandatory report rows but are not
converted into tuned pass thresholds. Their per-mode metrics are printed and
stored so limitations cannot be hidden.

Seeds `1234` and `20260710` are development/regression seeds. Seed `8675309` is
the frozen blind validation already run once. No coefficient, threshold, or
branch may change in response to its result. Source implementation must reproduce
the frozen diagnostic within numerical tolerance.

## Performance and Safety

The automatic path does not run PDHG. Median runtime must not exceed the original
`1.322108 s` CPU baseline and is expected to improve substantially.

Constant/tiny images, non-finite input, dtype/shape preservation, RGB shared
correction, process-size resize-back, determinism, and affine intensity behavior
remain tested. `proj=False` preserves unclipped floating output behavior.

## Documented Limitation

The simple controller is validated primarily for vertical battery-SEM curtaining.
Medium and strong oblique stripes perform well, but weak oblique direction
coverage is incomplete. Improving that requires a separate future design, not
additional hidden heuristics in this implementation.
