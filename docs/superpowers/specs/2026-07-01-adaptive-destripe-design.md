# Adaptive destripe design

Date: 2026-07-01

## Goal

Add an opt-in adaptive mode that avoids the current over-blurring failure mode caused by applying all five stripe directions with one global manual strength. The result remains the same public artifact as today: a cleaned NumPy image with the input shape and dtype preserved.

The adaptive mode should choose the necessary stripe directions and regularization strengths from the image itself. It should not become a brute-force grid search over many direction and parameter combinations.

## Public API

`destripe()` gains:

```python
adaptive: bool = False
directions: Sequence[int] | None = None
```

Manual mode:

- `adaptive=False` keeps existing behavior by default.
- `directions=None` means all five solver modes are active.
- `directions=[...]` restricts the solver to the listed modes.
- `mu1` and `mu2` keep their existing manual meaning.

Adaptive mode:

- `adaptive=True` ignores manual `directions`, `mu1`, and `mu2`.
- If any ignored manual value is supplied, emit `warnings.warn(..., UserWarning, stacklevel=2)`.
- The return value is still only the cleaned image.
- `verbose` should not print adaptive internals by default; the API goal is clean output, not parameter reporting.

To detect whether `mu1` and `mu2` were supplied, the high-level wrapper accepts `None` as "use the manual default" and internally maps it to `0.33` and `0.003` when `adaptive=False`. Existing callers that omit the parameters get identical behavior. Explicit numeric values can then be detected and warned about under `adaptive=True`.

`UniversalStripeRemover` gains manual `directions` support so the core solver can run on a subset of modes. Adaptive estimation stays inside the high-level NumPy wrapper for this iteration.

## Direction handling

Supported solver modes remain `0..4`, matching the existing directional difference operators:

- `0`: vertical stripe component, smooth along image rows.
- `1..4`: existing diagonal modes.

Validation:

- `directions=None` expands to `(0, 1, 2, 3, 4)`.
- A list must be non-empty.
- Every direction must be an integer in `0..4`.
- Duplicate directions are rejected rather than silently collapsed.

The solver must replace hard-coded `_NUM_DIRS` loop usage with the active direction tuple. Constraint projection denominators become `1 + len(active_directions)`, and clamp residual redistribution divides by `len(active_directions)`.

## Adaptive Principle

Adaptive mode uses one rule:

```text
stronger stripe evidence increases mu1;
greater structure/stripe ambiguity increases mu2.
```

`mu1` is the removal strength. `mu2` is the precaution against removing real structures. The ratio matters, so the estimator should produce a coherent pair rather than independently snapping to a cartesian product of values.

Parameter bounds:

- `mu1` is clamped to `[0.10, 0.50]`.
- `mu2` is clamped to `[0.0017, 0.017]`.
- `(0.33, 0.003)` is the neutral anchor.

These bounds come from the published paper's examples and parameter interpretation, plus the upstream author's current implementation comments after the 2025-07-29 scaling update.

## Evidence Estimation

Adaptive estimation runs on normalized grayscale data. For RGB input, this is the same Rec. 601 luminance path already used before subtracting the estimated stripe from all channels.

Before evidence computation, downsample large images to max side `512` using area interpolation. This keeps estimation cost small and avoids making adaptive runtime scale with full image size.

For each solver mode:

1. Build a high-pass analysis image by subtracting a reflect-padded box blur from the grayscale image. Use an odd kernel near 1.5% of the shorter analysis side, clamped to `[7, 31]`. This suppresses broad illumination trends and emphasizes stripe-scale artifacts.
2. Measure parallel smoothness with the same directional operator family as the solver, using the candidate mode's directional difference.
3. Measure cross-stripe contrast with finite differences projected approximately perpendicular to the candidate stripe direction.
4. Convert these into a robust stripe evidence score using quantiles of absolute values: `q75` for typical directional variation and `q90` for artifact-scale high-pass power. This avoids mean-dominated behavior from a few bright structures.

The selected directions are sparse:

- Usually choose one direction.
- Choose a second direction only if its evidence is at least `85%` of the top direction and above the same absolute evidence floor.
- Never let adaptive drift back into selecting all five directions for ordinary images.

The estimator also computes a structure ambiguity score. Ambiguity is high when strong image structures have the same directional footprint as the detected stripe evidence. High ambiguity raises `mu2`, making the stripe component more conservative.

## Mu Estimation

Map evidence to continuous values:

- Normalize global stripe evidence against robust image contrast.
- Increase `mu1` smoothly above the anchor when evidence is high.
- Decrease `mu1` toward the lower bound when evidence is weak.
- Increase `mu2` when ambiguity is high or selected directions are not cleanly separated.
- Keep `mu2` near the lower/neutral range when stripe evidence is strong and directionally clean.

This is implemented as a small deterministic mapping in log-space for smooth interpolation across the documented parameter range. The constants are local and documented in code, not exposed as public options.

## Confidence Gate

Adaptive should not always run multiple solvers. The normal path is:

```text
estimate directions and mu -> run the full solver once
```

A short pilot run is allowed only when confidence is low. Low confidence means one of:

- The top directions are close but weak.
- Estimated `mu` lies near a clamp boundary.
- Stripe evidence is moderate but ambiguity is high.

Pilot behavior:

- Run on the downsampled analysis image.
- Use short iterations only.
- Compare the estimated `mu` against one adjacent continuous alternative, not a full six-pair grid. The adjacent alternative is stronger when stripe residual evidence dominates, and more conservative when ambiguity/detail loss dominates.
- First require sufficient stripe residual reduction; among acceptable candidates, prefer lower detail loss.

The pilot is a calibration check, not an optimization search.

## Tiled Processing

Direction selection remains global. The user stated direction changes across the image are rare.

For `adaptive=True` and `tiles > 1`, `mu` may vary by tile because stripe strength can vary spatially. Use the same estimator per tile with global directions fixed. To avoid visible tile inconsistency:

- Estimate local `mu1` and `mu2` for each tile.
- Smooth the tile-grid parameter field with one 3x3 neighbor average pass in log-space before solving tiles.
- Process each tile with its local parameters, then keep the existing cosine-blended overlap reconstruction.

This is the same principle as global adaptive estimation applied locally. It avoids separate hand-written rules for each tile case.

## Errors And Warnings

Validation errors:

- Invalid `directions` values raise `ValueError`.
- Empty `directions` raises `ValueError`.
- Existing validation for image shape, finite values, `iterations`, `tol`, `tiles`, and `overlap` remains.

Warnings:

- `adaptive=True` with explicit `directions`, `mu1`, or `mu2` emits one `UserWarning` explaining that adaptive mode ignores manual direction and mu parameters.

## Tests

Focused tests should cover:

- Existing default behavior still preserves shape and dtype.
- `directions=None` matches the current all-direction path.
- `directions=[0]` runs with a subset and preserves shape/dtype.
- Invalid directions fail for out-of-range, non-integer, empty, and duplicate values.
- `adaptive=True` returns shape/dtype-compatible output for grayscale, single-channel, RGB, and tiled inputs.
- `adaptive=True` with explicit manual values emits a warning.
- Adaptive direction estimation selects the expected dominant mode for simple synthetic stripe images.
- Adaptive mode reduces synthetic stripe evidence while preserving more detail than the all-direction manual baseline on a small controlled fixture.
- Constant images still return an unchanged copy.

## Documentation

README should describe:

- `adaptive=False` as the compatibility default.
- `adaptive=True` as the recommended automatic path when the user does not want to tune direction and mu.
- Manual `directions` for advanced users.
- The documented `mu` range as guidance, not an optimality guarantee.

## Sources Checked

- Rottmayer, Redenbach, and Fahrbach, Optics Express 33(3), 5800-5809, 2025. DOI: https://doi.org/10.1364/OE.542868
- RPTU open-access record and PDF: https://kluedo.ub.rptu.de/frontdoor/index/index/year/2025/docId/8687
- Upstream implementation at commit `f1972aa47fc3ecf6580984f72ce7ce630b1ea683`: https://github.com/NiklasRottmayer/General-Stripe-Removal
