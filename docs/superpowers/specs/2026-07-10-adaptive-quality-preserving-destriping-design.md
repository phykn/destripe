# Adaptive Quality-Preserving Destriping Design

## Context

The current adaptive path selects directions and regularization parameters, runs
the PDHG solver once, and then always applies residual line-projection refinement.
The solver works well on medium and strong synthetic curtains, but the current
controller can remove real SEM structure when a stripe is weak or absent.

The existing benchmark at `adaptive=2` showed the trade-off clearly:

- weak synthetic stripes (`strength=0.01`) had a mean PSNR change of `-6.431 dB`;
- medium stripes (`0.03`) had a mean PSNR change of `+0.919 dB`;
- strong stripes (`0.06`) had a mean PSNR change of `+3.783 dB`;
- direction selection included the injected direction in `90.5%` of cases, so
  direction detection alone is not the main failure;
- successful and failed cases had overlapping confidence ranges, so a single
  confidence threshold cannot safely decide whether to run destriping.

The redesign therefore keeps the solver and replaces the unconditional adaptive
refinement with an automatic, quality-preserving controller.

## Goals

- Remove faint, line-coherent stripes without requiring ground truth or per-image
  parameter tuning.
- Preserve SEM edges and texture by reducing corrections that overlap
  direction-inconsistent image structure.
- Run the PDHG solver exactly once per existing solver invocation.
- Select the applied correction continuously from the closed interval `[0, 1]`,
  where `0` is the unchanged input and `1` is the full proposed correction.
- Keep the public `destripe(..., adaptive=0..3)` API and output shape/dtype
  behavior unchanged.
- Keep median adaptive runtime overhead at or below `15%` on the four clean SEM
  samples with `process_size=256`, `iterations=500`, and CPU execution.

## Non-goals

- Training a neural network or requiring a larger real-image dataset.
- Searching multiple `mu1`, `mu2`, or direction combinations by rerunning PDHG.
- Guaranteeing separation of a real structure that is mathematically identical
  to a full-length stripe. A single image contains no evidence that can resolve
  that ambiguity.
- Changing manual mode behavior.

## Architecture

Adaptive processing retains the existing estimator and solver, then passes the
input and solver output through a new safety controller:

```text
normalized gray
  -> adaptive direction/mu estimation
  -> one PDHG solve
  -> direction-aware structure protection
  -> robust residual stripe proposal
  -> analytic correction-strength selection
  -> protected adaptive output
```

The new controller operates at `processed_gray` resolution. Existing resize-back
logic then transfers the selected stripe field to the original resolution. RGB
handling remains luma-based and therefore automatically receives the same safety
decision for every channel.

## Components

### Direction-aware structure protection

Create `src/destripe/adaptive/safety.py` to own protection-map construction and
automatic correction selection. For each selected direction, it computes image
variation parallel to that direction using the existing directional offsets.
A true stripe is approximately constant along its direction, while curved
particles, crossings, and texture normally vary along it.

For parallel activity `a`, calculate `m = median(a)` and robust scale
`s = MAD(a) / 0.67448975`, falling back to the standard deviation and then
`EPS` only when necessary. The raw protection is

```text
clip((a - m) / (3*s + EPS), 0, 1).
```

It is max-filtered over a `3x3` neighborhood and Gaussian blurred with a `5x5`
kernel. The result is a protection map:

- `0` means the location is safe for stripe estimation and correction;
- `1` means the location is likely real structure and should resist correction.

The calculation is direction-specific. A cross-direction stripe boundary must
not be protected merely because it has a large gradient; otherwise the detector
would protect the artifact it is intended to remove.

### Robust line projection

Extend `src/destripe/adaptive/stripe.py` with weighted, robust projection while
preserving the existing unweighted `project()` behavior.

The robust projector uses two scatter-based passes:

1. Compute a protection-weighted mean for every directional line.
2. Estimate each line's weighted absolute residual scale, apply standard Huber
   weights with cutoff `1.345`, and recompute the line value.

This excludes localized SEM edges without Python loops over pixels or lines. The
existing split-half reliability calculation gains an optional weight map so that
protected structure does not inflate stripe coherence.

### Repeatability evidence

For each selected direction, calculate:

- split-half repeatability using alternating samples along the stripe direction;
- scale repeatability between the normal high-pass image and the high-pass image
  after a Gaussian blur with `sigma=1`;
- a combined reliability equal to the geometric mean of the two values.

Reliability is based on correlation rather than stripe amplitude. Consequently,
a faint but repeated curtain can remain reliable, while a strong localized edge
does not automatically qualify as a stripe.

### Residual stripe proposal

The solver correction is `gray - solver_clean`. The controller also projects the
remaining high-pass content of `solver_clean` through the robust projector.
Only the unprotected portion of that residual, scaled by its direction
reliability, is added to the proposed correction. When multiple directions are
present, divide the sum by `max(1, sum(reliabilities))` so overlapping proposals
cannot amplify the image.

This keeps the current ability to remove weak residual curtains but prevents the
unconditional full-line subtraction performed by the old refinement step.

### Analytic correction-strength selection

Let `c` be the combined solver and residual correction. Candidate outputs are
the continuous family

```text
y(alpha) = gray - alpha * c,  alpha in [0, 1].
```

For every reliable direction, robustly project the high-pass input (`p`) and the
high-pass correction (`q`). The stripe residual term is

```text
sum_d reliability_d * ||p_d - alpha*q_d||^2 / (||p_d||^2 + eps).
```

Combine direction-specific protection maps with a pixelwise maximum, call the
result `h`, and define the detail ratio

```text
D = sum(h*c^2) / (sum((1-h)*c^2) + eps).
```

For each direction, let `N_d = ||p_d||^2 + eps` and reliability be `r_d`. The
quadratic coefficients are

```text
A = sum_d r_d * ||q_d||^2 / N_d + D
B = sum_d r_d * <p_d, q_d> / N_d
alpha = clip(B / (A + eps), 0, 1).
```

This is the exact minimum of the normalized directional residual plus the
quadratic detail penalty. It makes structure-contaminated corrections expensive,
requires no candidate images, and requires no repeated solver calls. If the
correction is uncorrelated or anti-correlated with the detected stripe component,
the clipped solution is `alpha=0`.

The calculation uses normalized energy ratios and robust image statistics. It
does not introduce an image-specific threshold. Adaptive levels continue to set
the solver's `mu1`; the controller independently limits how much of that solver
result is safe to apply.

## Data Flow and Compatibility

`refine_clean()` remains the integration boundary used by `ops.py`, but its
implementation changes from unconditional residual subtraction to the safety
controller. Its existing arguments remain sufficient because correction
selection is data-driven and the adaptive level has already affected the solver.

Manual mode does not call the controller. Tiled mode still runs one batched PDHG
solve and then evaluates the blended result at processed-image resolution.
`process_size` behavior is unchanged. Clipping to `[0, 1]` occurs only when
`proj=True`, matching current behavior.

No new runtime dependency is introduced. NumPy, OpenCV, and PyTorch already cover
the required vectorized operations.

## Edge Cases and Failure Handling

- Constant images continue to return a copy before adaptive estimation.
- Images smaller than the protection or blur support use reduced kernels and the
  existing small-image high-pass fallback.
- Empty safe regions yield zero reliability and therefore `alpha=0`.
- Zero-energy or non-finite intermediate ratios are replaced by safe zero values;
  the controller never emits NaN or Inf.
- Multiple selected directions are evaluated independently, then combined using
  their reliability weights.
- A full-frame real line structure exactly aligned with a candidate stripe can
  remain ambiguous. The continuous blend and original-image fallback bound the
  damage but cannot prove semantic identity without additional observations.

## Testing

### Unit tests

- Weighted robust projection recovers a faint line profile in the presence of a
  localized high-amplitude structure.
- Protection is low on a constant directional curtain and high on curved or
  crossing structure.
- Split-half and scale reliability are high for a weak repeated curtain and low
  for inconsistent texture.
- Analytic selection returns nearly full correction for a matching unprotected
  stripe, reduces correction when energy overlaps protected structure, and
  returns zero for an uncorrelated correction.
- Selection is deterministic and invariant to affine intensity scaling.
- Existing projection, dtype, shape, RGB, tiling, and manual-mode tests remain
  unchanged and passing.

### Synthetic acceptance benchmark

Use `sample_02` through `sample_05` as clean targets and keep `sample_01` as
real-only visual evidence.

At `adaptive=2`, `iterations=500`, and `process_size=256`:

- mean PSNR and mean SSIM change for curtain patterns at `strength=0.01` must
  both be non-negative;
- mean PSNR gains for `strength=0.03` and `0.06` must not regress by more than
  `0.25 dB` from the recorded baselines (`+0.919 dB` and `+3.783 dB`);
- the mean output-to-input PSNR on unmodified `sample_02` through `sample_05`
  must exceed the current mean baseline of `33.386 dB`;
- all outputs must remain finite and within the projected range.

These are aggregate gates because an individual synthetic line can be
mathematically ambiguous with real structure. Per-pattern results remain in the
CSV for inspection.

### Performance acceptance

Measure median wall-clock time over repeated CPU runs on `sample_02` through
`sample_05` before and after the controller change. The new median must be no
more than `15%` slower with `process_size=256` and `iterations=500`. Timing uses
a warm-up run and excludes image loading.

## Documentation

Update the README adaptive-mode section to explain robust structure protection,
repeatability evidence, and automatic correction blending. Document that
adaptive processing is fully automatic at runtime and still cannot distinguish
perfectly stripe-like semantic structures from a single image.
