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

## Independent Review Outcome

An independent code-and-spec review rejected the first controller draft for four
reasons that this revision addresses:

- mean PSNR/SSIM gates allowed an all-no-op controller to pass without removing
  weak stripes;
- penalizing correction energy wherever an SEM edge exists also penalized a
  correct stripe crossing that edge;
- reliability was applied to both proposal construction and selection, causing
  weak evidence to be suppressed more than once;
- collapsing all solver components into one correction prevented local
  structural leakage from being attributed to its responsible direction.

The reviewer also evaluated the rejected protection-energy penalty on the four
clean SEM samples with exact synthetic corrections. Mean protection was
approximately `0.57..0.72`; even a perfect correction with reliability `1` would
have been reduced to approximately `alpha=0.28..0.43`. This revision removes that
penalty entirely and tests protected parallel curvature instead.

## Goals

- Remove faint, line-coherent stripes without requiring ground truth or per-image
  parameter tuning.
- Preserve SEM edges and texture by reducing corrections that overlap
  direction-inconsistent image structure.
- Run the PDHG solver exactly once per existing solver invocation.
- Select each direction's applied correction continuously from the closed
  interval `[0, 1]`, where all-zero coefficients produce the unchanged input and
  all-one coefficients produce the full proposed correction.
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
  -> one PDHG solve retaining direction-specific stripe components
  -> direction-aware structure protection
  -> robust residual stripe proposal
  -> per-direction analytic correction-strength selection
  -> protected adaptive output
```

The new controller operates at `processed_gray` resolution. The solver already
maintains one stripe component for each selected direction; adaptive mode retains
those components instead of collapsing them into `gray - solver_clean`. Existing
resize-back logic then transfers the selected combined stripe field to the
original resolution. RGB handling remains luma-based and therefore automatically
receives the same safety decision for every channel.

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

- `0` means the location is safe for estimating a stripe profile;
- `1` means the location is likely real structure and should be excluded from
  profile estimation and used to check the component for structural leakage.

The calculation is direction-specific. A cross-direction stripe boundary must
not be protected merely because it has a large gradient; otherwise the detector
would protect the artifact it is intended to remove.

The protection map is not multiplied into the final correction. A real stripe
continues underneath a particle edge, so spatially zeroing the correction at the
edge would both leave stripe residue and create seams.

### Robust line projection

Extend `src/destripe/adaptive/stripe.py` with weighted, robust projection while
preserving the existing unweighted `project()` behavior.

The robust projector uses two scatter-based passes:

1. Compute a protection-weighted mean for every directional line.
2. Estimate each line's weighted absolute residual scale, apply standard Huber
   weights with cutoff `1.345`, and recompute the line value.

This excludes localized SEM edges from profile estimation without Python loops
over pixels or lines. The estimated line value is then expanded over the full
line, including protected pixels, because a valid stripe does not stop at an SEM
edge. The existing split-half reliability calculation gains an optional weight
map so that protected structure does not inflate stripe coherence.

### Repeatability evidence

For each selected direction, calculate:

- split-half repeatability using alternating samples along the stripe direction;
- scale repeatability between the normal high-pass image and the high-pass image
  after a Gaussian blur with `sigma=1`;
- a combined reliability equal to the geometric mean of the two values.

Reliability is based on correlation rather than stripe amplitude. Consequently,
a faint but repeated curtain can remain reliable, while a strong localized edge
does not automatically qualify as a stripe.

### Direction-specific correction proposals

Adaptive solving returns the clean estimate and the already-computed solver
stripe component `s_d` for every selected direction. The controller also robustly
projects the remaining high-pass content of `solver_clean` to obtain a residual
line proposal `e_d`. The full proposal for that direction is

```text
c_d = s_d + e_d.
```

Neither `s_d` nor `e_d` is multiplied by reliability or spatially masked.
Reliability is used exactly once when selecting the final coefficient. This
avoids suppressing weak evidence multiple times and still lets a valid stripe be
corrected where it crosses protected structure.

This keeps the current ability to remove weak residual curtains, preserves the
solver's exact directional decomposition, and prevents unrelated structure in a
collapsed global correction from being accepted merely because another region
contains a stripe.

### Per-direction analytic correction selection

Candidate outputs form the continuous family

```text
y(alpha) = gray - sum_d alpha_d*c_d,  alpha_d in [0, 1].
```

For each direction, robustly project the high-pass input to `p_d` and the
high-pass proposal to `q_d`. Let `r_d` be repeatability and
`N_d = ||p_d||^2 + EPS`.

Raw correction energy at a protected pixel is not a defect: a correct stripe
must pass through SEM structure. Instead, structural leakage is measured from
the proposal's second difference along the stripe direction. Slowly varying or
line-constant stripe components have low parallel curvature, whereas a component
that has absorbed a particle boundary has localized high curvature. With
directional protection map `h_d`, define

```text
L_d = ||h_d * second_parallel_diff(c_d)||^2 / (||q_d||^2 + EPS)
A_d = ||q_d||^2 / N_d + L_d
B_d = r_d * <p_d, q_d> / N_d
alpha_d = clip(B_d / (A_d + EPS), 0, 1).
```

Reliability appears only in `B_d`. It therefore reduces the applied amount when
evidence is uncertain instead of cancelling between numerator and denominator.
For a perfectly matching, repeatable, line-constant correction, `alpha_d=1`
even where the stripe crosses protected edges. An uncorrelated or
anti-correlated proposal gives `alpha_d=0`.

Directions are selected sparsely by the existing estimator and their solver
components are already separated, so coefficients are calculated independently.
The final sum is clipped only when `proj=True`. The calculation requires no
candidate images, no image-specific threshold, and no repeated solver call.
Adaptive levels continue to set the solver's `mu1`; the controller independently
limits how much of each solver component is safe to apply.

## Data Flow and Compatibility

`UniversalStripeRemover` gains an internal adaptive solve path that returns the
clean estimate plus one stripe component per selected direction. Existing public
`process()` and `process_tiled()` return values remain unchanged. Tiled adaptive
solving blends every component with the same cosine window used for the clean
estimate so their spatial alignment and reconstruction identity are preserved.

`refine_clean()` remains the integration boundary used by `ops.py`, but it now
receives the direction-specific solver components and delegates protection,
residual proposal, and coefficient selection to `safety.py`.

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
- Empty safe regions yield zero reliability and therefore `alpha_d=0`.
- Zero-energy or non-finite intermediate ratios are replaced by safe zero values;
  the controller never emits NaN or Inf.
- Multiple selected directions retain separate solver components, coefficients,
  and diagnostics before their accepted corrections are summed.
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
- Solver and tiled-solver adaptive paths return one component per direction and
  preserve `clean + sum(components) == input` within floating-point tolerance.
- Analytic selection returns nearly full correction for a matching stripe even
  when it crosses protected structure, reduces a component with high protected
  parallel curvature, scales with reliability exactly once, and returns zero for
  an uncorrelated correction.
- Selection is deterministic and invariant to affine intensity scaling.
- Existing projection, dtype, shape, RGB, tiling, and manual-mode tests remain
  unchanged and passing.

### Synthetic acceptance benchmark

Use `sample_02` through `sample_05` as clean targets and keep `sample_01` as
real-only visual evidence.

Extend the benchmark with unmodified clean cases, multiplicative curtains,
multiple band widths, and off-grid oblique directions. Keep seed `1234` for
development comparison and use seed `20260710` only as a held-out acceptance
run. At `adaptive=2`, `iterations=500`, and `process_size=256`:

- mean PSNR gain for curtain patterns at `strength=0.01` must be at least
  `0.10 dB`, and mean SSIM gain must be at least `0.001`;
- at least `75%` of weak-curtain cases must individually gain at least `0.05 dB`
  PSNR and `0.0001` SSIM, no weak case may lose more than `1.0 dB` PSNR, and
  every supported direction must have a non-negative mean PSNR change;
- mean `stripe_projection_left_pct` for weak additive curtains must be at most
  `70%`, and at least `75%` of those cases must individually be at most `85%`,
  preventing all-no-op and mostly-no-op controllers from passing;
- mean PSNR gains for `strength=0.03` and `0.06` must not regress by more than
  `0.25 dB` from the recorded baselines (`+0.919 dB` and `+3.783 dB`);
- output-to-input fidelity on every unmodified `sample_02` through `sample_05`
  must reach at least `40 dB` PSNR and `0.99` SSIM. This is stronger than merely
  matching the current `31.438`, `35.890`, `30.714`, and `35.500 dB` baselines;
- all outputs must remain finite and within the projected range.

Both development and held-out seeds must pass. Aggregate, per-case, and
per-direction gates are all required because aggregate metrics alone can hide a
large failure in one image, direction, or pattern family. Per-pattern results
remain in the CSV for inspection.

### Performance acceptance

Measure median wall-clock time over repeated CPU runs on `sample_02` through
`sample_05` before and after the controller change. The measurement includes
retaining and blending direction-specific components. The new median must be no
more than `15%` slower with `process_size=256` and `iterations=500`. Timing uses
a warm-up run and excludes image loading.

## Documentation

Update the README adaptive-mode section to explain robust structure protection,
repeatability evidence, and automatic correction blending. Document that
adaptive processing is fully automatic at runtime and still cannot distinguish
perfectly stripe-like semantic structures from a single image.
