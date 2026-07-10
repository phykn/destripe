# Quality-First H3-Guided PDHG Destriping Design

## Goal

Replace direct full-line H3 subtraction with a parameter-free public automatic
path that uses H3 only for detection and uses PDHG for the actual spatially local
correction. Image quality is the acceptance criterion. Runtime is measured and
reported, but it is not a pass/fail gate.

The public API remains:

```python
clean = destripe(image, process_size=512)
```

No adaptive level, `mu1`, `mu2`, direction, tile, or iteration option is exposed
through `destripe()`.

## Evidence Behind the Change

The current H3 implementation estimates one value per stripe line and subtracts
that value along the complete line. On a synthetic vertical curtain present only
in the outer quarters, it reduced error in the striped region but introduced an
RMS error of `0.01315` into the clean middle half. The applied correction in the
clean and striped regions had the same RMS, confirming that the failure is caused
by the full-line model rather than a threshold.

The previous automatic adaptive-PDHG notebook produced visually preferable local
corrections, but its exact configuration took about `6.94 s` on sample 01 and
used two directions, 2x2 tile parameters, and 1000 iterations. A selected
single-direction PDHG solve produced local corrections in about `0.42-0.60 s`
in diagnostic runs. These timings are observations, not quality requirements.

## Architecture

The automatic path has four stages:

1. H3 detection and safety gating.
2. Data-driven PDHG candidate generation.
3. Analytic correction scaling and self-consistency selection.
4. Shape, channel, projection, and dtype restoration in the existing wrapper.

`UniversalStripeRemover` remains the only PDHG implementation. The hybrid path
does not duplicate solver mathematics.

## Stage 1: H3 Detection and Safety Gate

H3 continues to evaluate all five directions using robust protected line
profiles and scale repeatability. Its blocked repeatability changes from a
comparison of only the first and last quarters to four along-line quarters.

For each direction:

- estimate one protected robust profile in each quarter;
- center the four profiles across line identifiers;
- calculate all six positive pairwise cosine similarities;
- use their minimum as blocked consistency; and
- combine blocked consistency with scale repeatability using the existing H3
  reliability formula.

If any quarter lacks consistent support, blocked consistency becomes zero and
the automatic path returns the input unchanged. It does not extrapolate a stripe
through an unsupported middle region.

The selected H3 result supplies:

- one direction;
- a protected global stripe profile;
- reliability;
- the existing analytic H3 target strength; and
- the structure-protection map.

H3 no longer supplies the final clean image.

## Stage 2: PDHG Parameter Candidates

Only the H3-selected direction is sent to PDHG. There is no multi-direction or
tile-local automatic solve.

The old robust strength estimator is restored as an internal initializer,
restricted to the selected direction. It is not restored as a public adaptive
level API.

Parameter candidates are generated as follows:

- `mu1` uses the previously stable normalized-image candidates
  `{1/6, 1/5, 1/4, 1/3}`;
- `mu2` starts from the robust image-specific strength estimate and evaluates
  its adjacent lower and higher strength candidates within the solver's proven
  normalized range;
- duplicate boundary candidates are removed; and
- each pair runs the same single-direction PDHG solver with convergence-based
  early stopping and a maximum-iteration safety cap.

The returned `mu1` and `mu2` are therefore selected per image. The fixed values
are limited to a small safety search space inherited from the previously working
solver, rather than one globally forced parameter pair.

## Stage 3: Analytic Scaling and Candidate Selection

Let `P` be the H3 target correction and `C` be a PDHG candidate correction.
The target projection is `dot(P, P)` and the candidate projection is
`dot(C, P)`. Each candidate receives an analytic scalar `beta` that matches its
projection onto `P` without exceeding the H3 target:

```text
beta = clip(target_projection / candidate_projection, 0, 1)
scaled_correction = beta * C
```

Candidates with non-positive projection onto the H3 target cannot explain the
detected stripe and are rejected.

Selection is lexicographic, not a weighted score:

1. maximize the fraction of the H3 target projection explained;
2. minimize correction energy inside the H3 structure-protection map;
3. minimize total correction energy; and
4. break exact ties deterministically by smaller parameter tuple.

This avoids fitted objective weights. It also chooses the least destructive
candidate among candidates that explain the same amount of detected stripe.

The output is:

```text
clean = input - scaled_correction
```

Projection to `[0, 1]` is applied only when `proj=True`.

## Failure and Fallback Behavior

The input is returned unchanged when:

- H3 finds no positive four-quarter consistency;
- analytic H3 target strength is zero;
- no PDHG candidate has positive projection onto the H3 target;
- or every PDHG candidate fails or contains non-finite values.

Candidate failures are isolated. One failed candidate does not invalidate other
candidates. If all candidates fail, the operation is a no-op rather than a
full-line fallback.

## Quality Acceptance

Quality, not runtime, determines acceptance.

### Real SEM visual reference

For sample 01:

- compare original, previous adaptive-PDHG, current H3, and hybrid output at the
  same display range;
- show signed corrections using the same color limits;
- reject full-height line corrections, new banding, particle-edge halos, and
  broad contrast shifts; and
- require the hybrid result to be visually no worse than the previous
  adaptive-PDHG reference.

### Ground-truth synthetic cases

Add continuous and interrupted curtains for all five directions. Interrupted
cases include:

- outer quarters with a clean middle half;
- one-sided half-image support;
- a centered finite segment; and
- deterministic multiple separated segments.

For unsupported clean regions, output error must not exceed input error. Because
the input error is zero there, a safety-gated no-op is the expected behavior.

Continuous weak, medium, and strong curtains must improve PSNR and SSIM without
weakening clean-image fidelity. Existing three benchmark seeds remain regression
seeds; at least one newly generated interruption-mask seed is held back until the
algorithm and parameter search are frozen.

### Clean fidelity

Samples 02 through 05 retain the existing minimum clean gates of `40 dB` PSNR
and `0.99` SSIM. Per-sample metrics and worst cases are reported, not only means.

## Performance Reporting

Report separately:

- H3 detection time;
- parameter-candidate PDHG time;
- total wrapper time;
- selected candidate count; and
- early-stop iteration counts.

No runtime threshold can override a quality failure. Candidate pruning and
optimization are allowed only after the frozen quality suite passes.

## Scope and Simplicity Constraints

- No public adaptive levels or manual solver parameters on `destripe()`.
- No learned model, fitted dataset weights, spatial neural mask, or per-pattern
  branch.
- No tile-local automatic parameters.
- One H3-selected direction only.
- Reuse the existing PDHG core and the smallest necessary robust strength
  initializer.
- Do not tune after viewing the held-back interruption seed.

## Documentation and Notebook

The notebook must show:

- original image;
- previous adaptive-PDHG visual reference;
- final hybrid output;
- signed hybrid correction; and
- selected direction, `mu1`, `mu2`, H3 reliability, analytic scale, and PDHG
  convergence count as diagnostics, not user inputs.

The README describes the automatic hybrid path and keeps direct
`UniversalStripeRemover` usage as the explicit manual alternative.
