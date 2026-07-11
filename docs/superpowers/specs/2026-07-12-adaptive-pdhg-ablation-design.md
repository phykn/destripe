# Adaptive-PDHG Baseline and H3 Ablation Design

## Goal

Replace the current H3-guided single-direction hybrid with the smallest automatic
PDHG pipeline that is demonstrably no worse than the previously preferred
adaptive-PDHG result. H3 code survives only when an isolated ablation proves a
material quality benefit over that baseline.

The public API remains parameter-free:

```python
clean = destripe(image, process_size=256)
```

No adaptive level, direction, `mu1`, `mu2`, tile, or iteration option is restored
to `destripe()`.

## Frozen Baseline

The mandatory visual and quantitative baseline is the former adaptive path with:

- adaptive level `2`;
- analysis long edge `512`;
- `1000` PDHG iterations;
- `2x2` tiles with the old tile-local `mu2` estimates;
- selected directions `(0, 4)` on sample 01; and
- the former adaptive refinement/safety behavior.

The exact implementation is recoverable from `stash@{0}` and its parent history.
The baseline must be restored without mixing in current H3 code. Frozen outputs
and metrics are generated before any ablation is implemented.

## Candidate Architectures

Only four candidates are eligible:

1. **A — Adaptive baseline:** old adaptive direction scoring, parameter estimation,
   multi-direction PDHG, component handling, tile-local parameters when needed,
   and adaptive safety/refinement.
2. **B — Adaptive plus H3 protection:** A with one structure-protection map used
   to suppress correction on SEM structure.
3. **C — Adaptive plus soft repeatability:** A with H3 scale and regional
   repeatability used only as per-direction continuous weights.
4. **D — Adaptive plus protection and soft repeatability:** B and C together.

There is no H3 parameter grid, H3 target scaling, analytic beta, hard consistency
gate, single-direction final decision, or image-wide no-op threshold in these
candidates.

## Final Data Flow

The winning candidate has this shape:

```text
normalize and resize
-> adaptive direction scoring and parameter estimation
-> multi-direction PDHG with direction components
-> optional proven structure protection
-> optional proven per-direction soft repeatability
-> sum independently weighted components
-> restore original size, range, shape, and dtype
```

Each direction is controlled independently:

```python
correction = sum(alpha[direction] * component[direction] for direction in directions)
clean = input_image - correction
```

A weak direction cannot disable correction from a better-supported direction.
The PDHG components, not an H3 target, define the available correction magnitude.

## Module Boundaries

The intended final source structure is:

```text
src/destripe/
    ops.py
    automatic.py
    preprocess.py
    core/
    adaptive/
        estimate.py
        directions.py
        strength.py
        evidence.py
        safety.py
        stripe.py
```

- `automatic.py` orchestrates the internal automatic flow and exposes only the
  diagnostics actually consumed by benchmarks or the notebook.
- `estimate.py`, `directions.py`, and `strength.py` estimate directions and
  regularization parameters without public levels.
- `stripe.py` owns robust line projection and shared directional primitives.
- `evidence.py` owns the single retained protection implementation and soft
  repeatability, if their ablations win.
- `safety.py` converts component-local evidence into continuous direction weights.
- `core/` contains PDHG operations and component output only.

Duplicate projection, line-ID, and directional-difference implementations are
consolidated. Compatibility code for removed public adaptive arguments is not
restored.

## Ablation Protocol

### Comparison data

Every candidate uses identical input arrays, normalization, analysis size, random
seeds, and display limits. The fixed comparison set includes:

- real SEM sample 01;
- clean SEM samples 02 through 05;
- weak, medium, and strong continuous curtains;
- partially supported curtains with clean gaps;
- vertical and diagonal directions; and
- a held-back interruption seed used only after candidate logic is frozen.

Every visual comparison includes `correction = input - output`. Visible particles,
edges, broad bands, halos, or contrast shifts in correction count as structural
damage.

### Candidate selection rule

An H3 element is retained only when it satisfies all of the following relative
to A:

- stripe residual is equal or lower;
- clean-image PSNR and SSIM do not regress;
- particle-edge and broad-band correction does not increase;
- unsupported-region MSE does not increase; and
- improvement is reproducible across regression seeds rather than isolated to
  one sample or direction.

If no candidate beats A without a regression, A wins and all H3 production code
is deleted. There are no weighted aggregate scores that can hide a failed gate.

## Quality Gates

The final winner must satisfy:

- sample 01 visual stripe residual no worse than the frozen adaptive baseline;
- clean samples at least `40 dB` PSNR and `0.99` SSIM, and no worse than A;
- continuous stripes non-regressing against A in PSNR and SSIM at every tested
  strength;
- unsupported-region MSE no higher than A for partial stripes;
- correction-map structure leakage no higher than A;
- identical shape and dtype restoration for grayscale, single-channel, and RGB;
- full unit, integration, benchmark, and notebook execution success.

Runtime is reported but is not a pass/fail criterion.

## Failure Handling

Input validation remains explicit. Internal candidate or component failures do
not silently produce an H3 full-line fallback. A solver failure returns the
unmodified input only when no valid adaptive correction is available, and the
failure is surfaced in diagnostics used by tests.

## Deletion After Selection

After the winner is frozen, remove:

- `src/destripe/hybrid.py`;
- H3 candidate selection and parameter grid search;
- H3 target, beta, target-energy cap, and `sqrt(reliability)` scaling;
- H3 single-direction and hard-gate constants;
- unused diagnostic fields and dataclasses;
- `tests/test_hybrid.py` and obsolete H3-only behavior tests;
- losing protection or repeatability implementations;
- README and notebook descriptions of the H3-guided hybrid;
- comparison-only notebook branches; and
- all legacy public adaptive-level compatibility code.

Git history remains the archive for rejected implementations.

## Delivery

Work proceeds directly on `main`, without independent agents, as explicitly
requested. The final notebook is executed top-to-bottom, all tests and quality
gates are rerun, only `main` remains locally, and the final commits are pushed to
`origin/main`. Existing untracked benchmark CSV files are preserved and never
staged.
