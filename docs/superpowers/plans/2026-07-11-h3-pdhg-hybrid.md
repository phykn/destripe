# H3-Guided PDHG Hybrid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan.

**Goal:** Replace automatic full-line H3 subtraction with a quality-first, parameter-free H3-guided single-direction PDHG path that preserves clean and interrupted regions.

**Architecture:** H3 detects the direction, builds a protected target profile, and rejects directions that are not consistent across four along-line quarters. A small deterministic set of image-derived `mu1`/`mu2` candidates is solved by the existing PDHG core. Each correction is analytically scaled to the H3 target and selected lexicographically by explained target, protected leakage, then correction energy. The public `destripe()` signature remains unchanged.

**Tech Stack:** Python 3, NumPy, PyTorch, OpenCV, pytest, Jupyter notebook, Ruff.

**Global Constraints:** Work directly on `main` because the user explicitly requires main-only branch retention. Do not use subagents. Preserve the two untracked benchmark CSVs. Treat quality as the acceptance gate; record runtime but do not reject on runtime. Use the repository-local `.\.venv\Scripts\python.exe` for every Python command.

---

### Task 1: Freeze interrupted-stripe safety cases

**Files:**
- Modify: `benchmarks/synthetic.py`
- Modify: `tests/test_synthetic_benchmark.py`
- Modify: `tests/test_automatic.py`

**Step 1: Write the failing generator tests**

Add `make_support_mask(shape, kind, mode, rng)` tests covering:

```python
@pytest.mark.parametrize("mode", range(5))
@pytest.mark.parametrize("kind", ("outer_quarters", "first_half", "center", "segments"))
def test_interrupted_support_mask_has_clean_and_active_regions(kind, mode):
    mask = make_support_mask((96, 104), kind=kind, mode=mode,
                             rng=np.random.default_rng(731))
    assert mask.shape == (96, 104)
    assert mask.dtype == np.float64
    assert np.any(mask == 0.0)
    assert np.any(mask == 1.0)
```

Add an automatic-path regression in `tests/test_automatic.py` for the vertical outer-quarter case. Assert exact no-op in the unsupported middle and no worse MSE than input over the full image.

**Step 2: Run tests to verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_synthetic_benchmark.py tests/test_automatic.py -q
```

Expected: failure because `make_support_mask` does not exist and current H3 writes through the clean middle.

**Step 3: Implement deterministic support masks**

In `benchmarks/synthetic.py`, add a geometry-only helper based on normalized along-line position from `PARALLEL_OFFSETS`. Implement exactly four kinds:

- `outer_quarters`: `[0, .25] U [.75, 1]`
- `first_half`: `[0, .5]`
- `center`: `[.25, .75]`
- `segments`: three deterministic non-touching intervals drawn from the provided RNG

The function must validate shape, kind, mode, and return only `0.0`/`1.0`. Do not add these cases to the default benchmark matrix yet; the helper is for a separate safety suite so existing frozen pattern hashes stay unchanged.

**Step 4: Run the generator tests**

Run the focused tests. Expected: mask tests pass; automatic interrupted regression remains RED.

**Step 5: Commit**

```powershell
git add benchmarks/synthetic.py tests/test_synthetic_benchmark.py tests/test_automatic.py
git commit -m "test: freeze interrupted stripe safety cases"
```

---

### Task 2: Refactor H3 into detection and four-quarter gating

**Files:**
- Modify: `src/destripe/automatic.py`
- Modify: `tests/test_automatic.py`

**Step 1: Write failing detector tests**

Add tests for a private immutable `H3Detection` result with fields:

```python
direction: int
target: np.ndarray
protection: np.ndarray
reliability: float
alpha: float
consistent: bool
```

Test that:

- a continuous vertical curtain selects direction `0`, has positive reliability/alpha, and `consistent is True`;
- outer-quarter, first-half, centered, and separated-segment curtains return `consistent is False` and zero alpha;
- a constant image deterministically selects direction `0` and returns zero target/alpha;
- all five directions are still evaluated exactly once.

**Step 2: Run tests to verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_automatic.py -q
```

Expected: failures because the detector and four-quarter behavior do not exist.

**Step 3: Implement four-quarter repeatability**

Replace `_blocked_repeatability` with four quarter profiles computed over normalized along-line position. Center each profile on jointly usable line IDs, calculate all six pairwise positive centered cosines, and return the minimum. Return zero when any pair has fewer than two jointly supported lines or zero variance.

**Step 4: Separate detection from cleaning**

Add `_detect_h3(gray_array) -> H3Detection`. Move profile, protection, scale repeatability, leakage, and alpha computation into it. Store `target = alpha * selected_profile`. Set `consistent` only when blocked consistency, reliability, alpha, and target power are positive and finite.

Keep `automatic_clean()` temporarily compatible by subtracting `detection.target` only when consistent. This intermediate behavior is replaced in Task 5.

**Step 5: Run tests and inspect the interrupted regression**

Run the focused automatic tests. Expected: all pass, including exact no-op for every interrupted mask.

**Step 6: Commit**

```powershell
git add src/destripe/automatic.py tests/test_automatic.py
git commit -m "fix: gate H3 detection across four quarters"
```

---

### Task 3: Add the smallest data-driven parameter initializer

**Files:**
- Create: `src/destripe/hybrid.py`
- Create: `tests/test_hybrid.py`
- Reference only: `stash@{0}` old adaptive estimator, without restoring its public package/API

**Step 1: Write failing candidate-generation tests**

Define immutable internal records:

```python
@dataclass(frozen=True)
class ParameterCandidate:
    mu1: float
    mu2: float
```

Test that `_parameter_candidates(gray, direction, target)`:

- returns the Cartesian product of four `mu1` values `(1/6, 1/5, 1/4, 1/3)` and at most three unique positive finite `mu2` values;
- is deterministic for the same input;
- changes at least one `mu2` value when stripe strength changes materially;
- clips `mu2` to the normalized solver-safe range;
- never exposes an adaptive level, tile parameter, or direction search.

**Step 2: Run tests to verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_hybrid.py -q
```

Expected: import failure because `hybrid.py` does not exist.

**Step 3: Restore only robust strength estimation**

Inspect the old estimator in `stash@{0}` and copy only the robust scale calculation needed to initialize `mu2`. Remove level maps, tile logic, multi-direction ranking, and confidence scoring. Express all constants as named module constants with comments tying them to normalized solver bounds.

Generate adjacent `mu2` candidates around the estimate using a fixed multiplicative neighborhood, clip them to the proven normalized range, deduplicate after clipping, and sort ascending. This is a small safety search, not fitted per-dataset branching.

**Step 4: Run tests**

Run `tests/test_hybrid.py`. Expected: candidate tests pass.

**Step 5: Commit**

```powershell
git add src/destripe/hybrid.py tests/test_hybrid.py
git commit -m "feat: derive compact PDHG parameter candidates"
```

---

### Task 4: Expose used PDHG convergence diagnostics

**Files:**
- Modify: `src/destripe/core/remover.py`
- Modify: `tests/test_core.py`

**Step 1: Write failing solver-result tests**

Add a private immutable `_SolveResult(clean: torch.Tensor, iterations: int)` and test:

- `_solve()` reports `1 <= iterations <= max_iterations`;
- early stopping reports fewer than the cap on a converged constant image;
- public `process()` still returns only a tensor and preserves all existing behavior;
- tiled and explicit tile-mu paths consume `.clean` and do not leak the private record publicly.

**Step 2: Run tests to verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_core.py -q
```

Expected: failures because `_solve()` currently returns only a tensor.

**Step 3: Return the private result from `_solve()`**

Track the one-based executed iteration count in the existing loop. Return `_SolveResult`. Update `process()` and every tiled call site to use `.clean`. Add a private `_process_with_info()` helper used by the hybrid path; keep public `process()` unchanged.

Do not introduce mutable `last_iterations` state or duplicate the PDHG solver.

**Step 4: Run core tests**

Run `tests/test_core.py`. Expected: all pass.

**Step 5: Commit**

```powershell
git add src/destripe/core/remover.py tests/test_core.py
git commit -m "refactor: report internal PDHG convergence"
```

---

### Task 5: Implement analytic hybrid selection

**Files:**
- Modify: `src/destripe/hybrid.py`
- Modify: `src/destripe/automatic.py`
- Modify: `tests/test_hybrid.py`
- Modify: `tests/test_automatic.py`

**Step 1: Write failing scaling and selection tests**

Add immutable diagnostics:

```python
@dataclass(frozen=True)
class HybridDiagnostics:
    candidate_count: int
    mu1: float | None
    mu2: float | None
    beta: float
    iterations: int
```

Test pure helpers for:

- rejection of non-positive `dot(C, P)`;
- `beta = clip(dot(P,P) / dot(C,P), 0, 1)`;
- lexicographic preference for larger explained fraction;
- protected leakage tie-break before total correction energy;
- deterministic smaller `(mu1, mu2)` final tie-break;
- all-candidate failure returning an exact no-op and empty diagnostics.

Monkeypatch the solver for unit tests so selection logic is independent of PDHG runtime.

**Step 2: Run tests to verify RED**

Run `tests/test_hybrid.py tests/test_automatic.py`. Expected: selection and PDHG-core assertions fail.

**Step 3: Implement the hybrid runner**

Implement `_run_hybrid(gray, detection, proj)`:

1. Return no-op if `detection.consistent` is false or target power is non-positive.
2. Generate the compact parameter set.
3. For each candidate, construct `UniversalStripeRemover(mu1, mu2, directions=[direction])` and run private convergence-aware processing with a named maximum cap and tolerance.
4. Compute `C = gray - candidate_clean`.
5. Reject exceptions, non-finite corrections, and non-positive target projection.
6. Scale by analytic beta.
7. Rank by `(-explained_fraction, protected_energy, total_energy, mu1, mu2)`.
8. Apply the selected correction and project only when requested.

Explained fraction is capped at one after scaling. Energies are normalized by pixel count to keep comparisons shape-independent. Catch only candidate-local numerical/runtime errors; do not swallow input validation errors.

**Step 4: Make PDHG the automatic correction core**

Change `automatic_clean()` to call `_detect_h3`, then `_run_hybrid`. Extend `AutomaticResult` with diagnostics used by the notebook:

```python
reliability: float
mu1: float | None
mu2: float | None
beta: float
iterations: int
candidate_count: int
```

Keep existing `clean`, `direction`, and `alpha`. Do not add arguments to `automatic_clean()` or `destripe()`.

**Step 5: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_hybrid.py tests/test_automatic.py tests/test_ops.py -q
```

Expected: all pass. Verify interrupted cases are exact no-ops and continuous vertical cases improve MSE.

**Step 6: Commit**

```powershell
git add src/destripe/hybrid.py src/destripe/automatic.py tests/test_hybrid.py tests/test_automatic.py tests/test_ops.py
git commit -m "feat: use H3-guided PDHG automatic cleaning"
```

---

### Task 6: Build and freeze the quality suite

**Files:**
- Create: `benchmarks/hybrid_quality.py`
- Modify: `tests/test_synthetic_benchmark.py`
- Modify: `benchmarks/acceptance.py`

**Step 1: Write failing quality-report tests**

Test that the report includes:

- per-sample clean PSNR/SSIM for samples 02-05;
- continuous weak/medium/strong curtains for all five directions;
- all four interrupted supports for all five directions;
- active-region and unsupported-region MSE separately;
- detector time, candidate PDHG time, total time, candidate count, and iteration count;
- explicit worst-case rows rather than mean-only output.

**Step 2: Implement the runner**

Reuse `load_samples`, `make_stripe_pattern`, `make_support_mask`, `inject_stripe`, and `structural_similarity`. Do not alter frozen canonical pattern bytes. Use seeds `1234`, `20260710`, and the existing third regression seed. Keep one new interruption-mask seed in a named constant but do not run it during tuning.

**Step 3: Define quality gates**

Add checks:

- clean sample PSNR `>= 40 dB` and SSIM `>= .99`;
- unsupported-region output MSE `<=` input MSE plus floating tolerance;
- continuous cases improve aggregate PSNR and SSIM at each strength;
- no runtime pass/fail threshold.

**Step 4: Run the suite before the held-back seed**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_synthetic_benchmark.py -q
.\.venv\Scripts\python.exe -m benchmarks.hybrid_quality --asset-dir asset --process-size 256
```

If quality fails, adjust only general detector/candidate constants with evidence from the regression seeds. Do not add image-name or pattern-kind branches.

**Step 5: Freeze and run the held-back interruption seed once**

After code and constants are frozen, run the held-back seed. If it fails, report the failure and revise the design rather than tuning to that seed.

**Step 6: Commit**

```powershell
git add benchmarks/hybrid_quality.py benchmarks/acceptance.py tests/test_synthetic_benchmark.py
git commit -m "test: validate hybrid quality and interruption safety"
```

---

### Task 7: Update documentation and execute the notebook

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml`
- Modify and execute: `notebooks/test.ipynb`
- Modify: `tests/test_synthetic_benchmark.py`

**Step 1: Update documentation assertions first**

Change README/notebook source assertions to require “H3-guided PDHG”, parameter-free automatic usage, single selected direction, and interruption safety. Continue rejecting removed adaptive levels, tile-mu options, and public manual parameters.

**Step 2: Update README and project description**

Explain that H3 performs detection/safety gating while PDHG produces the local 2D correction. Keep the manual `UniversalStripeRemover` example as the explicit advanced path.

**Step 3: Rebuild notebook cells**

The notebook must display, at identical intensity ranges:

- original sample 01;
- previous adaptive-PDHG visual reference generated with the frozen manual reference configuration;
- final hybrid output;
- signed hybrid correction with symmetric color limits;
- direction, H3 reliability/alpha, selected `mu1`/`mu2`, beta, candidate count, iterations, and elapsed time.

Do not expose diagnostics as inputs. Preserve the user's current execution-count-only notebook changes by replacing the notebook intentionally and executing the final source.

**Step 4: Execute and validate notebook**

Run:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/test.ipynb --output test.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=600
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

Inspect rendered notebook output for full-height correction, new banding, edge halos, and broad contrast shifts. If visual quality is worse than the frozen adaptive reference, do not declare success.

**Step 5: Commit**

```powershell
git add README.md pyproject.toml notebooks/test.ipynb tests/test_synthetic_benchmark.py
git commit -m "docs: demonstrate H3-guided PDHG hybrid"
```

---

### Task 8: Final verification and main-only delivery

**Files:**
- Verify all tracked changes
- Preserve untracked CSVs

**Step 1: Run final evidence commands**

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m benchmarks.hybrid_quality --asset-dir asset --process-size 256
git status --short
git branch --list
```

Expected: tests and quality gates pass; only `main` exists locally; the two user CSVs remain untracked and untouched.

**Step 2: Review diff and history**

Confirm no public adaptive compatibility path, unused solver component plumbing, fitted image-specific branches, or accidental CSV staging was introduced.

**Step 3: Push main**

```powershell
git push origin main
```

**Step 4: Report evidence**

Report per-sample clean fidelity, continuous-stripe gains, interruption safety, sample 01 visual judgment, candidate/iteration diagnostics, measured timing, final commit, and push result. Do not characterize measured runtime as an acceptance criterion.
