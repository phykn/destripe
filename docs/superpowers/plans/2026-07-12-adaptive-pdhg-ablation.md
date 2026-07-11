# Adaptive-PDHG Baseline and H3 Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the preferred adaptive-PDHG baseline, test H3 protection and soft repeatability independently, ship only the non-regressing winner, and delete the current H3 hybrid pipeline.

**Architecture:** The old adaptive estimator selects multiple directions and image-specific regularization, while the existing PDHG core produces a clean estimate plus direction components through a private result. A deterministic ablation runner evaluates A/B/C/D against the exact adaptive baseline; the production controller is then reduced to the winning component set and all losing H3 code is removed.

**Tech Stack:** Python 3, NumPy, PyTorch, OpenCV, pytest, Matplotlib, nbformat/nbclient, Ruff.

## Global Constraints

- Public API stays `destripe(image, process_size=256)` with no adaptive level or manual automatic parameters.
- Frozen baseline is adaptive level 2, analysis size 512, 1000 iterations, 2x2 tiles, and the old tile-local parameters; sample 01 selects directions `(0, 4)`.
- Quality is gated against the adaptive baseline, never merely against the input or current H3.
- Runtime is reported but is not a pass/fail criterion.
- Work directly on `main`; do not create branches, worktrees, or independent agents.
- Preserve the user's modified `notebooks/test.ipynb` until the final notebook task.
- Never stage `synthetic_benchmark_seed1234.csv` or `synthetic_benchmark_seed20260710.csv`.
- Use `.\.venv\Scripts\python.exe` for every Python command.

---

### Task 1: Restore and Freeze the Exact Adaptive Baseline

**Files:**
- Create: `src/destripe/adaptive/__init__.py`
- Create: `src/destripe/adaptive/constants.py`
- Create: `src/destripe/adaptive/directions.py`
- Create: `src/destripe/adaptive/estimate.py`
- Create: `src/destripe/adaptive/local.py`
- Create: `src/destripe/adaptive/preprocess.py`
- Create: `src/destripe/adaptive/refine.py`
- Create: `src/destripe/adaptive/strength.py`
- Create: `src/destripe/adaptive/stripe.py`
- Create: `benchmarks/adaptive_baseline.py`
- Create: `tests/test_adaptive_baseline.py`

**Interfaces:**
- Produces: `AdaptiveParams(directions: tuple[int, ...], mu1: float, mu2: float, confidence: float)`.
- Produces: `estimate_adaptive_params(gray: np.ndarray) -> AdaptiveParams`, internally fixed to former level 2.
- Produces: `estimate_tile_mus(gray: np.ndarray, *, tiles: int, directions: tuple[int, ...]) -> tuple[tuple[float, float], ...]`.
- Produces: `run_adaptive_baseline(image: np.ndarray, *, process_size: int = 512) -> BaselineResult` using 1000 iterations and 2x2 tiles.

- [ ] **Step 1: Write failing baseline recovery tests**

Add tests that load sample 01, normalize it, and assert the recovered estimator returns:

```python
params = estimate_adaptive_params(processed_gray)
assert params.directions == (0, 4)
assert params.mu1 == 0.25
assert params.mu2 == 0.01
assert params.confidence == pytest.approx(0.233, abs=0.002)
```

Assert the saved tile values reproduce the previous notebook within `5e-5`:

```python
expected_mu2 = np.array([0.01150372, 0.01233593, 0.01040453, 0.01045183])
actual = np.array(estimate_tile_mus(processed_gray, tiles=2, directions=(0, 4)))
np.testing.assert_allclose(actual[:, 0], 0.25, atol=0, rtol=0)
np.testing.assert_allclose(actual[:, 1], expected_mu2, atol=5e-5, rtol=0)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_adaptive_baseline.py -q
```

Expected: import failure because `destripe.adaptive` and the baseline runner do not exist.

- [ ] **Step 3: Restore adaptive code exactly before simplifying it**

Restore the nine adaptive files from `stash@{0}`. Change only the public entry signatures so level 2 is internal:

```python
BASELINE_LEVEL = 2

def estimate_adaptive_params(gray: np.ndarray) -> AdaptiveParams:
    return _estimate_for_level(gray=gray, level=BASELINE_LEVEL)
```

Keep direction scoring, SURE-like `mu2` selection, local tile estimates, stripe projection, and refinement numerically unchanged. Do not import anything from current `automatic.py` or `hybrid.py`.

- [ ] **Step 4: Implement a baseline-only runner**

`run_adaptive_baseline()` must normalize exactly like `ops.destripe`, prepare a 512-long-edge solver image, estimate adaptive parameters, run `UniversalStripeRemover.process_tiled()` with 1000 iterations and 2x2 tiles, apply `refine_clean()`, resize the correction, and restore range/dtype. Return:

```python
@dataclass(frozen=True)
class BaselineResult:
    clean: np.ndarray
    correction: np.ndarray
    directions: tuple[int, ...]
    mu1: float
    mu2: float
    tile_mus: tuple[tuple[float, float], ...]
    elapsed_seconds: float
```

- [ ] **Step 5: Freeze baseline evidence**

Run sample 01 twice and assert deterministic output. Record SHA-256 of the float32 normalized correction, correction RMS, and estimator diagnostics in `benchmarks/adaptive_baseline.py --report`; do not add a binary golden image.

- [ ] **Step 6: Run focused tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_adaptive_baseline.py tests/test_core.py -q
git add src/destripe/adaptive benchmarks/adaptive_baseline.py tests/test_adaptive_baseline.py
git commit -m "test: freeze previous adaptive quality baseline"
```

---

### Task 2: Return Direction Components from the Existing PDHG Core

**Files:**
- Modify: `src/destripe/core/remover.py`
- Modify: `tests/test_core.py`

**Interfaces:**
- Produces private `_SolveResult(clean: torch.Tensor, components: tuple[torch.Tensor, ...], iterations: int)`.
- Public `UniversalStripeRemover.process(...) -> torch.Tensor` remains unchanged.
- Produces private `_process_with_info(...) -> _SolveResult` for automatic use.

- [ ] **Step 1: Write failing component invariants**

For a two-direction solve, assert:

```python
info = remover._process_with_info(image, iterations=40, tol=1e-5, proj=True)
assert len(info.components) == 2
torch.testing.assert_close(info.clean + sum(info.components), image, atol=2e-5, rtol=2e-5)
```

Add tiled tests asserting components use the same padding, crop, and cosine blend as clean output and reconstruct the input after blending.

- [ ] **Step 2: Verify RED**

Run `tests/test_core.py`; expect `_SolveResult` to lack `components`.

- [ ] **Step 3: Preserve components in `_solve`**

Return CPU copies of `stripe_components` with `clean`. Update `_process_with_info()` to squeeze each component consistently. Keep `process()` returning only `.clean`.

- [ ] **Step 4: Blend tiled components**

Refactor the existing tiled canvas blend into a private helper used identically for clean and each direction component. Do not rerun the solver per direction and do not reconstruct components from `data - clean`.

- [ ] **Step 5: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_core.py tests/test_adaptive_baseline.py -q
git add src/destripe/core/remover.py tests/test_core.py
git commit -m "refactor: expose internal PDHG direction components"
```

---

### Task 3: Build a Fixed Baseline-Relative Comparison Set

**Files:**
- Create: `benchmarks/adaptive_ablation.py`
- Modify: `benchmarks/synthetic.py`
- Create: `tests/test_adaptive_ablation.py`

**Interfaces:**
- Produces `AblationCase(name, clean, observed, support, mode, strength)`.
- Produces `CandidateMetrics(psnr, ssim, unsupported_mse, correction_edge_energy, stripe_projection_left)`.
- Produces `evaluate_candidate(name: str, cases: tuple[AblationCase, ...], runner: Callable) -> CandidateReport`.

- [ ] **Step 1: Write failing comparison-set tests**

Assert the deterministic set contains clean sample 02-05, weak/medium/strong continuous curtains, all four partial-support masks, vertical mode 0 and diagonal modes 2/4, plus sample 01 as visual-only.

Assert every quantitative row stores both candidate and adaptive-baseline metrics. A row lacking baseline metrics must be rejected by the gate evaluator.

- [ ] **Step 2: Verify RED**

Run `tests/test_adaptive_ablation.py`; expect missing module failure.

- [ ] **Step 3: Implement baseline-relative metrics**

Reuse existing PSNR, SSIM, injection, and support-mask functions. Define correction edge energy as:

```python
gx = np.diff(correction, axis=1, append=correction[:, -1:])
gy = np.diff(correction, axis=0, append=correction[-1:, :])
edge_energy = float(np.mean((gx * image_gx + gy * image_gy) ** 2))
```

Store per-case values and worst cases; do not collapse all directions into one weighted score.

- [ ] **Step 4: Implement strict non-regression gates**

For every case require candidate PSNR/SSIM, unsupported MSE, correction edge energy, and stripe projection residual to be no worse than baseline within numerical tolerances (`1e-8` for normalized MSE and `1e-6` for SSIM). Candidate improvement on another row cannot offset a failure.

- [ ] **Step 5: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_adaptive_ablation.py tests/test_synthetic_benchmark.py -q
git add benchmarks/adaptive_ablation.py benchmarks/synthetic.py tests/test_adaptive_ablation.py
git commit -m "test: add adaptive non-regression comparisons"
```

---

### Task 4: Implement the Four A/B/C/D Ablation Candidates

**Files:**
- Create: `src/destripe/adaptive/evidence.py`
- Create: `src/destripe/adaptive/safety.py`
- Modify: `src/destripe/adaptive/stripe.py`
- Create: `tests/test_adaptive_evidence.py`
- Modify: `benchmarks/adaptive_ablation.py`

**Interfaces:**
- Produces `Evidence(protection: np.ndarray, repeatability: dict[int, float])`.
- Produces `make_protection(gray: np.ndarray, directions: tuple[int, ...]) -> np.ndarray`.
- Produces `measure_repeatability(gray: np.ndarray, directions: tuple[int, ...]) -> dict[int, float]` with values in `[0, 1]` and no threshold.
- Produces `apply_component_safety(components, evidence, *, use_protection, use_repeatability) -> np.ndarray`.

- [ ] **Step 1: Write failing evidence tests**

Test that protection is high at a synthetic particle edge and low on a smooth curtain, repeatability is near one for a continuous curtain and lower for a partial curtain, and neither function returns a hard boolean decision.

- [ ] **Step 2: Verify RED**

Run `tests/test_adaptive_evidence.py`; expect missing module failure.

- [ ] **Step 3: Consolidate robust directional primitives**

Move the single retained weighted Huber projection, line-ID construction, and directional differences into `adaptive/stripe.py`. Both evidence functions import them. Do not import `automatic.py`.

- [ ] **Step 4: Implement continuous component weights**

For direction `d`, define candidate C as:

```python
component = components[index]
if use_protection:
    component = component * (1.0 - evidence.protection)
alpha = evidence.repeatability[d] if use_repeatability else 1.0
correction += alpha * component
```

There is no cutoff, target projection, beta, global no-op, or cross-direction suppression.

- [ ] **Step 5: Register A/B/C/D runners**

The ablation module exposes exactly:

```python
CANDIDATES = {
    "A": CandidateConfig(False, False),
    "B": CandidateConfig(True, False),
    "C": CandidateConfig(False, True),
    "D": CandidateConfig(True, True),
}
```

All four reuse one adaptive estimator and one multi-direction PDHG solve per input.

- [ ] **Step 6: Verify and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_adaptive_evidence.py tests/test_adaptive_ablation.py -q
git add src/destripe/adaptive benchmarks/adaptive_ablation.py tests/test_adaptive_evidence.py tests/test_adaptive_ablation.py
git commit -m "feat: evaluate adaptive correction evidence"
```

---

### Task 5: Run Ablation, Freeze the Winner, and Integrate One Automatic Controller

**Files:**
- Modify: `src/destripe/automatic.py`
- Modify: `src/destripe/ops.py`
- Modify: `benchmarks/adaptive_ablation.py`
- Create: `benchmarks/adaptive_ablation_winner.json`
- Modify: `tests/test_core.py`
- Modify: `tests/test_adaptive_ablation.py`

**Interfaces:**
- Produces a thin `automatic_clean(gray: np.ndarray, *, proj: bool) -> AutomaticResult` whose diagnostics contain only directions, selected `mu1`/`mu2`, iterations, and elapsed time when consumed by notebook/benchmark.

- [ ] **Step 1: Run regression seeds without the held-back seed**

```powershell
.\.venv\Scripts\python.exe -m benchmarks.adaptive_ablation --asset-dir asset --seeds 1234 20260710 --process-size 512
```

Write per-case reports for A/B/C/D and deterministically select the first candidate in `A, B, C, D` order that passes all adaptive non-regression gates and has at least one strict improvement. If none improves without regression, select A.

- [ ] **Step 2: Freeze the winner manifest**

Write `adaptive_ablation_winner.json` with candidate name, flags, baseline hash, regression seeds, per-gate worst cases, and source commit. This file is evidence, not runtime configuration.

- [ ] **Step 3: Integrate the winner**

Replace current H3 orchestration with internal adaptive level-2 estimation, multi-direction PDHG components, and only the winner's evidence operations. `ops.destripe()` signature remains unchanged and continues restoring RGB, dtype, endpoints, and resize behavior.

- [ ] **Step 4: Verify public behavior**

Add tests proving no removed `adaptive`, `mu1`, `mu2`, directions, tile, or iteration argument is accepted by `destripe()`, while the automatic result uses multiple directions when the estimator selects them.

- [ ] **Step 5: Commit integration**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_core.py tests/test_adaptive_ablation.py -q
git add src/destripe/automatic.py src/destripe/ops.py benchmarks/adaptive_ablation.py benchmarks/adaptive_ablation_winner.json tests
git commit -m "refactor: select winning adaptive automatic pipeline"
```

---

### Task 6: Delete the H3 Hybrid and Every Losing Component

**Files:**
- Delete: `src/destripe/hybrid.py`
- Delete: `tests/test_hybrid.py`
- Modify or delete: `src/destripe/adaptive/evidence.py` according to winner
- Modify or delete: `src/destripe/adaptive/safety.py` according to winner
- Modify: `src/destripe/automatic.py`
- Modify: `tests/test_structure.py`
- Modify: `tests/test_automatic.py`

**Interfaces:**
- Final source contains no runtime winner flag, H3 target, beta, candidate grid, hard gate, or unused diagnostics.

- [ ] **Step 1: Add structural deletion tests**

Assert `hybrid.py` and `test_hybrid.py` do not exist and recursively scan `src/destripe` for forbidden names:

```python
for forbidden in ("H3Detection", "ParameterCandidate", "beta", "target_power", "_MIN_RELIABILITY"):
    assert forbidden not in source
```

Also assert only evidence functions enabled by the frozen winner remain importable.

- [ ] **Step 2: Verify RED**

Run `tests/test_structure.py`; expect current hybrid files and symbols to fail.

- [ ] **Step 3: Delete losers physically**

Delete `hybrid.py`, its tests, obsolete H3 tests/constants/diagnostics, and any protection or repeatability module not selected. Inline no alternate path and retain no compatibility shim.

- [ ] **Step 4: Run all source tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_structure.py tests/test_automatic.py tests/test_core.py -q
git add -A src tests
git commit -m "refactor: remove H3 hybrid and losing components"
```

---

### Task 7: Run Held-Back Validation, Rebuild the Notebook, and Deliver

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml`
- Modify and execute: `notebooks/test.ipynb`
- Modify: `tests/test_synthetic_benchmark.py`

**Interfaces:**
- Notebook compares only original, frozen adaptive baseline, and final winner with identical image/correction scales.

- [ ] **Step 1: Run the held-back seed once**

```powershell
.\.venv\Scripts\python.exe -m benchmarks.adaptive_ablation --asset-dir asset --held-back --process-size 512
```

Do not tune to this seed. If the winner fails baseline non-regression, report the failure and stop delivery.

- [ ] **Step 2: Update documentation**

Describe the final adaptive automatic flow and only the evidence component that won. Remove every “H3-guided hybrid”, beta, target-cap, hard-gate, and public adaptive-level reference.

- [ ] **Step 3: Rebuild and execute the notebook**

Use `nbformat` and `nbclient` with the repository-local `python3` kernel. Display original, baseline, winner, baseline correction, winner correction, selected directions/parameters, per-gate metrics, and elapsed time. Preserve the user's current notebook until this intentional replacement.

- [ ] **Step 4: Validate notebook claims**

Validate notebook structure, require sequential execution counts, reject error outputs, inspect the rendered comparison, and reconcile every headline metric against `adaptive_ablation_winner.json`.

- [ ] **Step 5: Run final verification**

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
git status --short
git branch --list
```

Expected: all checks pass; only `main` exists; only the two preserved CSV files are untracked.

- [ ] **Step 6: Commit and push**

```powershell
git add README.md pyproject.toml notebooks/test.ipynb tests/test_synthetic_benchmark.py benchmarks/adaptive_ablation_winner.json
git commit -m "docs: document final adaptive destriping method"
git push origin main
```
