# Simple Automatic Destriping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace adaptive levels and the component controller with the frozen, blind-validated H3 automatic algorithm, deleting unused compatibility code.

**Architecture:** `destripe()` normalizes and resizes an image, calls one parameter-free automatic directional-profile routine, and applies its correction. Manual PDHG remains only in `UniversalStripeRemover`; the old adaptive package and wrapper-level manual/level arguments are removed.

**Tech Stack:** Python 3.9+, NumPy, OpenCV, PyTorch, pytest, Ruff, repository-local `.venv`.

## Global Constraints

- Use the repository-local `.venv` for every Python command.
- Implement frozen H3 exactly: direct robust profile, distant blocked repeatability, all five directions, top-1 reliability, existing analytic alpha.
- Add no fitted constant, learned threshold, confidence blend, pattern-specific branch, top-K search, or PDHG invocation in the automatic path.
- Preserve `UniversalStripeRemover` manual PDHG behavior.
- Remove unused adaptive levels and compatibility aliases instead of deprecating them.
- Preserve shape, dtype, RGB shared correction, determinism, `process_size`, and `proj` behavior.
- Do not weaken or retune gates in response to seeds `1234`, `20260710`, or frozen blind seed `8675309`.
- Preserve unstaged Task 7 CSVs until final verification, and stage only files named by the current task.

---

### Task 1: Implement the Frozen H3 Automatic Module

**Files:**
- Create: `src/destripe/automatic.py`
- Create: `tests/test_automatic.py`

**Interfaces:**
- Produces: `AutomaticResult(clean: np.ndarray, direction: int, alpha: float)`.
- Produces: `automatic_clean(gray, *, proj) -> AutomaticResult`.
- Internal: `_blocked_repeatability`, `_project_robust`, `_make_protection`, `_choose_alpha`.

- [ ] **Step 1: Write failing behavioral tests**

Create `tests/test_automatic.py` with focused tests for:

```python
def test_automatic_removes_faint_vertical_stripe_and_preserves_structure() -> None:
    rows, cols = np.indices((96, 96))
    clean = 0.45 + 0.2 * np.exp(-((rows - 48) ** 2 + (cols - 48) ** 2) / 300)
    stripe = 0.01 * np.sin(np.linspace(0, 10 * np.pi, 96))[None, :]
    observed = np.clip(clean + stripe, 0.0, 1.0)

    result = automatic_clean(observed, proj=True)

    assert result.direction == 0
    assert 0.0 <= result.alpha <= 1.0
    assert np.mean((result.clean - clean) ** 2) < np.mean((observed - clean) ** 2)


def test_automatic_noops_clean_curved_structure() -> None:
    rows, cols = np.indices((96, 96))
    radius = np.sqrt((rows - 48) ** 2 + (cols - 48) ** 2)
    clean = 0.3 + 0.4 * np.exp(-((radius - 24) ** 2) / 4)

    result = automatic_clean(clean, proj=False)

    assert np.sqrt(np.mean((result.clean - clean) ** 2)) < 0.003


def test_blocked_repeatability_rejects_adjacent_smooth_structure() -> None:
    weights = torch.ones((64, 16), dtype=torch.float32)
    localized = torch.zeros_like(weights)
    localized[8:24, 5] = 0.02
    repeated = torch.zeros_like(weights)
    repeated[:, 5] = 0.02

    assert _blocked_repeatability(localized, mode=0, weights=weights) == 0.0
    assert _blocked_repeatability(repeated, mode=0, weights=weights) > 0.9
```

Also cover all-zero weights, tiny arrays, deterministic ties choosing the lower
mode, finite output, and no spatial masking at a protected crossing.

- [ ] **Step 2: Verify RED**

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
.\.venv\Scripts\python.exe -m pytest tests/test_automatic.py -v
```

Expected: import failure because `destripe.automatic` does not exist.

- [ ] **Step 3: Port only the frozen H3 primitives**

Use the diagnostic implementation recorded in
`.superpowers/sdd/diagnose_block_repeatability.py` as the behavioral reference.
Move only the code needed for H3 into `automatic.py`:

```python
ALL_DIRECTIONS = (0, 1, 2, 3, 4)
PARALLEL_OFFSETS = {
    0: (1, 0),
    1: (2, 1),
    2: (1, 1),
    3: (2, -1),
    4: (1, -1),
}


@dataclass(frozen=True)
class AutomaticResult:
    clean: np.ndarray
    direction: int
    alpha: float
```

For every mode, calculate protection, robust direct high-pass profile, sigma-1
scale repeatability, and the H3 distant blocked repeatability. Select exactly:

```python
selected = max(ALL_DIRECTIONS, key=lambda mode: (reliability[mode], -mode))
```

Calculate alpha with the frozen quadratic and return:

```python
clean = gray_array - alpha * selected_profile
if proj:
    clean = np.clip(clean, 0.0, 1.0)
return AutomaticResult(clean=clean, direction=selected, alpha=alpha)
```

Do not import the old adaptive package and do not add an option surface.

- [ ] **Step 4: Add diagnostic equivalence tests**

For fixed seed `1234`, sample 02, canonical vertical strengths `0.01`, `0.03`,
and `0.06`, compare `AutomaticResult` to the frozen H3 diagnostic CSV/profile
within `1e-6` for direction and alpha and `1e-5` image RMSE. Copy the required
expected scalar values into the test; do not make tests depend on ignored files.

- [ ] **Step 5: Run focused and full tests**

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
.\.venv\Scripts\python.exe -m pytest tests/test_automatic.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: focused PASS; existing suite remains green before deletion work.

- [ ] **Step 6: Commit**

```powershell
git add src/destripe/automatic.py tests/test_automatic.py
git commit -m "feat: add simple automatic destriping"
```

---

### Task 2: Simplify the Wrapper API and Delete Adaptive Levels

**Files:**
- Modify: `src/destripe/ops.py`
- Modify: `src/destripe/__init__.py`
- Delete: `src/destripe/adaptive/`
- Modify: `tests/test_core.py`
- Modify: `tests/test_structure.py`

**Interfaces:**
- Produces: `destripe(image, *, process_size=None, proj=True) -> np.ndarray`.
- Preserves: `UniversalStripeRemover` public manual API.

- [ ] **Step 1: Write the new wrapper tests first**

Add tests asserting:

```python
def test_destripe_has_only_automatic_options() -> None:
    signature = inspect.signature(destripe)
    assert tuple(signature.parameters) == ("image", "process_size", "proj")


@pytest.mark.parametrize("old_name", [
    "adaptive", "mu1", "mu2", "iterations", "tol", "tiles", "overlap",
    "device", "verbose", "directions",
])
def test_removed_wrapper_arguments_raise_type_error(old_name: str) -> None:
    with pytest.raises(TypeError):
        destripe(np.ones((8, 8)), **{old_name: 1})
```

Update grayscale, RGB, uint8/float, constant, process-size, projection, invalid
shape, non-finite, and determinism tests to call the new wrapper. Keep all manual
`UniversalStripeRemover` tests unchanged.

- [ ] **Step 2: Verify RED against the old signature**

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
.\.venv\Scripts\python.exe -m pytest tests/test_core.py -k "automatic_options or removed_wrapper" -v
```

Expected: signature assertions fail.

- [ ] **Step 3: Simplify `ops.py`**

Implement only validation, normalization, luma conversion, process-size resize,
one `automatic_clean()` call, correction resize-back, RGB shared subtraction,
clipping, and dtype restoration. The signature is exact:

```python
def destripe(
    image: np.ndarray,
    *,
    process_size: int | None = None,
    proj: bool = True,
) -> np.ndarray:
```

There is no warning/ignore path and no reference to `UniversalStripeRemover` in
`ops.py`.

- [ ] **Step 4: Delete unused adaptive code and tests**

Delete the complete `src/destripe/adaptive/` directory. Remove tests for adaptive
levels, estimator confidence/mu selection, tile-mus, component-return plumbing,
adaptive refinement, and ignored manual arguments. Keep core solver tests and
automatic wrapper tests.

Update `tests/test_structure.py` to assert:

- `src/destripe/automatic.py` exists;
- `src/destripe/adaptive/` does not exist;
- `ops.py` imports `automatic_clean` and does not contain `adaptive_level`,
  `estimate_adaptive_params`, `estimate_tile_mus`, or `refine_clean`.

- [ ] **Step 5: Run core/manual/wrapper and full tests**

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
.\.venv\Scripts\python.exe -m pytest tests/test_core.py tests/test_structure.py tests/test_automatic.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass; test count may decrease because removed contracts are
deleted rather than preserved.

- [ ] **Step 6: Commit**

```powershell
git add src/destripe/ops.py src/destripe/__init__.py src/destripe/adaptive tests/test_core.py tests/test_structure.py
git commit -m "refactor: remove adaptive level compatibility"
```

---

### Task 3: Simplify Benchmarks, Documentation, and Verify Generalization

**Files:**
- Modify: `benchmarks/synthetic.py`
- Modify: `benchmarks/acceptance.py`
- Modify: `benchmarks/performance.py`
- Modify: `tests/test_synthetic_benchmark.py`
- Modify: `README.md`
- Modify: `pyproject.toml`

**Interfaces:**
- Removes: benchmark `level` field and `--levels` CLI option.
- Updates: acceptance to the vertical battery-SEM primary scope.

- [ ] **Step 1: Write failing no-level benchmark tests**

Assert `RESULT_FIELDS` has no `level`, rows contain no level, CLI rejects
`--levels`, and uniqueness/completeness identities work without level. Keep exact
19-pattern matrix and lossless metadata tests.

- [ ] **Step 2: Update benchmark calls and acceptance**

Call:

```python
output = destripe(observed, process_size=process_size)
```

Emit one clean/real/synthetic row per case. Primary strict weak gates filter exact
canonical additive mode-0 curtains. Keep medium/strong canonical gates across all
five modes. Keep weak-oblique and robustness metrics in CSV and printed summaries
without fitted pass thresholds.

- [ ] **Step 3: Update performance and README**

Remove iteration/device/adaptive arguments from `benchmarks/performance.py`.
Document automatic H3 usage as simply `clean = destripe(image,
process_size=256)`. Document manual PDHG through `UniversalStripeRemover` and the
weak-oblique limitation. Remove every adaptive-level and tile-mu description.

Update project description to mention automatic robust directional profiles with
an optional manual PDHG core.

- [ ] **Step 4: Resolve Ruff notebook scope explicitly**

Add a narrow Ruff per-file ignore for `E402` in `notebooks/*.ipynb`; do not
reorder notebook cells or globally disable the rule.

- [ ] **Step 5: Run unit/static verification**

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: both commands exit `0`.

- [ ] **Step 6: Run frozen-seed verification without retuning**

Run benchmark seeds `1234`, `20260710`, and `8675309` with the unchanged source.
All three must pass clean, vertical weak, and canonical medium/strong primary
gates. Compare seed `8675309` output to the frozen H3 diagnostic metrics; material
differences are implementation defects, not reasons to tune constants.

Record all weak-oblique and robustness metrics even when their diagnostic limits
remain incomplete.

- [ ] **Step 7: Re-measure performance and inspect sample 01**

Use the same four preloaded clean samples and three repeats. Median automatic
runtime must be no slower than `1.322108 s`. Recreate sample01 output/comparison
outside the repository and confirm no seam/halo or broad contrast shift.

- [ ] **Step 8: Commit**

```powershell
git add benchmarks README.md pyproject.toml tests/test_synthetic_benchmark.py
git commit -m "docs: finalize simple automatic destriping"
```
