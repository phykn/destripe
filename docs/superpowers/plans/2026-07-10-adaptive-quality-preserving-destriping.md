# Adaptive Quality-Preserving Destriping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully automatic adaptive controller that removes weak coherent stripes while preserving SEM structure, with one PDHG solve and no more than 15% median runtime overhead.

**Architecture:** Retain the PDHG solver's direction-specific stripe components, estimate stripe evidence with protected robust line projections, add any residual line component, and choose one analytic coefficient per direction. Reliability is used once in coefficient selection; structural risk is measured by protected parallel curvature rather than correction energy at SEM edges.

**Tech Stack:** Python 3.9+, NumPy, OpenCV, PyTorch, pytest, repository-local `.venv`.

## Global Constraints

- Use `D:\code\destripe\.venv\Scripts\python.exe` for every Python command.
- Preserve the public `destripe(..., adaptive=0..3)` contract, manual-mode behavior, output dtype, and output shape.
- Run PDHG exactly once per existing solver invocation; do not search `mu1`, `mu2`, direction, or image candidates by rerunning it.
- Use no new runtime dependency.
- Apply reliability exactly once, in the numerator of per-direction coefficient selection.
- Never spatially mask the accepted stripe correction at SEM edges.
- Treat `asset/sample_01.jpeg` as real-only and `sample_02.png` through `sample_05.png` as clean benchmark targets.
- Preserve the user's existing asset and notebook changes; stage only files named by the current task.

---

### Task 1: Capture Baselines and Retain Solver Components

**Files:**
- Create: `benchmarks/performance.py`
- Create: `src/destripe/core/result.py`
- Modify: `src/destripe/core/remover.py:15-448`
- Test: `tests/test_core.py:113-243`

**Interfaces:**
- Produces: `StripeResult(clean: torch.Tensor, components: tuple[torch.Tensor, ...])`.
- Produces: `UniversalStripeRemover.process_tiled_components(...) -> StripeResult`.
- Preserves: `process(...) -> torch.Tensor` and `process_tiled(...) -> torch.Tensor`.

- [ ] **Step 1: Add a repeatable timer and record the baseline before changing source**

Create `benchmarks/performance.py` with a CLI that preloads `sample_02` through
`sample_05`, accepts `--repeats`, `--iterations`, and `--process-size`, executes
one unmeasured warm-up, and prints the median of all measured `time.perf_counter()`
durations. The timed region must contain only:

```python
import argparse
import statistics
import time
from pathlib import Path

from destripe import destripe
from benchmarks.synthetic import load_samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, default=Path("asset"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--process-size", type=int, default=256)
    args = parser.parse_args()
    images = [sample.image for sample in load_samples(args.asset_dir) if sample.has_ground_truth]
    for image in images:
        destripe(
            image,
            adaptive=2,
            iterations=args.iterations,
            process_size=args.process_size,
            device="cpu",
        )
    durations = []
    for _ in range(args.repeats):
        for image in images:
            started = time.perf_counter()
            destripe(
                image,
                adaptive=2,
                iterations=args.iterations,
                process_size=args.process_size,
                device="cpu",
            )
            durations.append(time.perf_counter() - started)
    print(f"median_seconds={statistics.median(durations):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.performance --asset-dir asset --repeats 3 --iterations 500 --process-size 256
```

Expected: one finite positive `median_seconds` value. Keep it in the task notes;
do not encode a machine-specific runtime in source.

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: `185 passed`.

Then time one warm-up and three measured adaptive CPU calls for each clean sample
with `adaptive=2`, `iterations=500`, and `process_size=256`. Keep the median in
the task notes for comparison in Task 7; image loading must occur before timing.

- [ ] **Step 2: Write failing component-reconstruction tests**

Add to `tests/test_core.py`:

```python
@pytest.mark.parametrize("tiles", [1, 2])
def test_process_tiled_components_reconstructs_input(tiles: int) -> None:
    remover = UniversalStripeRemover(device="cpu", directions=[0, 2])
    image = torch.rand((32, 36), generator=torch.Generator().manual_seed(92))

    result = remover.process_tiled_components(
        image=image,
        tiles=tiles,
        overlap=4,
        iterations=8,
        proj=False,
    )

    assert result.clean.shape == image.shape
    assert len(result.components) == 2
    assert all(component.shape == image.shape for component in result.components)
    reconstructed = result.clean + torch.stack(result.components).sum(dim=0)
    assert torch.allclose(reconstructed, image, atol=2e-5, rtol=2e-5)


def test_process_tiled_components_handles_tiny_image() -> None:
    remover = UniversalStripeRemover(device="cpu", directions=[0, 3])
    image = torch.rand((1, 8), generator=torch.Generator().manual_seed(93))

    result = remover.process_tiled_components(image=image, tiles=3, iterations=1)

    assert torch.equal(result.clean, image)
    assert len(result.components) == 2
    assert all(torch.count_nonzero(component) == 0 for component in result.components)
```

- [ ] **Step 3: Run the focused tests and confirm the API is missing**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_core.py -k "process_tiled_components" -v
```

Expected: FAIL with `AttributeError: 'UniversalStripeRemover' object has no attribute 'process_tiled_components'`.

- [ ] **Step 4: Add the result type and component-returning solve path**

Create `src/destripe/core/result.py`:

```python
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StripeResult:
    clean: torch.Tensor
    components: tuple[torch.Tensor, ...]
```

In `remover.py`, import `StripeResult`. Refactor the current tiled body into
`_process_tiled(..., keep_components: bool) -> StripeResult`. Keep the existing
validation and solver loop unchanged. Public wrappers must be exactly:

```python
def process_tiled(self, image, tiles=1, iterations=500, tol=1e-5, overlap=64,
                  proj=True, verbose=False, tile_mus=None) -> torch.Tensor:
    return self._process_tiled(
        image=image,
        tiles=tiles,
        iterations=iterations,
        tol=tol,
        overlap=overlap,
        proj=proj,
        verbose=verbose,
        tile_mus=tile_mus,
        keep_components=False,
    ).clean


def process_tiled_components(
    self, image, tiles=1, iterations=500, tol=1e-5, overlap=64,
    proj=True, verbose=False, tile_mus=None,
) -> StripeResult:
    return self._process_tiled(
        image=image,
        tiles=tiles,
        iterations=iterations,
        tol=tol,
        overlap=overlap,
        proj=proj,
        verbose=verbose,
        tile_mus=tile_mus,
        keep_components=True,
    )
```

Make `_solve(..., keep_components: bool = False) -> StripeResult` return:

```python
return StripeResult(
    clean=clean.cpu(),
    components=(
        tuple(component.cpu() for component in stripe_components)
        if keep_components
        else ()
    ),
)
```

Update `process()` to return `_solve(..., keep_components=False).clean`, and
update every existing internal `_solve()` call to read `.clean` when components
are not requested.

When tiling, blend each returned component with the same indices, cosine window,
and blend sum used for the clean batch. Extract the shared accumulation into
`_blend_tiles(tile_batch, indices, tiles, core_h, core_w, padded_h, padded_w,
overlap_pixels)`. For a tiny image, return the unchanged clean image and one zero
array per configured direction when `keep_components=True`.

- [ ] **Step 5: Run solver tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_core.py -k "Process or components" -q
```

Expected: all selected tests PASS.

- [ ] **Step 6: Commit the component path**

```powershell
git add benchmarks/performance.py src/destripe/core/result.py src/destripe/core/remover.py tests/test_core.py
git commit -m "feat: retain adaptive stripe components"
```

---

### Task 2: Add Protected Robust Line Projection

**Files:**
- Modify: `src/destripe/adaptive/stripe.py:1-76`
- Test: `tests/test_core.py:664-733`

**Interfaces:**
- Produces: `project_robust(tensor, mode, weights) -> torch.Tensor`.
- Extends: `measure_shrinkage(tensor, mode, weights=None) -> float`.
- Preserves: `project(tensor, mode)` numerical behavior.

- [ ] **Step 1: Write failing robust-projection tests**

Add:

```python
def test_robust_projection_ignores_protected_local_structure() -> None:
    from destripe.adaptive.stripe import project_robust

    tensor = torch.zeros((32, 12), dtype=torch.float32)
    tensor[:, 5] = 0.02
    tensor[10:14, 5] = 1.0
    weights = torch.ones_like(tensor)
    weights[9:15, 5] = 0.0

    projected = project_robust(tensor, mode=0, weights=weights)

    assert torch.allclose(projected[:, 5], torch.full((32,), 0.02), atol=1e-4)


def test_weighted_shrinkage_ignores_protected_mismatch() -> None:
    from destripe.adaptive.stripe import measure_shrinkage

    tensor = torch.zeros((32, 8), dtype=torch.float32)
    tensor[:, 3] = 0.02
    tensor[[0, 2], 3] += 0.5
    weights = torch.ones_like(tensor)
    weights[[0, 2], 3] = 0.0

    assert measure_shrinkage(tensor, 0, weights=weights) > 0.9
```

- [ ] **Step 2: Verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_core.py -k "robust_projection or weighted_shrinkage" -v
```

Expected: FAIL because `project_robust` and the `weights` argument do not exist.

- [ ] **Step 3: Implement vectorized weighted Huber projection**

Add a private scatter helper that computes `sum(weights*values)/sum(weights)`
for each line id and expands it to image shape. Implement:

```python
def project_robust(
    tensor: torch.Tensor,
    mode: int,
    weights: torch.Tensor,
) -> torch.Tensor:
    safe = weights.to(dtype=tensor.dtype, device=tensor.device).clamp(0.0, 1.0)
    first = _project_weighted(tensor=tensor, mode=mode, weights=safe)
    residual = (tensor - first).abs()
    scale = _project_weighted(tensor=residual, mode=mode, weights=safe)
    cutoff = 1.345 * scale
    huber = torch.where(
        residual <= EPS,
        torch.ones_like(residual),
        torch.minimum(
            torch.ones_like(residual),
            cutoff / residual.clamp(min=EPS),
        ),
    )
    return _project_weighted(tensor=tensor, mode=mode, weights=safe * huber)
```

Change `project()` to call `_project_weighted()` with all-one weights. Add optional
weights to both split-half scatter reductions in `measure_shrinkage`; a split with
zero total weight is unusable. Do not change the default unweighted result.

- [ ] **Step 4: Run projection and legacy adaptive tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_core.py -k "AdaptiveRefine or robust_projection or weighted_shrinkage" -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/destripe/adaptive/stripe.py tests/test_core.py
git commit -m "feat: add protected robust stripe projection"
```

---

### Task 3: Build Direction Evidence and Structure Protection

**Files:**
- Create: `src/destripe/adaptive/safety.py`
- Modify: `tests/test_structure.py:146-172`
- Test: `tests/test_core.py:664-733`

**Interfaces:**
- Produces: `DirectionEvidence(protection, reliability, input_profile)`.
- Produces: `make_direction_evidence(gray, directions) -> dict[int, DirectionEvidence]`.

- [ ] **Step 1: Write failing evidence tests**

Add tests that construct a faint vertical curtain and a curved SEM-like ring:

```python
def test_direction_evidence_keeps_faint_coherent_stripe() -> None:
    from destripe.adaptive.safety import make_direction_evidence

    rng = np.random.default_rng(201)
    image = 0.5 + rng.normal(0.0, 0.002, (96, 96))
    image += 0.01 * np.sin(np.linspace(0, 12 * np.pi, 96))[None, :]

    evidence = make_direction_evidence(image, directions=(0,))[0]

    assert evidence.reliability >= 0.7
    assert float(evidence.protection.mean()) < 0.75


def test_direction_evidence_protects_curved_structure() -> None:
    from destripe.adaptive.safety import make_direction_evidence

    rows, cols = np.indices((96, 96))
    radius = np.sqrt((rows - 48) ** 2 + (cols - 48) ** 2)
    image = np.exp(-((radius - 24) ** 2) / 3.0)

    evidence = make_direction_evidence(image, directions=(0,))[0]
    ring = np.abs(radius - 24) < 2

    assert float(evidence.protection.numpy()[ring].mean()) > 0.6
```

- [ ] **Step 2: Verify failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_core.py -k "direction_evidence" -v
```

Expected: FAIL because `destripe.adaptive.safety` does not exist.

- [ ] **Step 3: Implement evidence construction**

Create `safety.py` with:

```python
from dataclasses import dataclass
import math

import cv2
import numpy as np
import torch

from .constants import EPS, NORMAL_MAD_SCALE, PARALLEL_OFFSETS
from .preprocess import extract_high_pass
from .stripe import measure_shrinkage, project_robust


@dataclass(frozen=True)
class DirectionEvidence:
    protection: torch.Tensor
    reliability: float
    input_profile: torch.Tensor
```

Implement `_parallel_activity()` with valid source/target slices and zero-valued
outside borders; do not use wrapping rolls. Implement protection exactly as the
spec: median, `MAD/NORMAL_MAD_SCALE` with standard-deviation fallback,
`clip((activity-median)/(3*scale+EPS), 0, 1)`, `cv2.dilate(..., 3x3)`, and
`cv2.GaussianBlur(..., 5x5, sigmaX=0)`.

In `make_direction_evidence()`, calculate normal and sigma-1 blurred high-pass
images. For each direction, use `safe=1-protection`, robustly project both
high-pass images, calculate positive centered cosine correlation as scale
repeatability, calculate weighted split-half shrinkage, and set:

```python
reliability = math.sqrt(split_repeatability * scale_repeatability)
```

Return finite values clipped to `[0, 1]`.

- [ ] **Step 4: Add structural ownership checks**

In `tests/test_structure.py`, assert `safety.py` exists, owns
`DirectionEvidence` and `make_direction_evidence`, and does not contain a Python
loop over image rows or columns.

- [ ] **Step 5: Run evidence and structure tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_core.py tests/test_structure.py -k "evidence or safety or structure" -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/destripe/adaptive/safety.py tests/test_core.py tests/test_structure.py
git commit -m "feat: measure protected stripe evidence"
```

---

### Task 4: Select Quality-Preserving Per-Direction Corrections

**Files:**
- Modify: `src/destripe/adaptive/safety.py`
- Modify: `src/destripe/adaptive/refine.py:1-39`
- Test: `tests/test_core.py:664-733`

**Interfaces:**
- Produces: `SafetyResult(clean: np.ndarray, alphas: tuple[float, ...])`.
- Produces: `choose_alpha(input_profile, proposal_profile, reliability, leakage) -> float`.
- Produces: `select_clean(gray, solver_clean, components, directions, proj) -> SafetyResult`.
- Changes: `refine_clean(..., components) -> np.ndarray` delegates to `select_clean`.

- [ ] **Step 1: Write failing coefficient tests**

Add:

```python
def test_safe_selection_accepts_stripe_crossing_structure() -> None:
    from destripe.adaptive.safety import select_clean

    rows, cols = np.indices((96, 96))
    structure = 0.2 * np.exp(-((rows - 48) ** 2 + (cols - 48) ** 2) / 250.0)
    stripe = 0.01 * np.sin(np.linspace(0, 10 * np.pi, 96))[None, :]
    gray = 0.5 + structure + stripe

    result = select_clean(
        gray=gray,
        solver_clean=gray - stripe,
        components=(np.broadcast_to(stripe, gray.shape).copy(),),
        directions=(0,),
        proj=False,
    )

    assert result.alphas[0] > 0.7
    assert np.mean((result.clean - (gray - stripe)) ** 2) < np.mean(stripe**2) * 0.2


def test_choose_alpha_uses_reliability_once() -> None:
    from destripe.adaptive import safety

    profile = torch.tensor(
        [[-0.02, 0.0, 0.02], [-0.02, 0.0, 0.02]],
        dtype=torch.float32,
    )

    alpha = safety.choose_alpha(
        input_profile=profile,
        proposal_profile=profile,
        reliability=0.5,
        leakage=0.0,
    )

    assert alpha == pytest.approx(0.5, abs=1e-6)


def test_choose_alpha_noops_uncorrelated_profile() -> None:
    from destripe.adaptive.safety import choose_alpha

    alpha = choose_alpha(
        input_profile=torch.tensor([[-1.0, 0.0, 1.0]]),
        proposal_profile=torch.tensor([[1.0, 0.0, 1.0]]),
        reliability=1.0,
        leakage=0.0,
    )

    assert alpha == 0.0
```

- [ ] **Step 2: Verify failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_core.py -k "safe_selection or choose_alpha" -v
```

Expected: FAIL because `select_clean` does not exist.

- [ ] **Step 3: Implement residual proposals and the analytic coefficient**

Add:

```python
@dataclass(frozen=True)
class SafetyResult:
    clean: np.ndarray
    alphas: tuple[float, ...]
```

Implement `choose_alpha()` as the single owner of the documented quadratic:

```python
def choose_alpha(
    *,
    input_profile: torch.Tensor,
    proposal_profile: torch.Tensor,
    reliability: float,
    leakage: float,
) -> float:
    normalizer = float(torch.sum(input_profile.square()).item()) + EPS
    a_value = float(torch.sum(proposal_profile.square()).item()) / normalizer
    a_value += max(0.0, float(leakage))
    b_value = float(reliability) * float(
        torch.sum(input_profile * proposal_profile).item()
    ) / normalizer
    return float(np.clip(b_value / (a_value + EPS), 0.0, 1.0))
```

Implement same-shape first and second parallel differences with valid slices.
For each `(mode, component)`, form:

```python
residual = project_robust(clean_high_pass, mode, safe)
proposal = np.asarray(component, dtype=np.float64) + residual.numpy()
proposal_high_pass = extract_high_pass(torch.as_tensor(proposal, dtype=torch.float32))
proposal_profile = project_robust(proposal_high_pass, mode, safe)
curvature = _second_parallel_diff(torch.as_tensor(proposal, dtype=torch.float32), mode)
proposal_power = float(torch.sum(proposal_profile.square()).item())
leakage = float(torch.sum((evidence.protection * curvature).square()).item())
alpha = choose_alpha(
    input_profile=evidence.input_profile,
    proposal_profile=proposal_profile,
    reliability=evidence.reliability,
    leakage=leakage / (proposal_power + EPS),
)
```

Accumulate `alpha*proposal` without multiplying by the protection map. Return
`gray - accepted_correction`, clipped only when `proj=True`. Shape mismatch,
component-count mismatch, or non-finite components must raise `ValueError` with
messages naming `components`.

Update `refine_clean()` to accept `components: tuple[np.ndarray, ...]` and return
`select_clean(...).clean`. Remove its unconditional projection loop.

- [ ] **Step 4: Run controller tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_core.py -k "AdaptiveRefine or safe_selection or choose_alpha" -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/destripe/adaptive/safety.py src/destripe/adaptive/refine.py tests/test_core.py
git commit -m "feat: select safe adaptive corrections"
```

---

### Task 5: Wire the Controller into Adaptive Destriping

**Files:**
- Modify: `src/destripe/ops.py:147-219`
- Test: `tests/test_core.py:773-1251`

**Interfaces:**
- Consumes: `process_tiled_components()` and direction-aligned components.
- Produces: unchanged public `destripe()` output contract.

- [ ] **Step 1: Write a failing one-solver integration test**

Add or replace the current adaptive refinement spy with:

```python
def test_adaptive_uses_one_component_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    from destripe import ops
    from destripe.adaptive import AdaptiveParams
    from destripe.core.result import StripeResult

    calls = {"components": 0, "plain": 0}

    monkeypatch.setattr(
        ops,
        "estimate_adaptive_params",
        lambda *_args, **_kwargs: AdaptiveParams(
            directions=(0,), mu1=1 / 4, mu2=1 / 300, confidence=1.0
        ),
    )

    class FakeRemover:
        def __init__(self, **_: object) -> None:
            pass

        def process_tiled(self, image: np.ndarray, **_: object) -> torch.Tensor:
            calls["plain"] += 1
            return torch.as_tensor(image)

        def process_tiled_components(
            self, image: np.ndarray, **_: object
        ) -> StripeResult:
            calls["components"] += 1
            tensor = torch.as_tensor(image)
            return StripeResult(clean=tensor, components=(torch.zeros_like(tensor),))

    monkeypatch.setattr(ops, "UniversalStripeRemover", FakeRemover)

    image = np.random.default_rng(203).random((24, 24))
    result = destripe(image, adaptive=2, iterations=1)

    assert result.shape == image.shape
    assert calls == {"components": 1, "plain": 0}
```

- [ ] **Step 2: Verify the test fails on the plain solve path**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_core.py -k "one_component_solve" -v
```

Expected: FAIL with `calls == {"components": 0, "plain": 1}`.

- [ ] **Step 3: Route only adaptive mode through components**

In `_destripe_grayscale()`, branch at the solve call:

```python
solver_kwargs = {
    "image": solver_input,
    "tiles": tiles,
    "iterations": iterations,
    "tol": tol,
    "overlap": overlap,
    "proj": proj,
    "verbose": verbose,
    "tile_mus": tile_mus,
}
if adaptive_level is None:
    solver_clean = remover.process_tiled(**solver_kwargs).numpy()
else:
    solver_result = remover.process_tiled_components(**solver_kwargs)
    solver_clean = solver_result.clean.numpy()
    solver_clean = refine_clean(
        gray=processed_gray,
        clean=solver_clean,
        components=tuple(component.numpy() for component in solver_result.components),
        directions=resolved_directions,
        proj=proj,
    )
```

Pass the existing identical tiling, iteration, tolerance, overlap, projection,
verbosity, and `tile_mus` arguments to both methods. Update existing fake removers
and refinement spies to implement the new adaptive method/signature. Do not alter
manual-mode calls or resize-back behavior.

- [ ] **Step 4: Run API, RGB, tiled, and process-size tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_core.py -k "adaptive or process_size or rgb or tiled" -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/destripe/ops.py tests/test_core.py
git commit -m "feat: apply safe controller in adaptive mode"
```

---

### Task 6: Strengthen Synthetic and Clean Acceptance Gates

**Files:**
- Modify: `benchmarks/synthetic.py`
- Create: `benchmarks/acceptance.py`
- Modify: `tests/test_synthetic_benchmark.py`

**Interfaces:**
- Extends: result rows with `seed`, `carrier`, `profile_scale`, and `angle_offset`.
- Extends: `PatternSpec` with `carrier`, `profile_scale`, and `angle_offset`.
- Produces: clean rows with `pattern="none"`, `strength=0.0`.
- Produces: `evaluate_acceptance(rows) -> list[str]`; an empty list means pass.

- [ ] **Step 1: Write failing generation and acceptance tests**

Cover these exact behaviors:

```python
def test_multiplicative_injection_reports_actual_stripe() -> None:
    clean = np.full((12, 14), 0.5)
    pattern = np.tile(np.linspace(-1, 1, 14), (12, 1))
    observed, actual = inject_stripe(
        clean, pattern, strength=0.02, carrier="multiplicative"
    )
    assert np.allclose(actual, observed - clean)
    assert not np.allclose(actual, 0.02 * pattern)


def test_acceptance_rejects_noop_weak_rows() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=100.0, clean_psnr=100.0)
    failures = evaluate_acceptance(rows)
    assert any("projection" in failure for failure in failures)


def test_acceptance_rejects_clean_fidelity_below_absolute_gate() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=39.9)
    failures = evaluate_acceptance(rows)
    assert any("clean PSNR" in failure for failure in failures)
```

Define the fixture in the same test module:

```python
def make_acceptance_fixture(
    *,
    weak_projection_left: float,
    clean_psnr: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    samples = [f"sample_{index:02d}.png" for index in range(2, 6)]
    for sample in samples:
        rows.append(
            {
                "seed": 1234,
                "sample": sample,
                "case_type": "clean",
                "pattern": "none",
                "mode": None,
                "strength": 0.0,
                "carrier": "additive",
                "profile_scale": 9,
                "angle_offset": 0.0,
                "input_psnr": float("inf"),
                "output_psnr": clean_psnr,
                "input_ssim": 1.0,
                "output_ssim": 1.0,
                "stripe_projection_left_pct": 0.0,
            }
        )
        for mode in range(5):
            for strength, psnr_gain, ssim_gain in (
                (0.01, 0.2, 0.002),
                (0.03, 1.0, 0.003),
                (0.06, 4.0, 0.004),
            ):
                rows.append(
                    {
                        "seed": 1234,
                        "sample": sample,
                        "case_type": "synthetic",
                        "pattern": f"curtain_m{mode}",
                        "mode": mode,
                        "strength": strength,
                        "carrier": "additive",
                        "profile_scale": 9,
                        "angle_offset": 0.0,
                        "input_psnr": 40.0,
                        "output_psnr": 40.0 + psnr_gain,
                        "input_ssim": 0.95,
                        "output_ssim": 0.95 + ssim_gain,
                        "stripe_projection_left_pct": (
                            weak_projection_left if strength == 0.01 else 50.0
                        ),
                    }
                )
    return rows
```

- [ ] **Step 2: Verify failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_synthetic_benchmark.py -k "multiplicative or acceptance" -v
```

Expected: FAIL because the carrier argument and acceptance module do not exist.

- [ ] **Step 3: Extend pattern generation without changing existing defaults**

Add `carrier: str = "additive"`, `profile_scale: int = 9`, and
`angle_offset: float = 0.0` to `PatternSpec`. Existing defaults must reproduce
current patterns for seed `1234`. Add narrow (`profile_scale=3`), broad
(`profile_scale=15`), multiplicative, and off-grid (`angle_offset=7.5` degrees)
specs. For off-grid coordinates, rotate the normalized line normal derived from
`PARALLEL_OFFSETS[mode]`, then interpolate a smoothed one-dimensional random
profile at the continuous coordinate with `np.interp`.

Extend `inject_stripe()`:

```python
if carrier == "additive":
    proposed = clean_array + strength * pattern_array
elif carrier == "multiplicative":
    proposed = clean_array * (1.0 + strength * pattern_array)
else:
    raise ValueError("carrier must be additive or multiplicative.")
observed = np.clip(proposed, 0.0, 1.0)
return observed, observed - clean_array
```

Add one unmodified clean row per clean sample and adaptive level. Calculate its
output-to-input PSNR/SSIM with the sample as reference. Keep `sample_01` real-only.
Add `seed`, `carrier`, `profile_scale`, and `angle_offset` to `RESULT_FIELDS` and
every emitted row so CSVs retain the evidence needed for grouping.

- [ ] **Step 4: Implement deterministic acceptance evaluation**

In `benchmarks/acceptance.py`, group rows by seed/case/pattern/strength/mode and
emit explicit failure strings for every spec gate:

- weak mean PSNR gain `<0.10 dB` or SSIM gain `<0.001`;
- fewer than 75% weak cases with PSNR gain `>=0.05 dB` and SSIM gain `>=0.0001`;
- any weak loss worse than `-1.0 dB`;
- any supported direction with negative weak mean PSNR gain;
- weak additive mean projection left `>70%` or fewer than 75% cases `<=85%`;
- medium or strong mean PSNR gain more than `0.25 dB` below `0.919` or `3.783`;
- any clean sample below `40 dB` PSNR or `0.99` SSIM;
- any missing sample, strength, supported mode, or non-finite metric.

Apply the recorded medium/strong baselines only to canonical additive patterns
with `angle_offset=0` and `profile_scale=9`. For narrow, broad, multiplicative,
and off-grid robustness rows, require non-negative mean PSNR gain and forbid any
case from losing more than `1.0 dB`; do not compare them to the canonical
baseline.

Add `--check-acceptance` to the benchmark CLI. After writing CSV, print every
failure and return exit code `1`; return `0` only for an empty failure list.

- [ ] **Step 5: Run benchmark unit tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_synthetic_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add benchmarks/synthetic.py benchmarks/acceptance.py tests/test_synthetic_benchmark.py
git commit -m "test: enforce adaptive quality gates"
```

---

### Task 7: Document, Profile, and Verify the Complete Change

**Files:**
- Modify: `README.md:17-49`
- Modify: `tests/test_structure.py`

**Interfaces:**
- Documents: automatic protection, repeatability, per-direction coefficients,
  one-solver behavior, and the unavoidable single-image ambiguity.

- [ ] **Step 1: Update README adaptive-mode behavior**

Replace the unconditional residual-step description with:

```markdown
Adaptive mode keeps the solver's direction-specific stripe components, estimates
line profiles from pixels that are consistent with the stripe direction, and
checks split-half and multi-scale repeatability. It then chooses a continuous
coefficient from 0 to 1 for each direction. The original image is the automatic
fallback when a proposed component is unsupported or contains protected
parallel curvature.

The PDHG solver still runs once. Protection changes how evidence and structural
leakage are measured; it does not punch holes in a stripe correction at SEM
edges. No ground truth or per-image threshold is required at runtime. A real
full-length structure that is mathematically identical to a stripe remains
ambiguous from a single image.
```

- [ ] **Step 2: Run formatting and the full unit suite**

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: Ruff exits `0`; all tests pass with more than the `185`-test baseline.

- [ ] **Step 3: Run development and held-out acceptance benchmarks**

```powershell
.\.venv\Scripts\python.exe -m benchmarks.synthetic --asset-dir asset --output synthetic_benchmark_seed1234.csv --seed 1234 --levels 2 --iterations 500 --process-size 256 --check-acceptance
.\.venv\Scripts\python.exe -m benchmarks.synthetic --asset-dir asset --output synthetic_benchmark_seed20260710.csv --seed 20260710 --levels 2 --iterations 500 --process-size 256 --check-acceptance
```

Expected: both commands exit `0` with no acceptance failures. Do not commit the
generated CSV files.

- [ ] **Step 4: Re-run the performance measurement**

Use the same preloaded images, warm-up, three measured runs, CPU device,
`adaptive=2`, `iterations=500`, and `process_size=256` as Task 1. Calculate:

```text
overhead_pct = 100 * (new_median / baseline_median - 1)
```

Expected: `overhead_pct <= 15`. If it fails, profile before changing algorithms;
cache high-pass tensors and line ids first, because those are reused per selected
direction. Quality gates must remain unchanged.

- [ ] **Step 5: Inspect real-only sample output**

Run `sample_01.jpeg` once at `adaptive=2`, save the result outside the repository,
and inspect input, output, and amplified difference side-by-side. Confirm there
are no seams at particle edges and no broad contrast shift. This is diagnostic,
not a substitute for the synthetic gates.

- [ ] **Step 6: Review the final diff and commit documentation**

```powershell
git diff --check
git status --short
git add README.md tests/test_structure.py
git commit -m "docs: explain safe adaptive destriping"
```

Expected: only intentional source, test, benchmark, documentation, and existing
user asset/notebook changes are listed; generated CSV and real-output images are
not staged.
