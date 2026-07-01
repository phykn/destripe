# Adaptive Destripe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in adaptive destriping that selects sparse directions and continuous `mu` values while preserving the existing default behavior.

**Architecture:** Extend the existing PDHG solver to accept an active direction subset, then add a high-level adaptive estimator that computes global directions and `mu` from normalized grayscale evidence. Tiled adaptive processing keeps one global direction set and estimates tile-local `mu` values smoothed in log-space before reusing the existing cosine blending pipeline.

**Tech Stack:** Python 3.9+, NumPy, PyTorch, pytest, existing `src/destripe` package.

---

## File Structure

- Modify `src/destripe/core.py`
  - Add direction validation and active-direction storage.
  - Change solver loops from fixed five directions to `self.directions`.
  - Add optional tile-local `mu` support to `process_tiled`.
- Create `src/destripe/adaptive.py`
  - Own adaptive evidence estimation, continuous `mu` mapping, confidence gate, and tile-grid smoothing.
  - Keep public return value internal: `AdaptiveParams(directions, mu1, mu2, confidence)`.
- Modify `src/destripe/ops.py`
  - Add `adaptive` and `directions` arguments.
  - Use `None` sentinel for detecting explicit `mu1` and `mu2`.
  - Emit warnings when adaptive ignores manual values.
  - Route grayscale/RGB/tiled calls through the adaptive estimator.
- Modify `tests/test_core.py`
  - Add tests for directions, warnings, adaptive estimator, adaptive output, and tiled adaptive behavior.
- Modify `README.md`
  - Document `adaptive`, manual `directions`, warning behavior, and updated recommended usage.

## Task 1: Solver Direction Subsets

**Files:**
- Modify: `src/destripe/core.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Add failing tests for manual direction validation and subset execution**

Add this test class after `TestAdjointConsistency` in `tests/test_core.py`:

```python
class TestDirections:
    def test_default_directions_are_all_modes(self) -> None:
        remover = UniversalStripeRemover(device="cpu")
        assert remover.directions == (0, 1, 2, 3, 4)

    def test_subset_directions_preserve_shape(self) -> None:
        remover = UniversalStripeRemover(device="cpu", directions=[0])
        img = torch.rand(24, 24)
        result = remover.process(image=img, iterations=5)
        assert remover.directions == (0,)
        assert result.shape == img.shape

    @pytest.mark.parametrize(
        "directions",
        [
            [],
            [0, 0],
            [-1],
            [5],
            [1.5],
            ["0"],
        ],
    )
    def test_invalid_directions(self, directions: object) -> None:
        with pytest.raises(ValueError, match="directions"):
            UniversalStripeRemover(device="cpu", directions=directions)
```

- [ ] **Step 2: Run the focused failing tests**

Run:

```bash
python -m pytest tests/test_core.py::TestDirections -q
```

Expected: FAIL because `UniversalStripeRemover.__init__` does not accept `directions`.

- [ ] **Step 3: Implement direction validation in `core.py`**

In `src/destripe/core.py`, import `Sequence` and update constants:

```python
from collections.abc import Sequence
```

Keep `_NUM_DIRS = 5`, remove `_NUM_VARS`, and add:

```python
_ALL_DIRECTIONS = tuple(range(_NUM_DIRS))
```

Update the constructor:

```python
def __init__(
    self,
    mu1: float = 0.33,
    mu2: float = 0.003,
    device: torch.device | str | None = None,
    directions: Sequence[int] | None = None,
) -> None:
    self.device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    self.mu1 = mu1
    self.mu2 = mu2
    self.directions = self._validate_directions(directions)
    self.tau = 0.35
    self.sigma = 0.35
```

Add this validation method under the existing validation section:

```python
@staticmethod
def _validate_directions(directions: Sequence[int] | None) -> tuple[int, ...]:
    if directions is None:
        return _ALL_DIRECTIONS

    try:
        values = tuple(directions)
    except TypeError as exc:
        raise ValueError("directions must be None or a sequence of integers.") from exc

    if not values:
        raise ValueError("directions must not be empty.")
    if len(set(values)) != len(values):
        raise ValueError("directions must not contain duplicates.")
    for mode in values:
        if not isinstance(mode, int) or isinstance(mode, bool):
            raise ValueError("directions must contain integers in the range 0..4.")
        if mode < 0 or mode >= _NUM_DIRS:
            raise ValueError("directions must contain integers in the range 0..4.")
    return values
```

Update docstrings for `UniversalStripeRemover` and `process` to mention `directions`.

- [ ] **Step 4: Replace fixed solver loops with active directions**

In `_solve`, replace fixed component allocation and loops with active direction indexing:

```python
num_stripes = len(self.directions)
num_vars = 1 + num_stripes

clean = data.clone()
stripe_components = [torch.zeros_like(input=data) for _ in self.directions]
dir_dual = [torch.zeros_like(input=data) for _ in self.directions]
dir_dual_bar = [torch.zeros_like(input=data) for _ in self.directions]
l2_dual = [torch.zeros_like(input=data) for _ in self.directions]
l2_dual_bar = [torch.zeros_like(input=data) for _ in self.directions]
```

Change all `for mode in range(_NUM_DIRS):` loops to:

```python
for component_idx, mode in enumerate(self.directions):
```

and index arrays with `component_idx`:

```python
self._adjoint_dir(
    target=stripe_components[component_idx],
    q=dir_dual_bar[component_idx],
    mode=mode,
    a=step_size,
)
stripe_components[component_idx].sub_(
    l2_dual_bar[component_idx], alpha=step_size
)
```

Use the same pattern in the dual update loop:

```python
dir_dual_bar[component_idx].copy_(dir_dual[component_idx])
self._dir_diff(
    x=stripe_components[component_idx],
    mode=mode,
    out=directional_diff,
)
dir_dual[component_idx].add_(directional_diff).clamp_(
    min=-dir_dual_clip,
    max=dir_dual_clip,
)
dir_dual_bar[component_idx].mul_(-1).add_(dir_dual[component_idx], alpha=2)

l2_dual_bar[component_idx].copy_(l2_dual[component_idx])
l2_dual[component_idx].add_(stripe_components[component_idx]).clamp_(
    min=-l2_dual_clip,
    max=l2_dual_clip,
)
l2_dual_bar[component_idx].mul_(-1).add_(l2_dual[component_idx], alpha=2)
```

Change the constraint denominator:

```python
scratch.sub_(clean).div_(num_vars)
```

Change projection residual redistribution:

```python
scratch.div_(num_stripes)
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
python -m pytest tests/test_core.py::TestDirections tests/test_core.py::TestAdjointConsistency -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/destripe/core.py tests/test_core.py
git commit -m "feat: support manual stripe directions"
```

## Task 2: Public Wrapper API And Warnings

**Files:**
- Modify: `src/destripe/ops.py`
- Modify: `tests/test_core.py`

- [ ] **Step 1: Add failing tests for wrapper arguments**

Add these tests inside `TestDestripe`:

```python
def test_manual_directions(self, gray_image: np.ndarray) -> None:
    result = destripe(gray_image, directions=[0], iterations=10)
    assert result.shape == gray_image.shape
    assert result.dtype == gray_image.dtype

def test_invalid_manual_directions(self) -> None:
    with pytest.raises(ValueError, match="directions"):
        destripe(np.random.default_rng(12).random((16, 16)), directions=[5])

def test_adaptive_warns_when_manual_values_are_ignored(self) -> None:
    img = np.random.default_rng(13).random((24, 24))
    with pytest.warns(UserWarning, match="adaptive=True ignores"):
        result = destripe(
            img,
            adaptive=True,
            directions=[0],
            mu1=0.5,
            mu2=0.017,
            iterations=5,
        )
    assert result.shape == img.shape
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_core.py::TestDestripe::test_manual_directions tests/test_core.py::TestDestripe::test_invalid_manual_directions tests/test_core.py::TestDestripe::test_adaptive_warns_when_manual_values_are_ignored -q
```

Expected: FAIL because `destripe()` does not accept `directions` or `adaptive`.

- [ ] **Step 3: Update `destripe()` signature and warning path**

In `src/destripe/ops.py`, add imports:

```python
from collections.abc import Sequence
import warnings
```

Change the function signature:

```python
def destripe(
    image: np.ndarray,
    mu1: float | None = None,
    mu2: float | None = None,
    iterations: int = 500,
    tol: float = 1e-5,
    tiles: int = 1,
    overlap: int = 64,
    proj: bool = True,
    device: torch.device | str | None = None,
    verbose: bool = False,
    adaptive: bool = False,
    directions: Sequence[int] | None = None,
) -> np.ndarray:
```

Add defaults and warning detection after finite validation:

```python
manual_mu1 = mu1 is not None
manual_mu2 = mu2 is not None
manual_directions = directions is not None

if adaptive and (manual_mu1 or manual_mu2 or manual_directions):
    warnings.warn(
        "adaptive=True ignores manual directions, mu1, and mu2 values.",
        UserWarning,
        stacklevel=2,
    )

effective_mu1 = 0.33 if mu1 is None else mu1
effective_mu2 = 0.003 if mu2 is None else mu2
effective_directions = None if adaptive else directions
```

For now, before adaptive estimation exists, construct the remover with manual defaults even under adaptive:

```python
remover = UniversalStripeRemover(
    mu1=effective_mu1,
    mu2=effective_mu2,
    device=device,
    directions=effective_directions,
)
```

Do not change `_run_grayscale` in this task; it should continue receiving a configured remover and calling `process_tiled`.

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m pytest tests/test_core.py::TestDestripe::test_manual_directions tests/test_core.py::TestDestripe::test_invalid_manual_directions tests/test_core.py::TestDestripe::test_adaptive_warns_when_manual_values_are_ignored -q
```

Expected: PASS.

- [ ] **Step 5: Run existing wrapper tests**

Run:

```bash
python -m pytest tests/test_core.py::TestDestripe -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/destripe/ops.py tests/test_core.py
git commit -m "feat: add adaptive wrapper switches"
```

## Task 3: Adaptive Estimator Module

**Files:**
- Create: `src/destripe/adaptive.py`
- Modify: `tests/test_core.py`

- [ ] **Step 1: Add failing tests for estimator behavior**

At the top of `tests/test_core.py`, add:

```python
from destripe.adaptive import estimate_adaptive_params
```

Add this class before `TestDestripe`:

```python
class TestAdaptiveEstimator:
    def test_vertical_stripes_select_mode_zero(self) -> None:
        img = np.zeros((64, 64), dtype=np.float64)
        img[:, 12] = 1.0
        img[:, 32] = 0.8
        params = estimate_adaptive_params(img)
        assert params.directions[0] == 0
        assert 0.10 <= params.mu1 <= 0.50
        assert 0.0017 <= params.mu2 <= 0.017

    def test_estimator_is_deterministic(self) -> None:
        rng = np.random.default_rng(14)
        img = rng.random((48, 48))
        p1 = estimate_adaptive_params(img)
        p2 = estimate_adaptive_params(img)
        assert p1 == p2

    def test_tile_mu_smoothing_preserves_shape(self) -> None:
        from destripe.adaptive import smooth_tile_mus

        mus = np.array(
            [
                [[0.10, 0.0017], [0.50, 0.017]],
                [[0.33, 0.0030], [0.40, 0.007]],
            ],
            dtype=np.float64,
        )
        smoothed = smooth_tile_mus(mus)
        assert smoothed.shape == mus.shape
        assert np.all(smoothed[..., 0] >= 0.10)
        assert np.all(smoothed[..., 0] <= 0.50)
        assert np.all(smoothed[..., 1] >= 0.0017)
        assert np.all(smoothed[..., 1] <= 0.017)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_core.py::TestAdaptiveEstimator -q
```

Expected: FAIL because `destripe.adaptive` does not exist.

- [ ] **Step 3: Create `adaptive.py`**

Create `src/destripe/adaptive.py`:

```python
"""Adaptive direction and parameter estimation for destriping."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F

_ALL_DIRECTIONS = (0, 1, 2, 3, 4)
_PARALLEL_OFFSETS = {
    0: (1, 0),
    1: (2, 1),
    2: (1, 1),
    3: (2, -1),
    4: (1, -1),
}
_CROSS_OFFSETS = {
    0: (0, 1),
    1: (1, -2),
    2: (1, -1),
    3: (1, 2),
    4: (1, 1),
}

_MU1_MIN = 0.10
_MU1_ANCHOR = 0.33
_MU1_MAX = 0.50
_MU2_MIN = 0.0017
_MU2_ANCHOR = 0.003
_MU2_MAX = 0.017
_EPS = 1e-9


@dataclass(frozen=True)
class AdaptiveParams:
    directions: tuple[int, ...]
    mu1: float
    mu2: float
    confidence: float


def estimate_adaptive_params(
    gray: np.ndarray,
    *,
    fixed_directions: tuple[int, ...] | None = None,
) -> AdaptiveParams:
    analysis = _analysis_tensor(gray)
    high_pass = _high_pass(analysis)
    contrast = _robust_contrast(high_pass)

    scores = {
        mode: _direction_score(high_pass, mode=mode, contrast=contrast)
        for mode in _ALL_DIRECTIONS
    }
    directions = fixed_directions or _select_directions(scores)
    selected_scores = [scores[mode] for mode in directions]
    top_score = max(selected_scores)
    second_score = sorted(scores.values(), reverse=True)[1]
    ambiguity = _ambiguity_score(scores=scores, selected=directions)

    mu1 = _estimate_mu1(top_score)
    mu2 = _estimate_mu2(ambiguity)
    confidence = _confidence(top_score=top_score, second_score=second_score, ambiguity=ambiguity)
    return AdaptiveParams(
        directions=tuple(directions),
        mu1=mu1,
        mu2=mu2,
        confidence=confidence,
    )


def smooth_tile_mus(mus: np.ndarray) -> np.ndarray:
    if mus.ndim != 3 or mus.shape[-1] != 2:
        raise ValueError("mus must have shape (rows, cols, 2).")

    clipped = np.empty_like(mus, dtype=np.float64)
    clipped[..., 0] = np.clip(mus[..., 0], _MU1_MIN, _MU1_MAX)
    clipped[..., 1] = np.clip(mus[..., 1], _MU2_MIN, _MU2_MAX)

    log_mus = np.log(clipped)
    padded = np.pad(log_mus, ((1, 1), (1, 1), (0, 0)), mode="edge")
    out = np.zeros_like(log_mus)
    for row_offset in range(3):
        for col_offset in range(3):
            out += padded[
                row_offset : row_offset + log_mus.shape[0],
                col_offset : col_offset + log_mus.shape[1],
                :,
            ]
    out /= 9.0
    smoothed = np.exp(out)
    smoothed[..., 0] = np.clip(smoothed[..., 0], _MU1_MIN, _MU1_MAX)
    smoothed[..., 1] = np.clip(smoothed[..., 1], _MU2_MIN, _MU2_MAX)
    return smoothed


def _analysis_tensor(gray: np.ndarray) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(gray), dtype=torch.float32)
    if t.dim() != 2:
        raise ValueError("gray must have shape (H, W).")
    h, w = t.shape
    max_side = max(h, w)
    if max_side <= 512:
        return t
    scale = 512 / max_side
    size = (max(8, round(h * scale)), max(8, round(w * scale)))
    return F.interpolate(
        t.unsqueeze(0).unsqueeze(0),
        size=size,
        mode="area",
    ).squeeze(0).squeeze(0)


def _high_pass(t: torch.Tensor) -> torch.Tensor:
    h, w = t.shape
    kernel = int(round(min(h, w) * 0.015))
    kernel = max(7, min(31, kernel | 1))
    pad = kernel // 2
    padded = F.pad(
        t.unsqueeze(0).unsqueeze(0),
        pad=(pad, pad, pad, pad),
        mode="reflect",
    )
    blur = F.avg_pool2d(padded, kernel_size=kernel, stride=1)
    return t - blur.squeeze(0).squeeze(0)


def _offset_diff(t: torch.Tensor, row_step: int, col_step: int) -> torch.Tensor:
    row_start_a = max(0, -row_step)
    row_start_b = max(0, row_step)
    col_start_a = max(0, -col_step)
    col_start_b = max(0, col_step)
    rows = t.shape[0] - abs(row_step)
    cols = t.shape[1] - abs(col_step)
    if rows <= 0 or cols <= 0:
        return torch.zeros(1, dtype=t.dtype, device=t.device)
    a = t[row_start_a : row_start_a + rows, col_start_a : col_start_a + cols]
    b = t[row_start_b : row_start_b + rows, col_start_b : col_start_b + cols]
    return b - a


def _robust_contrast(t: torch.Tensor) -> float:
    return float(torch.quantile(t.abs().reshape(-1), 0.90).item()) + _EPS


def _direction_score(t: torch.Tensor, *, mode: int, contrast: float) -> float:
    parallel = _offset_diff(t, *_PARALLEL_OFFSETS[mode]).abs().reshape(-1)
    cross = _offset_diff(t, *_CROSS_OFFSETS[mode]).abs().reshape(-1)
    parallel_q = float(torch.quantile(parallel, 0.75).item()) + _EPS
    cross_q = float(torch.quantile(cross, 0.90).item()) + _EPS
    power_q = float(torch.quantile(t.abs().reshape(-1), 0.90).item()) + _EPS
    return (cross_q / parallel_q) * (power_q / contrast)


def _select_directions(scores: dict[int, float]) -> tuple[int, ...]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_mode, top_score = ranked[0]
    directions = [top_mode]
    if ranked[1][1] >= 0.85 * top_score and ranked[1][1] >= 1.15:
        directions.append(ranked[1][0])
    return tuple(directions)


def _estimate_mu1(score: float) -> float:
    strength = min(1.0, max(0.0, (score - 1.0) / 3.0))
    if strength <= 0.5:
        return _log_interp(_MU1_MIN, _MU1_ANCHOR, strength / 0.5)
    return _log_interp(_MU1_ANCHOR, _MU1_MAX, (strength - 0.5) / 0.5)


def _estimate_mu2(ambiguity: float) -> float:
    if ambiguity <= 0.5:
        return _log_interp(_MU2_MIN, _MU2_ANCHOR, ambiguity / 0.5)
    return _log_interp(_MU2_ANCHOR, _MU2_MAX, (ambiguity - 0.5) / 0.5)


def _ambiguity_score(
    *,
    scores: dict[int, float],
    selected: tuple[int, ...],
) -> float:
    ranked = sorted(scores.values(), reverse=True)
    top = ranked[0] + _EPS
    second = ranked[1] if len(ranked) > 1 else 0.0
    direction_confusion = second / top
    multi_direction_penalty = 0.25 if len(selected) > 1 else 0.0
    return min(1.0, max(0.0, direction_confusion + multi_direction_penalty))


def _confidence(*, top_score: float, second_score: float, ambiguity: float) -> float:
    dominance = 1.0 - min(1.0, second_score / (top_score + _EPS))
    strength = min(1.0, max(0.0, (top_score - 1.0) / 3.0))
    return min(1.0, max(0.0, 0.5 * dominance + 0.5 * strength - 0.25 * ambiguity))


def _log_interp(lo: float, hi: float, t: float) -> float:
    t = min(1.0, max(0.0, t))
    return float(math.exp(math.log(lo) * (1.0 - t) + math.log(hi) * t))
```

- [ ] **Step 4: Run estimator tests**

Run:

```bash
python -m pytest tests/test_core.py::TestAdaptiveEstimator -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/destripe/adaptive.py tests/test_core.py
git commit -m "feat: estimate adaptive stripe parameters"
```

## Task 4: Integrate Adaptive Mode For Non-Tiled Images

**Files:**
- Modify: `src/destripe/ops.py`
- Modify: `tests/test_core.py`

- [ ] **Step 1: Add failing adaptive output tests**

Add these tests inside `TestDestripe`:

```python
def test_adaptive_uses_estimated_parameters(self, monkeypatch: pytest.MonkeyPatch) -> None:
    from destripe import ops
    from destripe.adaptive import AdaptiveParams

    calls = {}

    def fake_estimate(gray: np.ndarray, *, fixed_directions=None) -> AdaptiveParams:
        calls["shape"] = gray.shape
        calls["fixed_directions"] = fixed_directions
        return AdaptiveParams(directions=(0,), mu1=0.10, mu2=0.0017, confidence=1.0)

    monkeypatch.setattr(ops, "estimate_adaptive_params", fake_estimate, raising=False)
    img = np.random.default_rng(17).random((24, 24))
    result = destripe(img, adaptive=True, iterations=5)

    assert result.shape == img.shape
    assert calls == {"shape": img.shape, "fixed_directions": None}

def test_adaptive_grayscale_float64(self, gray_image: np.ndarray) -> None:
    result = destripe(gray_image, adaptive=True, iterations=10)
    assert result.shape == gray_image.shape
    assert result.dtype == gray_image.dtype

def test_adaptive_rgb(self) -> None:
    img = np.random.default_rng(15).random((32, 32, 3)).astype(np.float32)
    result = destripe(img, adaptive=True, iterations=10)
    assert result.shape == img.shape
    assert result.dtype == img.dtype

def test_adaptive_constant_returns_copy(self) -> None:
    img = np.full((16, 16), 12, dtype=np.uint8)
    result = destripe(img, adaptive=True)
    assert np.array_equal(result, img)
    assert result is not img
```

- [ ] **Step 2: Run estimator integration test to verify failure**

Run:

```bash
python -m pytest tests/test_core.py::TestDestripe::test_adaptive_uses_estimated_parameters -q
```

Expected: FAIL because `ops.py` does not call `estimate_adaptive_params`.

- [ ] **Step 3: Route adaptive through the estimator**

In `src/destripe/ops.py`, add:

```python
from .adaptive import estimate_adaptive_params
```

Refactor remover creation so it happens after grayscale selection. Add a helper:

```python
def _make_remover(
    *,
    gray: np.ndarray,
    adaptive: bool,
    mu1: float,
    mu2: float,
    directions: Sequence[int] | None,
    device: torch.device | str | None,
) -> UniversalStripeRemover:
    if adaptive:
        params = estimate_adaptive_params(gray)
        return UniversalStripeRemover(
            mu1=params.mu1,
            mu2=params.mu2,
            device=device,
            directions=params.directions,
        )
    return UniversalStripeRemover(
        mu1=mu1,
        mu2=mu2,
        device=device,
        directions=directions,
    )
```

In the grayscale branch:

```python
remover = _make_remover(
    gray=normalized,
    adaptive=adaptive,
    mu1=effective_mu1,
    mu2=effective_mu2,
    directions=effective_directions,
    device=device,
)
clean = _run_grayscale(...)
```

In the RGB/single-channel branch, compute `gray` first, then create the remover from that grayscale image:

```python
remover = _make_remover(
    gray=gray,
    adaptive=adaptive,
    mu1=effective_mu1,
    mu2=effective_mu2,
    directions=effective_directions,
    device=device,
)
clean_gray = _run_grayscale(...)
```

Remove the earlier unconditional `remover = UniversalStripeRemover(...)`.

- [ ] **Step 4: Run adaptive and wrapper tests**

Run:

```bash
python -m pytest tests/test_core.py::TestAdaptiveEstimator tests/test_core.py::TestDestripe::test_adaptive_uses_estimated_parameters tests/test_core.py::TestDestripe::test_adaptive_grayscale_float64 tests/test_core.py::TestDestripe::test_adaptive_rgb tests/test_core.py::TestDestripe::test_adaptive_constant_returns_copy -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/destripe/ops.py tests/test_core.py
git commit -m "feat: integrate adaptive destriping"
```

## Task 5: Tile-Local Adaptive Mu

**Files:**
- Modify: `src/destripe/core.py`
- Modify: `src/destripe/ops.py`
- Modify: `tests/test_core.py`

- [ ] **Step 1: Add failing tiled adaptive test**

Add this test inside `TestDestripe`:

```python
def test_adaptive_tiled_passes_tile_mus(
    self,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from destripe import ops
    from destripe.adaptive import AdaptiveParams

    seen = {}

    def fake_estimate(gray: np.ndarray, *, fixed_directions=None) -> AdaptiveParams:
        return AdaptiveParams(directions=(0,), mu1=0.33, mu2=0.003, confidence=1.0)

    original = UniversalStripeRemover.process_tiled

    def spy_process_tiled(self, *args, **kwargs):
        seen["tile_mus"] = kwargs.get("tile_mus")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ops, "estimate_adaptive_params", fake_estimate)
    monkeypatch.setattr(UniversalStripeRemover, "process_tiled", spy_process_tiled)

    img = np.random.default_rng(18).random((32, 32))
    result = destripe(img, adaptive=True, iterations=5, tiles=2, overlap=4)

    assert result.shape == img.shape
    assert seen["tile_mus"] is not None
    assert len(seen["tile_mus"]) == 4

def test_adaptive_tiled_dtype_shape(self) -> None:
    img = np.random.default_rng(16).random((48, 40, 3)).astype(np.float32)
    result = destripe(img, adaptive=True, iterations=5, tiles=3, overlap=6)
    assert result.shape == img.shape
    assert result.dtype == img.dtype
```

- [ ] **Step 2: Run the test**

Run:

```bash
python -m pytest tests/test_core.py::TestDestripe::test_adaptive_tiled_passes_tile_mus -q
```

Expected: FAIL because `ops.py` does not compute or pass `tile_mus`.

- [ ] **Step 3: Add optional tile-local `mu` support in `core.py`**

Update `process_tiled` signature:

```python
def process_tiled(
    self,
    image: torch.Tensor | np.ndarray,
    tiles: int = 1,
    iterations: int = 500,
    tol: float = 1e-5,
    overlap: int = 64,
    proj: bool = True,
    verbose: bool = False,
    tile_mus: Sequence[tuple[float, float]] | None = None,
) -> torch.Tensor:
```

After `tile_tensor = torch.stack(...)`, replace the single `self.process(...)` call with:

```python
if tile_mus is None:
    cleaned_tiles = self.process(
        image=tile_tensor,
        iterations=iterations,
        tol=tol,
        proj=proj,
        verbose=verbose,
    )
else:
    if len(tile_mus) != len(tiles_batch):
        raise ValueError("tile_mus length must match the number of tiles.")
    original_mu1, original_mu2 = self.mu1, self.mu2
    cleaned_list = []
    try:
        for tile, (tile_mu1, tile_mu2) in zip(tile_tensor, tile_mus):
            self.mu1 = float(tile_mu1)
            self.mu2 = float(tile_mu2)
            cleaned_list.append(
                self.process(
                    image=tile,
                    iterations=iterations,
                    tol=tol,
                    proj=proj,
                    verbose=verbose,
                )
            )
    finally:
        self.mu1, self.mu2 = original_mu1, original_mu2
    cleaned_tiles = torch.stack(tensors=cleaned_list)
```

This preserves the existing batch path when `tile_mus` is absent.

- [ ] **Step 4: Compute tile-local mus in `ops.py`**

In `src/destripe/ops.py`, import:

```python
from .adaptive import estimate_adaptive_params, smooth_tile_mus
```

Add helper:

```python
def _estimate_tile_mus(
    gray: np.ndarray,
    *,
    tiles: int,
    directions: tuple[int, ...],
) -> list[tuple[float, float]]:
    if tiles <= 1:
        return []
    h, w = gray.shape
    pad_h = (tiles - h % tiles) % tiles
    pad_w = (tiles - w % tiles) % tiles
    padded = np.pad(gray, ((0, pad_h), (0, pad_w)), mode="reflect")
    core_h = padded.shape[0] // tiles
    core_w = padded.shape[1] // tiles
    mus = np.empty((tiles, tiles, 2), dtype=np.float64)
    for row in range(tiles):
        for col in range(tiles):
            tile = padded[
                row * core_h : (row + 1) * core_h,
                col * core_w : (col + 1) * core_w,
            ]
            params = estimate_adaptive_params(tile, fixed_directions=directions)
            mus[row, col, 0] = params.mu1
            mus[row, col, 1] = params.mu2
    smoothed = smooth_tile_mus(mus)
    return [
        (float(smoothed[row, col, 0]), float(smoothed[row, col, 1]))
        for row in range(tiles)
        for col in range(tiles)
    ]
```

Update `_run_grayscale` signature:

```python
def _run_grayscale(
    remover: UniversalStripeRemover,
    gray: np.ndarray,
    iterations: int,
    tol: float,
    tiles: int,
    overlap: int,
    proj: bool,
    verbose: bool,
    tile_mus: list[tuple[float, float]] | None = None,
) -> np.ndarray:
```

Pass `tile_mus` into `process_tiled`:

```python
out = remover.process_tiled(
    image=gray,
    tiles=tiles,
    iterations=iterations,
    tol=tol,
    overlap=overlap,
    proj=proj,
    verbose=verbose,
    tile_mus=tile_mus,
)
```

In adaptive branches, after creating the global adaptive remover:

```python
tile_mus = None
if adaptive and tiles > 1:
    tile_mus = _estimate_tile_mus(
        gray=gray,
        tiles=tiles,
        directions=remover.directions,
    )
```

Pass `tile_mus=tile_mus` to `_run_grayscale`.

- [ ] **Step 5: Run tiled and process tests**

Run:

```bash
python -m pytest tests/test_core.py::TestProcessTiled tests/test_core.py::TestDestripe::test_adaptive_tiled_passes_tile_mus tests/test_core.py::TestDestripe::test_adaptive_tiled_dtype_shape -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/destripe/core.py src/destripe/ops.py tests/test_core.py
git commit -m "feat: adapt mu per tile"
```

## Task 6: README And Full Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README quick start**

Change the quick start call to recommend adaptive mode:

```python
clean = destripe(
    image,
    adaptive=True,  # automatically chooses directions and mu strengths
    iterations=500,
    tiles=1,        # >1 for n x n tiled processing
    device="cpu",   # "cpu", "cuda", or None to auto-select
)
```

- [ ] **Step 2: Update README feature and parameter sections**

Add feature bullets:

```markdown
- Optional adaptive mode selects sparse stripe directions and `mu` strengths from the image.
- Manual mode can restrict solver directions with `directions=[...]`; `directions=None` keeps all five modes.
```

Add parameter bullets:

```markdown
- `adaptive` (default `False`): when `True`, estimate directions and `mu` automatically. Explicit `directions`, `mu1`, and `mu2` are ignored with a warning.
- `directions` (default `None`): manual solver modes to use when `adaptive=False`. `None` uses all five modes; pass a non-empty list containing integers `0..4` to restrict the solver.
```

Update `mu` bullets:

```markdown
- `mu1` (default `0.33`): manual TV weight used when `adaptive=False`.
- `mu2` (default `0.003`): manual stripe penalty used when `adaptive=False`.
```

Update suggested pairs text:

```markdown
Adaptive mode clamps its estimates to the documented range. These pairs remain manual-mode anchors, not guaranteed optima.
```

- [ ] **Step 3: Run full tests**

Run:

```bash
python -m pytest -q
```

Expected: PASS.

- [ ] **Step 4: Inspect final diff**

Run:

```bash
git diff --stat
git diff -- src/destripe/core.py src/destripe/ops.py src/destripe/adaptive.py tests/test_core.py README.md
```

Expected: Diff only covers adaptive/directions implementation, tests, and README.

- [ ] **Step 5: Commit**

Run:

```bash
git add README.md tests/test_core.py src/destripe/core.py src/destripe/ops.py src/destripe/adaptive.py
git commit -m "docs: document adaptive destriping"
```

## Final Verification

- [ ] Run:

```bash
python -m pytest -q
```

Expected: all tests pass.

- [ ] Run:

```bash
git status --short --branch
```

Expected: clean working tree, branch ahead by the feature commits.

- [ ] Report:

Include the final commit list, the full test command result, and whether adaptive tiled processing uses global directions with local smoothed `mu` values.
