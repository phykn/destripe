# destripe Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply findings F-01/F-02/F-03 from the 2026-04-11 refactor diagnosis — deduplicate solver-parameter validation, remove the redundant `_run` tiles branch, and rename the misleading `correction` scratch buffer in `_solve`.

**Architecture:** Three small, independent edits to the two-file `destripe` package. F-02 is applied first because it is a prerequisite for safely removing `ops.py`'s validation in F-01 (after F-02, `destripe()` always routes through `process_tiled`, which validates `tiles`/`overlap` itself). F-03 is a pure rename in `core._solve` and has no coupling to the others.

**Tech Stack:** Python 3, NumPy, PyTorch, pytest. Tests live in `tests/test_core.py` (36 tests) and are the sole verification layer — no separate lint/type-check step is configured in this repo.

---

## File Structure

No new files. All edits are confined to:

- `src/destripe/ops.py` — Task 1 (drop `_run` branch), Task 2 (drop validation block).
- `src/destripe/core.py` — Task 3 (rename `correction` → `scratch` in `_solve`).
- `tests/test_core.py` — not modified; used as regression oracle at every task.

Existing tests already assert the invariants this plan preserves:

- `test_invalid_iterations`, `test_invalid_tol`, `test_invalid_tiles`, `test_invalid_overlap`, `test_invalid_non_finite` (in both `TestProcess`/`TestProcessTiled`/`TestDestripe`) cover every validation path.
- `test_reproducible_output_for_fixed_input` locks numerical output of `destripe()` across calls.
- `TestAdjointConsistency` locks the mathematical identities the `_solve` buffers participate in.

---

## Task 1: Remove `_run`'s redundant `tiles>1` branch (F-02)

**Rationale:** `core.process_tiled` already falls back to `process` when `tiles <= 1` (see `src/destripe/core.py:128-135`). Duplicating that branch in `ops._run` adds zero value and blocks F-01.

**Files:**
- Modify: `src/destripe/ops.py:105-133`
- Test (regression oracle): `tests/test_core.py`

- [ ] **Step 1: Run the full test suite to establish a green baseline**

Run: `pytest tests/ -q`
Expected: `36 passed` in a few seconds. If anything fails, stop and investigate — the baseline must be clean before refactoring.

- [ ] **Step 2: Simplify `_run` to always call `process_tiled`**

In `src/destripe/ops.py`, replace the current `_run` body (lines 105-133) so it unconditionally delegates to `process_tiled`. The full replacement:

```python
def _run(
    remover: UniversalStripeRemover,
    gray: np.ndarray,
    iterations: int,
    tol: float,
    tiles: int,
    overlap: int,
    proj: bool,
    verbose: bool,
) -> np.ndarray:
    out = remover.process_tiled(
        image=gray,
        tiles=tiles,
        iterations=iterations,
        tol=tol,
        overlap=overlap,
        proj=proj,
        verbose=verbose,
    )
    return out.numpy().astype(np.float64)
```

Note: `process_tiled(tiles=1)` internally short-circuits to `process(...)` at `core.py:128-135`, so the numerical result for the `tiles=1` path is bit-identical to the previous direct call.

- [ ] **Step 3: Run tests to verify no regression**

Run: `pytest tests/ -q`
Expected: `36 passed`. Pay particular attention to:
- `TestDestripe::test_grayscale_float64`, `test_grayscale_uint8`, `test_rgb`, `test_single_channel` — exercise the `tiles=1` (default) path that changed.
- `TestDestripe::test_reproducible_output_for_fixed_input` — locks numerical output.

If any of these fail, revert this task and investigate; do not proceed.

- [ ] **Step 4: Commit**

```bash
git add src/destripe/ops.py
git commit -m "refactor: drop redundant tiles>1 branch in ops._run

process_tiled already short-circuits to process when tiles<=1,
so _run's branch is dead. Prerequisite for removing ops.py's
duplicated solver-parameter validation."
```

---

## Task 2: Remove `ops.destripe`'s duplicated solver-parameter validation (F-01)

**Rationale:** After Task 1, every call site in `destripe()` routes through `process_tiled`, which in turn calls `_validate_solver_params` and `_validate_tiling_params` in `core.py`. The four `ValueError` raises in `ops.py:49-56` are now byte-for-byte duplicates of `core`'s checks.

**Files:**
- Modify: `src/destripe/ops.py:49-56`
- Test (regression oracle): `tests/test_core.py`

- [ ] **Step 1: Confirm core's error messages match what tests expect**

Read `src/destripe/core.py:327-339` to confirm that `_validate_solver_params` and `_validate_tiling_params` raise with messages beginning with `"iterations must be a positive integer"`, `"tol must be non-negative"`, `"tiles must be a positive integer"`, `"overlap must be non-negative"`. These are the exact strings the `pytest.raises(ValueError, match=...)` clauses in `TestDestripe::test_invalid_iterations/tol/tiles/overlap` check against.

No code change in this step — it is a read-only guard so you know the next edit is safe.

- [ ] **Step 2: Delete the duplicated validation block in `destripe()`**

In `src/destripe/ops.py`, remove lines 49-56 (the four `if ... raise ValueError` blocks for `iterations`, `tol`, `tiles`, `overlap`). The function should now start its body with:

```python
    input_array = np.asarray(image)
    if not np.issubdtype(input_array.dtype, np.number):
        raise ValueError("image must contain numeric values.")
    if not np.isfinite(input_array).all():
        raise ValueError("image must not contain NaN or Inf values.")
```

Leave every other line of `destripe()` untouched. Do **not** remove the `numeric`/`finite` checks — those live only in `ops.py` and have no `core` equivalent; they are still needed.

- [ ] **Step 3: Run tests to verify validation still fires from core**

Run: `pytest tests/test_core.py::TestDestripe -q`
Expected: `16 passed`. In particular, these four tests must still pass — they prove the validation still fires, now from `core` instead of `ops`:
- `test_invalid_iterations`
- `test_invalid_tol`
- `test_invalid_tiles`
- `test_invalid_overlap`

If any fails with `DID NOT RAISE` or a message-match error, stop and investigate — likely a core error message was updated and this spec is stale.

- [ ] **Step 4: Run the full suite**

Run: `pytest tests/ -q`
Expected: `36 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/destripe/ops.py
git commit -m "refactor: drop duplicated solver-parameter validation in ops

iterations/tol/tiles/overlap are now validated once by
core.UniversalStripeRemover. ops.destripe keeps only its
numpy-specific checks (numeric dtype, finite values)."
```

---

## Task 3: Rename the `correction` scratch buffer in `_solve` (F-03)

**Rationale:** In `core._solve`, the buffer declared as `correction` at `core.py:234` is reused as (a) the `u + Σs = data` constraint residual, (b) the `proj` clamp residual, (c) the `_forward_diff` output, and (d) the TV-dual scaling multiplier. The name fits only (a) and misleads readers at (c)/(d). Renaming to `scratch` makes its role as a generic reusable buffer explicit and matches the nearby dedicated buffers (`directional_diff`, `grad_norm`) which are already named by use.

**Files:**
- Modify: `src/destripe/core.py:234-292` (declaration + every use inside `_solve`)
- Test (regression oracle): `tests/test_core.py`

- [ ] **Step 1: Confirm `correction` is a local variable of `_solve` only**

Run: `grep -n correction src/destripe/core.py`
Expected: every hit is inside `_solve` (line ~234 declaration plus uses up through ~292). There must be no other references — `correction` is not an attribute, not a parameter, not documented anywhere. If `grep` returns matches outside `_solve`, stop — the assumption of a localized rename has been violated and this task needs rescoping.

- [ ] **Step 2: Rename `correction` → `scratch` inside `_solve`**

In `src/destripe/core.py`, replace every occurrence of the local name `correction` with `scratch` inside the `_solve` method. This is a pure rename; do not change any other token, whitespace, or line.

Specifically, the following lines (using the current line numbers) must change:

- Line 234: `correction = torch.empty_like(input=data)` → `scratch = torch.empty_like(input=data)`
- Lines 260-266 (constraint enforcement block): every `correction` → `scratch`
- Lines 271-275 (`proj` clamp residual block): every `correction` → `scratch`
- Lines 281, 283 (`_forward_diff(..., out=correction)` calls): → `out=scratch`
- Line 289 (`torch.div(tv_dual_radius, grad_norm, out=correction)`): → `out=scratch`
- Line 290 (`correction.clamp_(max=1.0)`): → `scratch.clamp_(max=1.0)`
- Lines 291-292 (`grad_row.mul_(correction)` / `grad_col.mul_(correction)`): → `.mul_(scratch)`
- Line 299 (`self._dir_diff(..., out=directional_diff)`): **unchanged** — `directional_diff` is a different buffer.
- Line 314 (`torch.sub(input=clean, other=prev_clean, out=correction)`): → `out=scratch`

Do not rename `directional_diff`, `grad_norm`, `prev_clean`, or any other buffer. Do not introduce a second buffer. Do not touch adjoint helpers or `_dir_diff`. This is a name change, nothing more.

- [ ] **Step 3: Sanity-check with grep**

Run: `grep -n correction src/destripe/core.py`
Expected: **zero matches.** If any remain, fix the missed sites before moving on.

Run: `grep -n scratch src/destripe/core.py`
Expected: one declaration (`scratch = torch.empty_like...`) plus ~12 uses, all within `_solve`.

- [ ] **Step 4: Run the full test suite**

Run: `pytest tests/ -q`
Expected: `36 passed`. The critical regressions a bad rename would surface:
- `TestAdjointConsistency` (all 7 parametrized cases) — would detect any accidental change to buffer wiring that breaks the `<Dx,y> = <x,D^T y>` identities.
- `TestProcess::test_grayscale_2d`, `test_batch_3d`, `test_constant_image` — exercise the full `_solve` loop end-to-end.
- `TestDestripe::test_reproducible_output_for_fixed_input` — locks the bit-identical output of `destripe()` across runs; any silent numerical drift caused by rename-induced logic bugs fails here.

If any fail, the rename was not pure — diff against HEAD and find the typo.

- [ ] **Step 5: Commit**

```bash
git add src/destripe/core.py
git commit -m "refactor: rename correction scratch buffer in _solve

The same buffer is reused for the constraint residual, the proj
clamp residual, forward_diff output, and the TV dual multiplier.
'correction' only names the first use; 'scratch' names the role."
```

---

## Final Verification

- [ ] **Step 1: Full suite, fresh run**

Run: `pytest tests/ -v`
Expected: `36 passed` with no warnings beyond the pre-existing baseline.

- [ ] **Step 2: Inspect the three-commit diff**

Run: `git log --oneline -3` — should show three `refactor:` commits in the order: F-02, F-01, F-03.
Run: `git diff HEAD~3 -- src/destripe/ops.py src/destripe/core.py` and confirm:
- `ops.py` is shorter (validation block gone, `_run` slimmed).
- `core.py` has `scratch` where `correction` used to be, and nothing else changed.
- No other files are touched.

- [ ] **Step 3: Confirm success criteria from the spec**

Re-read `docs/superpowers/specs/2026-04-11-refactor-destripe-design.md` §"Success criteria". Verify:
1. (F-01) Validation exists in exactly one place (`core`).
2. (F-02) `_run` no longer branches on `tiles`.
3. (F-03) No `correction` identifier remains in `core.py`.
4. (Regression) 36/36 tests pass.

If all four hold, the plan is complete.
