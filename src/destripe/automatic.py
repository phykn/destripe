from dataclasses import dataclass
import time

import numpy as np
import torch

from .adaptive import estimate_adaptive_params, estimate_tile_mus
from .adaptive.analysis import extract_high_pass
from .adaptive.refine import refine_clean
from .adaptive.profiles import MIN_PROFILE_REPETITION, project
from .adaptive.structure import extract_sparse_profile_structure
from .core import UniversalStripeRemover
from .preprocess import prepare_solver_gray, resize_to_shape, validate_process_size

AUTOMATIC_ITERATIONS = 1000
AUTOMATIC_TILES = 2
AUTOMATIC_OVERLAP = 64
AUTOMATIC_MIN_TILE_SIDE = 3
AUTOMATIC_FULL_FRAME_MAX_SIDE = 128
AUTOMATIC_MIN_CONFIDENCE = 0.1
AUTOMATIC_RESIZED_REFINEMENT_PASSES = 4
AUTOMATIC_CPU_MAX_PIXELS = 128 * 128
AUTOMATIC_MIN_WORKING_REDUCTION = 0.2
AUTOMATIC_MAX_RESIZED_RESIDUAL_RATIO = 0.5
AUTOMATIC_MIN_STRIPE_EVIDENCE = 0.02


@dataclass(frozen=True)
class AutomaticResult:
    clean: np.ndarray
    directions: tuple[int, ...]
    mu1: float
    mu2: float
    confidence: float
    elapsed_seconds: float


def automatic_clean(
    gray: np.ndarray,
    *,
    process_size: int | None = None,
    proj: bool,
) -> AutomaticResult:
    process_size_value = validate_process_size(process_size)
    values = _validate_gray(gray)
    if min(values.shape) < 2 or float(np.ptp(values)) < 1e-12:
        return AutomaticResult(
            clean=np.clip(values, 0.0, 1.0) if proj else values.copy(),
            directions=(),
            mu1=0.25,
            mu2=0.01,
            confidence=0.0,
            elapsed_seconds=0.0,
        )

    started = time.perf_counter()
    params = estimate_adaptive_params(values)
    if (
        not params.directions
        or params.confidence < AUTOMATIC_MIN_CONFIDENCE
        or params.stripe_evidence < AUTOMATIC_MIN_STRIPE_EVIDENCE
        or params.profile_repetition < MIN_PROFILE_REPETITION
    ):
        return AutomaticResult(
            clean=np.clip(values, 0.0, 1.0) if proj else values.copy(),
            directions=(),
            mu1=params.mu1,
            mu2=params.mu2,
            confidence=params.confidence,
            elapsed_seconds=time.perf_counter() - started,
        )

    prepared = _prepare_solver_input(
        values,
        directions=params.directions,
        proj=proj,
    )
    if prepared is None:
        return AutomaticResult(
            clean=np.clip(values, 0.0, 1.0) if proj else values.copy(),
            directions=(),
            mu1=params.mu1,
            mu2=params.mu2,
            confidence=params.confidence,
            elapsed_seconds=time.perf_counter() - started,
        )
    solver_values, preserved_structure = prepared

    working_values = prepare_solver_gray(
        gray=solver_values,
        process_size=process_size_value,
    )
    is_resized = _can_use_working_resolution(
        native_values=solver_values,
        working_values=working_values,
        directions=params.directions,
    )
    if not is_resized:
        working_values = solver_values
    refined_clean = _solve_and_refine(
        native_values=solver_values,
        working_values=working_values,
        mu1=params.mu1,
        mu2=params.mu2,
        directions=params.directions,
        proj=proj,
        is_resized=is_resized,
    )
    if is_resized and not _working_result_is_safe(
        source=solver_values,
        clean=refined_clean,
        directions=params.directions,
    ):
        refined_clean = _solve_and_refine(
            native_values=solver_values,
            working_values=solver_values,
            mu1=params.mu1,
            mu2=params.mu2,
            directions=params.directions,
            proj=proj,
            is_resized=False,
        )
    clean = refined_clean + preserved_structure
    if proj:
        clean = np.clip(clean, 0.0, 1.0)
    return AutomaticResult(
        clean=clean,
        directions=params.directions,
        mu1=params.mu1,
        mu2=params.mu2,
        confidence=params.confidence,
        elapsed_seconds=time.perf_counter() - started,
    )


def _solve_and_refine(
    *,
    native_values: np.ndarray,
    working_values: np.ndarray,
    mu1: float,
    mu2: float,
    directions: tuple[int, ...],
    proj: bool,
    is_resized: bool,
) -> np.ndarray:
    tiles = _select_tile_count(working_values.shape)
    tile_mus = (
        estimate_tile_mus(
            native_values,
            tiles=tiles,
            directions=directions,
        )
        if tiles > 1
        else None
    )
    remover = UniversalStripeRemover(
        mu1=mu1,
        mu2=mu2,
        directions=directions,
        device="cpu" if working_values.size <= AUTOMATIC_CPU_MAX_PIXELS else None,
    )
    solver_clean = remover.process_tiled(
        np.asarray(working_values, dtype=np.float32),
        tiles=tiles,
        iterations=AUTOMATIC_ITERATIONS,
        overlap=AUTOMATIC_OVERLAP,
        proj=proj,
        tile_mus=tile_mus,
    ).numpy()
    if is_resized:
        working_correction = working_values - solver_clean
        solver_clean = native_values - resize_to_shape(
            working_correction,
            shape=native_values.shape,
        )
    refinement_passes = AUTOMATIC_RESIZED_REFINEMENT_PASSES if is_resized else 1
    refined_clean = solver_clean
    for _ in range(refinement_passes):
        next_clean = refine_clean(
            gray=native_values,
            clean=refined_clean,
            directions=directions,
            proj=proj,
        )
        if np.array_equal(next_clean, refined_clean):
            break
        refined_clean = next_clean
    return refined_clean


def _working_result_is_safe(
    *,
    source: np.ndarray,
    clean: np.ndarray,
    directions: tuple[int, ...],
) -> bool:
    source_high_pass = extract_high_pass(torch.as_tensor(source, dtype=torch.float32))
    clean_high_pass = extract_high_pass(torch.as_tensor(clean, dtype=torch.float32))
    for mode in directions:
        source_rms = float(project(source_high_pass, mode).square().mean().sqrt())
        clean_rms = float(project(clean_high_pass, mode).square().mean().sqrt())
        if clean_rms > source_rms * AUTOMATIC_MAX_RESIZED_RESIDUAL_RATIO:
            return False
    return True


def _can_use_working_resolution(
    *,
    native_values: np.ndarray,
    working_values: np.ndarray,
    directions: tuple[int, ...],
) -> bool:
    if working_values.shape == native_values.shape:
        return False
    reduction = 1.0 - working_values.size / native_values.size
    return reduction >= AUTOMATIC_MIN_WORKING_REDUCTION and _preserves_directions(
        working_values, directions=directions
    )


def _preserves_directions(
    values: np.ndarray,
    *,
    directions: tuple[int, ...],
) -> bool:
    working_params = estimate_adaptive_params(values)
    return all(mode in working_params.directions for mode in directions)


def _select_tile_count(shape: tuple[int, int]) -> int:
    if max(shape) <= AUTOMATIC_FULL_FRAME_MAX_SIDE:
        return 1
    core_shape = tuple((dim + AUTOMATIC_TILES - 1) // AUTOMATIC_TILES for dim in shape)
    if min(core_shape) < AUTOMATIC_MIN_TILE_SIDE:
        return 1
    return AUTOMATIC_TILES


def _prepare_solver_input(
    values: np.ndarray,
    *,
    directions: tuple[int, ...],
    proj: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    source = torch.as_tensor(values, dtype=torch.float32)
    structures = [extract_sparse_profile_structure(source, mode) for mode in directions]
    nonempty = [
        structure
        for structure in structures
        if float(structure.abs().max().item()) > 0.0
    ]
    # Projected sparse structures can overlap across modes. Preserve the input
    # rather than risk subtracting the same scene edge more than once.
    if len(directions) > 1 and nonempty:
        return None
    if not nonempty:
        return values, np.zeros_like(values)

    structure = nonempty[0].cpu().numpy().astype(np.float64)
    solver_values = values - structure
    if proj:
        low = float(solver_values.min())
        high = float(solver_values.max())
        if high - low > 1.0 + 1e-9:
            return None
        shift = (low + high - 1.0) / 2.0
        solver_values = solver_values - shift
        structure = structure + shift
    return solver_values, structure


def _validate_gray(gray: np.ndarray) -> np.ndarray:
    try:
        array = np.asarray(gray)
    except (TypeError, ValueError):
        raise TypeError("gray must be a numeric array.") from None
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError("gray must be a numeric array.")
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError("gray must be a non-empty two-dimensional array.")
    values = np.asarray(array, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("gray must contain only finite values.")
    return values
