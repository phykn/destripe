from dataclasses import dataclass
import time

import numpy as np
import torch

from .adaptive import estimate_adaptive_params, estimate_tile_mus
from .adaptive.refine import refine_clean
from .adaptive.stripe import (
    MIN_PROFILE_REPETITION,
    extract_sparse_profile_structure,
)
from .core import UniversalStripeRemover


AUTOMATIC_ITERATIONS = 1000
AUTOMATIC_TILES = 2
AUTOMATIC_OVERLAP = 64
AUTOMATIC_MIN_TILE_SIDE = 3
AUTOMATIC_FULL_FRAME_MAX_SIDE = 128
AUTOMATIC_MIN_CONFIDENCE = 0.1
AUTOMATIC_MIN_STRIPE_EVIDENCE = 0.02


@dataclass(frozen=True)
class AutomaticResult:
    clean: np.ndarray
    directions: tuple[int, ...]
    mu1: float
    mu2: float
    confidence: float
    elapsed_seconds: float


def automatic_clean(gray: np.ndarray, *, proj: bool) -> AutomaticResult:
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

    tiles = _select_tile_count(values.shape)
    tile_mus = (
        estimate_tile_mus(
            solver_values,
            tiles=tiles,
            directions=params.directions,
        )
        if tiles > 1
        else None
    )
    remover = UniversalStripeRemover(
        mu1=params.mu1,
        mu2=params.mu2,
        directions=params.directions,
    )
    solver_clean = remover.process_tiled(
        np.asarray(solver_values, dtype=np.float32),
        tiles=tiles,
        iterations=AUTOMATIC_ITERATIONS,
        overlap=AUTOMATIC_OVERLAP,
        proj=proj,
        tile_mus=tile_mus,
    ).numpy()
    refined_clean = refine_clean(
        gray=solver_values,
        clean=solver_clean,
        directions=params.directions,
        proj=proj,
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
