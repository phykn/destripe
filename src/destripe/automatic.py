from dataclasses import dataclass
import time

import numpy as np

from .adaptive import estimate_adaptive_params, estimate_tile_mus
from .adaptive.refine import refine_clean
from .core import UniversalStripeRemover


AUTOMATIC_ITERATIONS = 1000
AUTOMATIC_TILES = 2
AUTOMATIC_OVERLAP = 64
AUTOMATIC_TOLERANCE = 1e-5


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
    tile_mus = estimate_tile_mus(
        values,
        tiles=AUTOMATIC_TILES,
        directions=params.directions,
    )
    remover = UniversalStripeRemover(
        mu1=params.mu1,
        mu2=params.mu2,
        directions=params.directions,
    )
    solver_clean = remover.process_tiled(
        np.asarray(values, dtype=np.float32),
        tiles=AUTOMATIC_TILES,
        iterations=AUTOMATIC_ITERATIONS,
        tol=AUTOMATIC_TOLERANCE,
        overlap=AUTOMATIC_OVERLAP,
        proj=proj,
        tile_mus=tile_mus,
    ).numpy()
    clean = refine_clean(
        gray=values,
        clean=solver_clean,
        directions=params.directions,
        proj=proj,
    )
    return AutomaticResult(
        clean=clean,
        directions=params.directions,
        mu1=params.mu1,
        mu2=params.mu2,
        confidence=params.confidence,
        elapsed_seconds=time.perf_counter() - started,
    )


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
