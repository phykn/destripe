import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np

from destripe.adaptive import estimate_adaptive_params, estimate_tile_mus
from destripe.adaptive.refine import refine_clean
from destripe.core import UniversalStripeRemover
from destripe.preprocess import prepare_solver_gray, resize_to_shape


BASELINE_PROCESS_SIZE = 512
BASELINE_ITERATIONS = 1000
BASELINE_TILES = 2


@dataclass(frozen=True)
class BaselineResult:
    clean: np.ndarray
    correction: np.ndarray
    directions: tuple[int, ...]
    mu1: float
    mu2: float
    confidence: float
    tile_mus: tuple[tuple[float, float], ...]
    elapsed_seconds: float


def run_adaptive_baseline(
    image: np.ndarray,
    *,
    process_size: int = BASELINE_PROCESS_SIZE,
) -> BaselineResult:
    input_array = np.asarray(image)
    if input_array.ndim != 2 or not np.issubdtype(input_array.dtype, np.number):
        raise ValueError("baseline image must be a two-dimensional numeric array.")
    if not np.isfinite(input_array).all():
        raise ValueError("baseline image must contain only finite values.")

    original_dtype = input_array.dtype
    values = input_array.astype(np.float64)
    low = float(values.min())
    scale = float(values.max()) - low
    if scale < 1e-12:
        return BaselineResult(
            clean=input_array.copy(),
            correction=np.zeros_like(values),
            directions=(),
            mu1=0.25,
            mu2=0.01,
            confidence=0.0,
            tile_mus=(),
            elapsed_seconds=0.0,
        )

    normalized = (values - low) / scale
    solver_gray = prepare_solver_gray(gray=normalized, process_size=process_size)
    params = estimate_adaptive_params(solver_gray)
    tile_mus = tuple(
        estimate_tile_mus(
            solver_gray,
            tiles=BASELINE_TILES,
            directions=params.directions,
        )
    )

    remover = UniversalStripeRemover(
        mu1=params.mu1,
        mu2=params.mu2,
        directions=params.directions,
        device="cpu",
    )
    started = time.perf_counter()
    solver_clean = remover.process_tiled(
        np.asarray(solver_gray, dtype=np.float32),
        tiles=BASELINE_TILES,
        iterations=BASELINE_ITERATIONS,
        overlap=64,
        proj=True,
        tile_mus=tile_mus,
    ).numpy()
    solver_clean = refine_clean(
        gray=solver_gray,
        clean=solver_clean,
        directions=params.directions,
        proj=True,
    )
    elapsed_seconds = time.perf_counter() - started

    correction = resize_to_shape(
        solver_gray - solver_clean,
        shape=normalized.shape,
    )
    clean_normalized = np.clip(normalized - correction, 0.0, 1.0)
    clean_values = clean_normalized * scale + low
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        clean_values = np.clip(clean_values, info.min, info.max)
    clean = clean_values.astype(original_dtype)
    return BaselineResult(
        clean=clean,
        correction=values - clean.astype(np.float64),
        directions=params.directions,
        mu1=params.mu1,
        mu2=params.mu2,
        confidence=params.confidence,
        tile_mus=tile_mus,
        elapsed_seconds=elapsed_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the adaptive-PDHG baseline.")
    parser.add_argument("--image", default="asset/sample_01.jpeg")
    args = parser.parse_args(argv)
    image = cv2.imread(str(Path(args.image)), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not read baseline image: {args.image}")
    result = run_adaptive_baseline(image)
    normalized_correction = result.correction.astype(np.float32)
    digest = hashlib.sha256(normalized_correction.tobytes()).hexdigest()
    print(f"directions={result.directions}")
    print(f"mu1={result.mu1:.8f}, mu2={result.mu2:.8f}")
    print(f"confidence={result.confidence:.6f}")
    print(f"tile_mus={result.tile_mus}")
    print(f"correction_rms={np.sqrt(np.mean(result.correction**2)):.8f}")
    print(f"correction_sha256={digest}")
    print(f"elapsed_seconds={result.elapsed_seconds:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
