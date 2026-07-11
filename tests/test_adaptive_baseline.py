from pathlib import Path

import cv2
import numpy as np
import pytest

from benchmarks.adaptive_baseline import (
    run_adaptive_baseline,
)
from destripe.adaptive import estimate_adaptive_params, estimate_tile_mus
from destripe.preprocess import prepare_solver_gray


ASSET_DIR = Path(__file__).resolve().parents[1] / "asset"


def _sample_one_solver_gray() -> np.ndarray:
    image = cv2.imread(str(ASSET_DIR / "sample_01.jpeg"), cv2.IMREAD_GRAYSCALE)
    assert image is not None
    values = image.astype(np.float64)
    normalized = (values - values.min()) / (values.max() - values.min())
    return prepare_solver_gray(gray=normalized, process_size=512)


def test_restored_adaptive_estimator_matches_frozen_sample_one_diagnostics() -> None:
    gray = _sample_one_solver_gray()

    params = estimate_adaptive_params(gray)

    assert params.directions == (0, 4)
    assert params.mu1 == 0.25
    assert params.mu2 == 0.01
    assert params.confidence == pytest.approx(0.233, abs=0.002)


def test_restored_tile_parameters_match_frozen_sample_one_diagnostics() -> None:
    gray = _sample_one_solver_gray()

    tile_mus = np.array(
        estimate_tile_mus(
            gray,
            tiles=2,
            directions=(0, 4),
        )
    )

    np.testing.assert_allclose(tile_mus[:, 0], 0.25, atol=1e-15, rtol=0.0)
    np.testing.assert_allclose(
        tile_mus[:, 1],
        np.array([0.01150372, 0.01233593, 0.01040453, 0.01045183]),
        atol=5e-5,
        rtol=0.0,
    )


def test_baseline_runner_preserves_shape_dtype_and_is_deterministic() -> None:
    image = cv2.imread(str(ASSET_DIR / "sample_01.jpeg"), cv2.IMREAD_GRAYSCALE)
    assert image is not None

    first = run_adaptive_baseline(image, process_size=128)
    second = run_adaptive_baseline(image, process_size=128)

    assert first.clean.shape == image.shape
    assert first.clean.dtype == image.dtype
    assert first.directions == second.directions
    assert first.tile_mus == second.tile_mus
    np.testing.assert_array_equal(first.clean, second.clean)
    np.testing.assert_array_equal(first.correction, second.correction)
