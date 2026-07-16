from pathlib import Path

import cv2
import numpy as np
import pytest

from destripe.adaptive.directions import score_directions, supported_directions
from destripe.adaptive.preprocess import extract_high_pass, make_analysis_tensor
from destripe.automatic import _select_tile_count, automatic_clean
from destripe.preprocess import prepare_solver_gray


ASSET_DIR = Path(__file__).resolve().parents[1] / "asset"


def test_automatic_rejects_nonnumeric_gray() -> None:
    with pytest.raises(TypeError, match=r"^gray must be a numeric array\.$"):
        automatic_clean(np.array([["bad"]]), proj=True)


@pytest.mark.parametrize(
    "gray",
    (np.ones(8), np.ones((2, 2, 2)), np.empty((0, 8)), np.empty((8, 0))),
)
def test_automatic_rejects_non_image_shapes(gray: np.ndarray) -> None:
    with pytest.raises(ValueError, match="non-empty two-dimensional"):
        automatic_clean(gray, proj=True)


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_automatic_rejects_nonfinite_gray(value: float) -> None:
    gray = np.zeros((8, 8), dtype=np.float64)
    gray[0, 0] = value
    with pytest.raises(ValueError, match="finite values"):
        automatic_clean(gray, proj=True)


def test_automatic_noops_constant_and_tiny_images() -> None:
    for gray in (np.full((8, 8), 0.25), np.array([[0.25]])):
        result = automatic_clean(gray, proj=False)
        np.testing.assert_array_equal(result.clean, gray)
        assert result.directions == ()


def test_automatic_noops_without_coherent_stripe_evidence() -> None:
    rows, cols = np.indices((48, 64))
    images = (
        0.2 + 0.6 * np.exp(-((rows - 23.5) ** 2 + (cols - 31.5) ** 2) / 220),
        0.2 + 0.6 * (rows / 47 + cols / 63) / 2,
        np.clip(
            0.5 + np.random.default_rng(7).normal(0, 0.08, (48, 64)),
            0.0,
            1.0,
        ),
    )

    for image in images:
        result = automatic_clean(image, proj=True)

        assert result.directions == ()
        np.testing.assert_array_equal(result.clean, image)


def test_thin_analysis_uses_only_supported_directions_and_one_tile() -> None:
    image = np.linspace(0.0, 1.0, 2 * 2048).reshape(2, 2048)

    analysis = make_analysis_tensor(image)
    scores = score_directions(extract_high_pass(analysis))

    assert analysis.shape == image.shape
    assert supported_directions(analysis.shape) == (0, 2, 4)
    assert tuple(scores) == (0, 2, 4)
    assert _select_tile_count(image.shape) == 1


def test_automatic_restores_preferred_sample_configuration() -> None:
    encoded = cv2.imread(str(ASSET_DIR / "sample.jpeg"), cv2.IMREAD_GRAYSCALE)
    assert encoded is not None
    values = encoded.astype(np.float64)
    normalized = (values - values.min()) / (values.max() - values.min())
    processed = prepare_solver_gray(gray=normalized, process_size=512)

    result = automatic_clean(processed, proj=True)

    assert result.directions == (0, 4)
    assert result.mu1 == 0.25
    assert result.mu2 > 0.0
    assert result.confidence > 0.0
    assert result.clean.shape == processed.shape
    assert np.isfinite(result.clean).all()
    correction_rms = float(np.sqrt(np.mean((processed - result.clean) ** 2)))
    assert correction_rms == pytest.approx(0.03544478, abs=1e-6)


def test_automatic_is_deterministic() -> None:
    rows, cols = np.indices((48, 64))
    clean = 0.4 + 0.15 * np.exp(-((rows - 24) ** 2 + (cols - 32) ** 2) / 200)
    stripe = 0.03 * np.sin(np.linspace(0, 8 * np.pi, 64))[None, :]
    observed = np.clip(clean + stripe, 0.0, 1.0)

    first = automatic_clean(observed, proj=True)
    second = automatic_clean(observed, proj=True)

    assert first.directions == second.directions
    assert first.mu1 == second.mu1
    assert first.mu2 == second.mu2
    np.testing.assert_array_equal(first.clean, second.clean)
