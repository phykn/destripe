from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from destripe.automatic import (
    _blocked_repeatability,
    _extract_high_pass,
    _make_protection,
    _project_robust,
    automatic_clean,
)
from destripe.preprocess import prepare_solver_gray, resize_to_shape


_ASSET_DIR = Path(__file__).resolve().parents[1] / "asset"


def test_automatic_removes_faint_vertical_stripe_and_preserves_structure() -> None:
    rows, cols = np.indices((96, 96))
    clean = 0.45 + 0.2 * np.exp(
        -((rows - 48) ** 2 + (cols - 48) ** 2) / 300
    )
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

    assert np.sqrt(np.mean((result.clean - clean) ** 2)) < 0.0032


def test_blocked_repeatability_rejects_adjacent_smooth_structure() -> None:
    weights = torch.ones((64, 16), dtype=torch.float32)
    localized = torch.zeros_like(weights)
    localized[8:24, 5] = 0.02
    repeated = torch.zeros_like(weights)
    repeated[:, 5] = 0.02

    assert _blocked_repeatability(localized, mode=0, weights=weights) == 0.0
    assert _blocked_repeatability(repeated, mode=0, weights=weights) > 0.9


def test_weighted_primitives_noop_when_all_weights_are_zero() -> None:
    values = torch.arange(48, dtype=torch.float32).reshape(8, 6)
    weights = torch.zeros_like(values)

    assert torch.count_nonzero(_project_robust(values, mode=2, weights=weights)) == 0
    assert _blocked_repeatability(values, mode=2, weights=weights) == 0.0


def test_automatic_handles_tiny_arrays_with_finite_output() -> None:
    gray = np.array([[0.25]], dtype=np.float64)

    result = automatic_clean(gray, proj=False)

    assert result.clean.shape == gray.shape
    assert np.isfinite(result.clean).all()
    assert np.isfinite(result.alpha)


def test_automatic_tie_chooses_lower_mode() -> None:
    gray = np.full((8, 8), 0.25, dtype=np.float64)

    result = automatic_clean(gray, proj=False)

    assert result.direction == 0
    assert result.alpha == 0.0
    np.testing.assert_array_equal(result.clean, gray)


def test_automatic_applies_profile_through_a_protected_crossing() -> None:
    stripe = 0.04 * np.sin(np.linspace(0, 8 * np.pi, 96))[None, :]
    observed = np.broadcast_to(0.4 + stripe, (96, 96)).copy()
    observed[46:50] += 0.35

    tensor = torch.as_tensor(observed, dtype=torch.float32)
    protection = _make_protection(_extract_high_pass(tensor), mode=0)
    result = automatic_clean(observed, proj=False)
    correction = observed - result.clean

    assert float(protection[46:50].max()) > 0.5
    assert result.direction == 0
    assert result.alpha > 0.0
    np.testing.assert_allclose(correction[48], correction[24], atol=1e-7)


@pytest.mark.parametrize(
    ("strength", "strength_index", "expected_alpha", "expected_image_rmse"),
    (
        (0.01, 0, 0.6074783741352221, 0.008922481178076635),
        (0.03, 1, 0.8495958038398771, 0.018394820727646824),
        (0.06, 2, 0.8842938772147949, 0.038339197406545476),
    ),
)
def test_automatic_matches_frozen_h3_vertical_diagnostic(
    strength: float,
    strength_index: int,
    expected_alpha: float,
    expected_image_rmse: float,
) -> None:
    clean, observed = _make_frozen_vertical_case(
        strength=strength,
        strength_index=strength_index,
    )
    low = float(observed.min())
    scale = float(observed.max()) - low
    normalized = (observed - low) / scale
    processed = prepare_solver_gray(gray=normalized, process_size=256)

    result = automatic_clean(processed, proj=False)
    correction = resize_to_shape(processed - result.clean, shape=clean.shape)
    output = np.clip(normalized - correction, 0.0, 1.0) * scale + low
    image_rmse = float(np.sqrt(np.mean((output - clean) ** 2)))

    assert result.direction == 0
    assert abs(result.alpha - expected_alpha) < 1e-6
    assert abs(image_rmse - expected_image_rmse) < 1e-5


def _make_frozen_vertical_case(
    *,
    strength: float,
    strength_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    encoded = cv2.imread(str(_ASSET_DIR / "sample_02.png"), cv2.IMREAD_UNCHANGED)
    assert encoded is not None
    clean = encoded.astype(np.float64) / np.iinfo(encoded.dtype).max

    rng = np.random.default_rng(1234 + 10_000 + strength_index)
    profile = rng.normal(size=clean.shape[1])
    profile = np.convolve(profile, np.ones(9, dtype=np.float64) / 9, mode="same")
    pattern = np.broadcast_to(profile[::-1], clean.shape).copy()
    pattern -= float(pattern.mean())
    pattern /= float(pattern.std())
    return clean, np.clip(clean + strength * pattern, 0.0, 1.0)
