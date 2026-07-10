from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import destripe.automatic as automatic_module
from destripe.automatic import (
    _blocked_repeatability,
    _detect_h3,
    _extract_high_pass,
    _make_protection,
    _project_robust,
    automatic_clean,
)
from destripe.preprocess import prepare_solver_gray, resize_to_shape
from benchmarks.synthetic import make_stripe_pattern, make_support_mask


_ASSET_DIR = Path(__file__).resolve().parents[1] / "asset"


def test_automatic_rejects_nonnumeric_gray() -> None:
    gray = np.array([["not numeric"]])

    with pytest.raises(TypeError, match=r"^gray must be a numeric array\.$"):
        automatic_clean(gray, proj=True)


@pytest.mark.parametrize(
    "gray",
    (
        np.ones(8),
        np.ones((2, 2, 2)),
        np.empty((0, 8)),
        np.empty((8, 0)),
    ),
    ids=("one-dimensional", "three-dimensional", "zero-rows", "zero-columns"),
)
def test_automatic_rejects_non_image_shapes(gray: np.ndarray) -> None:
    with pytest.raises(
        ValueError,
        match=r"^gray must be a non-empty two-dimensional array\.$",
    ):
        automatic_clean(gray, proj=True)


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_automatic_rejects_nonfinite_gray(value: float) -> None:
    gray = np.zeros((8, 8), dtype=np.float64)
    gray[0, 0] = value

    with pytest.raises(ValueError, match=r"^gray must contain only finite values\.$"):
        automatic_clean(gray, proj=True)


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


def test_automatic_noops_outer_quarter_stripe_instead_of_filling_clean_gap() -> None:
    rows, cols = np.indices((96, 96))
    clean = 0.42 + 0.12 * np.exp(
        -((rows - 48) ** 2 + (cols - 48) ** 2) / 420
    )
    pattern = make_stripe_pattern(
        shape=clean.shape,
        kind="curtain",
        mode=0,
        rng=np.random.default_rng(908),
    )
    support = make_support_mask(
        clean.shape,
        kind="outer_quarters",
        mode=0,
        rng=np.random.default_rng(731),
    )
    observed = np.clip(clean + 0.03 * pattern * support, 0.0, 1.0)

    result = automatic_clean(observed, proj=False)
    unsupported = support == 0.0

    np.testing.assert_array_equal(result.clean[unsupported], observed[unsupported])
    assert np.mean((result.clean - clean) ** 2) <= np.mean((observed - clean) ** 2)


def test_h3_detection_reports_continuous_vertical_support() -> None:
    stripe = 0.03 * np.sin(np.linspace(0, 10 * np.pi, 96))[None, :]
    observed = np.broadcast_to(0.45 + stripe, (96, 96)).copy()

    detection = _detect_h3(observed)

    assert detection.direction == 0
    assert detection.consistent is True
    assert detection.reliability > 0.0
    assert detection.alpha > 0.0
    assert np.any(detection.target)


@pytest.mark.parametrize(
    "kind",
    ("outer_quarters", "first_half", "center", "segments"),
)
def test_h3_detection_rejects_interrupted_vertical_support(kind: str) -> None:
    clean = np.full((96, 96), 0.45)
    pattern = make_stripe_pattern(
        shape=clean.shape,
        kind="curtain",
        mode=0,
        rng=np.random.default_rng(908),
    )
    support = make_support_mask(
        clean.shape,
        kind=kind,
        mode=0,
        rng=np.random.default_rng(731),
    )

    detection = _detect_h3(clean + 0.03 * pattern * support)

    assert detection.consistent is False
    assert detection.alpha == 0.0
    assert not np.any(detection.target)


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


def test_automatic_evaluates_all_five_and_selects_top_reliability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluated: list[int] = []
    blocked_scores = {0: 0.04, 1: 0.16, 2: 0.36, 3: 1.0, 4: 0.64}

    def controlled_blocked_repeatability(
        tensor: torch.Tensor,
        mode: int,
        weights: torch.Tensor,
    ) -> float:
        del tensor, weights
        evaluated.append(mode)
        return blocked_scores[mode]

    monkeypatch.setattr(
        automatic_module,
        "_blocked_repeatability",
        controlled_blocked_repeatability,
    )
    monkeypatch.setattr(
        automatic_module,
        "_positive_centered_cosine",
        lambda _first, _second: 1.0,
    )

    result = automatic_clean(np.arange(64).reshape(8, 8), proj=False)

    assert evaluated == [0, 1, 2, 3, 4]
    assert result.direction == 3


def test_automatic_projection_clips_unprojected_result() -> None:
    gray = np.full((8, 8), 1.25, dtype=np.float64)

    unprojected = automatic_clean(gray, proj=False)
    projected = automatic_clean(gray, proj=True)

    assert np.any((unprojected.clean < 0.0) | (unprojected.clean > 1.0))
    np.testing.assert_array_equal(projected.clean, np.clip(unprojected.clean, 0.0, 1.0))
    assert projected.direction == unprojected.direction
    assert projected.alpha == unprojected.alpha


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
    assert not np.array_equal(correction[48], correction[24])
    np.testing.assert_allclose(correction[48], correction[24], atol=0.001)


@pytest.mark.parametrize(
    ("strength", "strength_index"),
    (
        (0.01, 0),
        (0.03, 1),
        (0.06, 2),
    ),
)
def test_automatic_improves_frozen_vertical_diagnostic(
    strength: float,
    strength_index: int,
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
    input_rmse = float(np.sqrt(np.mean((observed - clean) ** 2)))

    assert result.direction == 0
    assert result.alpha > 0.0
    assert image_rmse < input_rmse


def _make_frozen_vertical_case(
    *,
    strength: float,
    strength_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    encoded = cv2.imread(str(_ASSET_DIR / "sample_02.tif"), cv2.IMREAD_GRAYSCALE)
    assert encoded is not None
    clean = encoded.astype(np.float64) / np.iinfo(encoded.dtype).max

    rng = np.random.default_rng(1234 + 10_000 + strength_index)
    profile = rng.normal(size=clean.shape[1])
    profile = np.convolve(profile, np.ones(9, dtype=np.float64) / 9, mode="same")
    pattern = np.broadcast_to(profile[::-1], clean.shape).copy()
    pattern -= float(pattern.mean())
    pattern /= float(pattern.std())
    return clean, np.clip(clean + strength * pattern, 0.0, 1.0)
