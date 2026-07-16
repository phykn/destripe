from pathlib import Path

import cv2
import numpy as np
import pytest

from destripe.adaptive import AdaptiveParams, estimate_adaptive_params
from destripe.adaptive.constants import PARALLEL_OFFSETS
from destripe.adaptive.directions import (
    make_selection_weights,
    score_directions,
    select_directions,
    supported_directions,
)
from destripe.adaptive.preprocess import extract_high_pass, make_analysis_tensor
from destripe.adaptive.strength import _measure_concentration
from destripe.automatic import (
    AUTOMATIC_MIN_STRIPE_EVIDENCE,
    _select_tile_count,
    automatic_clean,
)
from destripe.preprocess import prepare_solver_gray


ASSET_DIR = Path(__file__).resolve().parents[1] / "asset"


def _make_multidirection_image(edge_amplitude: float) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices((32, 32))
    line_ids = rows - 2 * cols
    unique_ids = np.unique(line_ids)
    threshold = unique_ids[len(unique_ids) // 2]
    edge = edge_amplitude * (line_ids >= threshold)
    target = 0.5 + edge
    stripe = 0.03 * np.sin(2 * np.pi * 3 * cols / 32)
    return target + stripe, target


def _make_full_height_bars() -> np.ndarray:
    image = np.zeros((48, 64), dtype=np.float64)
    image[:, 8:16] = 0.6
    image[:, 40:52] = 1.0
    return image


def _make_directional_stripe(
    mode: int,
    *,
    shape: tuple[int, int] = (64, 64),
    cycles: int = 6,
    amplitude: float = 0.05,
) -> np.ndarray:
    rows, cols = np.indices(shape)
    row_step, col_step = PARALLEL_OFFSETS[mode]
    line_ids = col_step * rows - row_step * cols
    phase = (line_ids - line_ids.min()) / (line_ids.max() - line_ids.min() + 1)
    return 0.5 + amplitude * np.sin(2 * np.pi * cycles * phase)


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


def test_automatic_preserves_vertical_step_edge() -> None:
    image = np.zeros((48, 64), dtype=np.float64)
    image[:, 32:] = 1.0

    result = automatic_clean(image, proj=True)

    assert result.directions == ()
    np.testing.assert_array_equal(result.clean, image)


@pytest.mark.parametrize("row_bounds", ((16, 32), (0, 32), (0, 36), (6, 42)))
def test_automatic_preserves_partial_height_vertical_structures(
    row_bounds: tuple[int, int],
) -> None:
    image = np.zeros((48, 64), dtype=np.float64)
    start, stop = row_bounds
    image[start:stop, 8:16] = 0.6
    image[start:stop, 40:52] = 1.0

    result = automatic_clean(image, proj=True)

    assert result.directions == ()
    np.testing.assert_array_equal(result.clean, image)


def test_automatic_preserves_full_height_repeated_vertical_structures() -> None:
    image = _make_full_height_bars()

    result = automatic_clean(image, proj=True)

    assert result.directions == ()
    np.testing.assert_array_equal(result.clean, image)


def test_automatic_preserves_clustered_equal_height_vertical_structures() -> None:
    image = np.zeros((48, 64), dtype=np.float64)
    image[:, 8:16] = 0.8
    image[:, 24:32] = 0.8

    result = automatic_clean(image, proj=True)

    assert result.directions == ()
    np.testing.assert_array_equal(result.clean, image)


def test_automatic_preserves_regular_full_width_vertical_structures() -> None:
    image = np.zeros((48, 64), dtype=np.float64)
    for start in (4, 20, 36, 52):
        image[:, start : start + 8] = 0.8

    result = automatic_clean(image, proj=True)

    assert result.directions == ()
    np.testing.assert_array_equal(result.clean, image)


def test_automatic_preserves_monotone_vertical_staircase() -> None:
    image = np.zeros((48, 64), dtype=np.float64)
    image[:, 16:32] = 0.2
    image[:, 32:48] = 0.4
    image[:, 48:] = 0.6

    result = automatic_clean(image, proj=True)

    assert result.directions == ()
    np.testing.assert_array_equal(result.clean, image)


def test_automatic_removes_stripe_while_preserving_full_height_bars() -> None:
    target = _make_full_height_bars()
    _, cols = np.indices(target.shape)
    stripe = 0.03 * np.sin(2 * np.pi * 4 * cols / target.shape[1])
    observed = target + stripe

    result = automatic_clean(observed, proj=True)

    assert result.directions == (0,)
    input_rms = float(np.sqrt(np.mean((observed - target) ** 2)))
    result_rms = float(np.sqrt(np.mean((result.clean - target) ** 2)))
    assert result_rms < 0.3 * input_rms
    assert float(np.max(np.abs(result.clean - target))) < 0.01


def test_automatic_removes_stripe_without_damaging_vertical_step() -> None:
    target = np.zeros((48, 64), dtype=np.float64)
    target[:, 32:] = 1.0
    _, cols = np.indices(target.shape)
    stripe = 0.03 * np.sin(2 * np.pi * 4 * cols / target.shape[1])

    result = automatic_clean(target + stripe, proj=True)

    assert result.directions == (0,)
    np.testing.assert_allclose(result.clean, target, atol=1e-7)


@pytest.mark.parametrize("mode", (2, 4))
def test_automatic_preserves_small_diagonal_repeated_structures(mode: int) -> None:
    rows, cols = np.indices((8, 8))
    row_step, col_step = PARALLEL_OFFSETS[mode]
    line_ids = col_step * rows - row_step * cols
    normalized = (line_ids - line_ids.min()) / (line_ids.max() - line_ids.min() + 1)
    image = (
        0.2
        + 0.45 * ((normalized >= 0.12) & (normalized < 0.25))
        + 0.7 * ((normalized >= 0.58) & (normalized < 0.78))
    )

    result = automatic_clean(image, proj=True)

    assert result.directions == ()
    np.testing.assert_array_equal(result.clean, image)


def test_automatic_preserves_gradient_with_unstructured_noise() -> None:
    rows, cols = np.indices((48, 64))
    target = 0.2 + 0.6 * (rows / 47 + cols / 63) / 2
    image = target + np.random.default_rng(123).normal(0.0, 0.01, target.shape)

    result = automatic_clean(image, proj=True)

    assert result.directions == ()
    np.testing.assert_array_equal(result.clean, image)


def test_gradient_with_low_noise_is_rejected_across_random_seeds() -> None:
    rows, cols = np.indices((48, 64))
    target = 0.2 + 0.6 * (rows / 47 + cols / 63) / 2

    for seed in range(100):
        image = target + np.random.default_rng(seed).normal(0.0, 0.005, target.shape)

        assert estimate_adaptive_params(image).directions == ()


def test_vertical_gradient_decision_is_resolution_invariant() -> None:
    images = tuple(
        np.broadcast_to(np.linspace(0.0, 1.0, width), (height, width)).copy()
        for height, width in ((48, 64), (128, 128))
    )
    params = tuple(
        estimate_adaptive_params(image, fixed_directions=(0,)) for image in images
    )

    assert params[0].stripe_evidence == pytest.approx(
        params[1].stripe_evidence,
        rel=1e-4,
    )
    assert all(item.profile_repetition < 1.0 for item in params)
    for image in images:
        assert estimate_adaptive_params(image).directions == ()
        result = automatic_clean(image, proj=True)
        assert result.directions == ()
        np.testing.assert_array_equal(result.clean, image)


def test_smooth_stripe_repetition_is_resolution_invariant() -> None:
    for width in (64, 128, 256, 512):
        height = round(width * 0.75)
        profile = 0.5 + 0.05 * np.sin(
            np.linspace(0.0, 4 * np.pi, width, endpoint=False)
        )
        image = np.broadcast_to(profile, (height, width)).copy()

        params = estimate_adaptive_params(image)

        assert params.directions == (0,)
        assert params.profile_repetition == 1.0


@pytest.mark.parametrize(
    ("kind", "width", "cycles"),
    (
        ("square", 64, 8),
        ("square", 128, 16),
        ("sawtooth", 64, 2),
        ("sawtooth", 64, 4),
        ("sawtooth", 64, 8),
        ("sawtooth", 128, 2),
        ("sawtooth", 128, 4),
        ("sawtooth", 128, 8),
    ),
)
def test_nonsmooth_repeated_stripes_are_not_treated_as_sparse_structure(
    kind: str,
    width: int,
    cycles: int,
) -> None:
    phase = (cycles * np.arange(width) / width) % 1.0
    profile = (
        np.where(phase < 0.5, -0.05, 0.05) if kind == "square" else 0.1 * (phase - 0.5)
    )
    image = np.broadcast_to(0.5 + profile, (round(width * 0.75), width)).copy()

    params = estimate_adaptive_params(image)

    assert params.directions == (0,)
    assert params.profile_repetition == 1.0


def test_long_low_frequency_sawtooth_is_not_treated_as_sparse_structure() -> None:
    width = 512
    phase = (2 * np.arange(width) / width) % 1.0
    profile = 0.1 * (phase - 0.5)
    image = np.broadcast_to(0.5 + profile, (384, width)).copy()

    params = estimate_adaptive_params(image)

    assert params.directions == (0,)
    assert params.profile_repetition == 1.0


def test_native_analysis_preserves_narrow_high_resolution_stripes() -> None:
    for width in (512, 1024):
        profile = 0.5 + 0.05 * ((-1.0) ** np.arange(width))
        image = np.broadcast_to(profile, (256, width)).copy()

        analysis = make_analysis_tensor(image)
        params = estimate_adaptive_params(image)

        assert analysis.shape == image.shape
        assert float((analysis.max() - analysis.min()).item()) > 0.0
        assert params.directions == (0,)


def test_native_analysis_preserves_stripes_above_the_former_budget() -> None:
    profile = 0.5 + 0.05 * ((-1.0) ** np.arange(1024))
    image = np.broadcast_to(profile, (257, 1024)).copy()

    analysis = make_analysis_tensor(image)
    params = estimate_adaptive_params(image)

    assert analysis.shape == image.shape
    assert float((analysis.max() - analysis.min()).item()) > 0.0
    assert params.directions == (0,)


def test_aperiodic_stripes_have_distributed_profile_evidence() -> None:
    rng = np.random.default_rng(17)
    knots = rng.normal(0.0, 1.0, 17)
    profile = np.interp(np.linspace(0.0, 16.0, 64), np.arange(17), knots)
    profile -= profile.mean()
    profile *= 0.04 / profile.std()
    image = np.clip(0.45 + profile[None, :] + np.zeros((48, 1)), 0.0, 1.0)

    params = estimate_adaptive_params(image)

    assert params.directions == (0,)
    assert params.profile_repetition == 1.0
    assert params.stripe_evidence >= AUTOMATIC_MIN_STRIPE_EVIDENCE


@pytest.mark.parametrize("mode", range(5))
def test_direction_coverage_preserves_all_supported_stripe_modes(mode: int) -> None:
    params = estimate_adaptive_params(_make_directional_stripe(mode))

    assert params.directions == (mode,)
    assert params.profile_repetition == 1.0


def test_stripe_is_reselected_after_dominant_edge_is_rejected() -> None:
    image, target = _make_multidirection_image(edge_amplitude=0.3)

    high_pass = extract_high_pass(make_analysis_tensor(image))
    initial_weights = make_selection_weights(score_directions(high_pass))
    initial = select_directions(initial_weights)
    params = estimate_adaptive_params(image)
    result = automatic_clean(image, proj=True)

    assert initial == (1,)
    assert initial_weights[0] == 0.0
    assert params.directions == (0,)
    assert result.directions == (0,)
    input_rms = float(np.sqrt(np.mean((image - target) ** 2)))
    result_rms = float(np.sqrt(np.mean((result.clean - target) ** 2)))
    assert result_rms < input_rms


def test_direction_coverage_preserves_stripe_under_strong_edge() -> None:
    image, _ = _make_multidirection_image(edge_amplitude=0.35)

    params = estimate_adaptive_params(image)

    assert params.directions == (0,)
    assert params.profile_repetition == 1.0


def test_nonrepeated_secondary_direction_is_not_sent_to_solver() -> None:
    image, target = _make_multidirection_image(edge_amplitude=0.12)

    high_pass = extract_high_pass(make_analysis_tensor(image))
    initial = select_directions(make_selection_weights(score_directions(high_pass)))
    params = estimate_adaptive_params(image)
    result = automatic_clean(image, proj=True)

    assert initial == (0, 1)
    assert params.directions == (0,)
    assert result.directions == (0,)
    result_rms = float(np.sqrt(np.mean((result.clean - target) ** 2)))
    assert result_rms < 0.01


def test_automatic_preserves_sparse_structure_with_multiple_directions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = _make_full_height_bars()
    params = AdaptiveParams(
        directions=(0, 2),
        mu1=0.25,
        mu2=1 / 300,
        confidence=1.0,
        stripe_evidence=1.0,
        profile_repetition=1.0,
    )
    monkeypatch.setattr(
        "destripe.automatic.estimate_adaptive_params",
        lambda _: params,
    )

    result = automatic_clean(image, proj=True)

    assert result.directions == ()
    np.testing.assert_array_equal(result.clean, image)


def test_fixed_directions_are_not_refiltered_by_local_repetition() -> None:
    image, _ = _make_multidirection_image(edge_amplitude=0.2)

    params = estimate_adaptive_params(image, fixed_directions=(1, 0))

    assert params.directions == (1, 0)
    assert params.profile_repetition < 1.0


def test_single_direction_concentration_is_fully_concentrated() -> None:
    assert _measure_concentration(np.array([1.0])) == 1.0


def test_thin_analysis_uses_only_supported_directions_and_one_tile() -> None:
    image = np.linspace(0.0, 1.0, 2 * 2048).reshape(2, 2048)

    analysis = make_analysis_tensor(image)
    scores = score_directions(extract_high_pass(analysis))

    assert analysis.shape == image.shape
    assert supported_directions(analysis.shape) == (0, 2, 4)
    assert tuple(scores) == (0, 2, 4)
    assert _select_tile_count(image.shape) == 1


def test_small_images_use_full_frame_solver() -> None:
    assert _select_tile_count((64, 64)) == 1
    assert _select_tile_count((128, 128)) == 1
    assert _select_tile_count((256, 256)) == 2


def test_small_diagonal_stripe_full_frame_quality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("destripe.automatic.AUTOMATIC_ITERATIONS", 300)
    target = np.full((64, 64), 0.5)
    image = _make_directional_stripe(2, cycles=4, amplitude=0.03)

    result = automatic_clean(image, proj=True)

    assert result.directions == (2,)
    result_rms = float(np.sqrt(np.mean((result.clean - target) ** 2)))
    assert result_rms < 0.0045


def test_automatic_restores_preferred_sample_configuration() -> None:
    encoded = cv2.imread(str(ASSET_DIR / "sample.jpeg"), cv2.IMREAD_GRAYSCALE)
    assert encoded is not None
    values = encoded.astype(np.float64)
    normalized = (values - values.min()) / (values.max() - values.min())
    processed = prepare_solver_gray(gray=normalized, process_size=512)

    result = automatic_clean(processed, proj=True)

    assert result.directions == (0,)
    assert result.mu1 == 0.25
    assert result.mu2 > 0.0
    assert result.confidence > 0.0
    assert result.clean.shape == processed.shape
    assert np.isfinite(result.clean).all()
    correction_rms = float(np.sqrt(np.mean((processed - result.clean) ** 2)))
    assert correction_rms == pytest.approx(0.02971715, abs=1e-6)


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
