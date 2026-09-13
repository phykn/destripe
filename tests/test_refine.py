import numpy as np
import pytest

from destripe.adaptive.constants import PARALLEL_OFFSETS
from destripe.adaptive.refine import refine_clean


def _make_vertical_stripe(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    profile = 0.03 * np.sin(2 * np.pi * 4 * np.arange(width) / width)
    return np.broadcast_to(profile, (height, width)).copy()


def test_refinement_preserves_a_clean_step_edge() -> None:
    target = np.zeros((48, 64), dtype=np.float64)
    target[:, 32:] = 1.0
    observed = target + _make_vertical_stripe(target.shape)

    refined = refine_clean(
        gray=observed,
        clean=target,
        directions=(0,),
        proj=True,
    ).clean

    np.testing.assert_array_equal(refined, target)


def test_refinement_removes_supported_residual_stripe() -> None:
    rows, cols = np.indices((48, 64))
    target = 0.4 + 0.15 * np.exp(-((rows - 24) ** 2 + (cols - 32) ** 2) / 200)
    stripe = _make_vertical_stripe(target.shape)
    observed = target + stripe
    solver_clean = target + 0.4 * stripe

    refined = refine_clean(
        gray=observed,
        clean=solver_clean,
        directions=(0,),
        proj=True,
    ).clean

    before_rms = float(np.sqrt(np.mean((solver_clean - target) ** 2)))
    after_rms = float(np.sqrt(np.mean((refined - target) ** 2)))
    assert after_rms < before_rms


def test_refinement_requires_solver_residual_support() -> None:
    rows, cols = np.indices((48, 64))
    clean = 0.4 + 0.03 * np.sin(2 * np.pi * 4 * cols / 64)

    refined = refine_clean(
        gray=clean,
        clean=clean,
        directions=(0,),
        proj=True,
    ).clean

    np.testing.assert_array_equal(refined, clean)


def test_refinement_is_bounded_by_weak_residual_support() -> None:
    rows, cols = np.indices((48, 64))
    pattern = np.sin(2 * np.pi * 4 * cols / 64)
    clean = 0.4 + 0.03 * pattern
    observed = clean + 1e-4 * pattern

    refined = refine_clean(
        gray=observed,
        clean=clean,
        directions=(0,),
        proj=True,
    ).clean

    correction_rms = float(np.sqrt(np.mean((refined - clean) ** 2)))
    residual_rms = float(np.sqrt(np.mean((observed - clean) ** 2)))
    assert correction_rms <= residual_rms


@pytest.mark.parametrize("passes", (1, 4, 8, 32))
@pytest.mark.parametrize("directions", ((0,), (2,), (0, 2, 4)))
@pytest.mark.parametrize("proj", (False, True))
def test_refinement_shares_initial_budget_across_passes_and_directions(
    passes: int,
    directions: tuple[int, ...],
    proj: bool,
) -> None:
    rows, cols = np.indices((48, 64))
    pattern = np.zeros_like(rows, dtype=np.float64)
    for mode in directions:
        row_step, col_step = PARALLEL_OFFSETS[mode]
        pattern += np.sin(2 * np.pi * (col_step * rows - row_step * cols) / 8)
    source = 0.4 + 0.03 * pattern
    initial = source - 0.001 * pattern

    result = refine_clean(
        gray=source,
        clean=initial,
        directions=directions,
        proj=proj,
        passes=passes,
    )

    initial_rms = float(np.sqrt(np.mean((source - initial) ** 2)))
    additional_rms = float(np.sqrt(np.mean((result.clean - initial) ** 2)))
    assert additional_rms <= initial_rms + 1e-9
    assert abs(float((result.clean - initial).mean())) < 1e-12


def test_repeated_refinement_still_improves_supported_stripes() -> None:
    rows, cols = np.indices((48, 64))
    target = 0.4 + 0.15 * np.exp(-((rows - 24) ** 2 + (cols - 32) ** 2) / 200)
    stripe = _make_vertical_stripe(target.shape)
    observed = target + stripe
    initial = target + 0.4 * stripe

    once = refine_clean(gray=observed, clean=initial, directions=(0,), proj=True)
    repeated = refine_clean(
        gray=observed, clean=initial, directions=(0,), proj=True, passes=8
    )
    exhausted = refine_clean(
        gray=observed, clean=initial, directions=(0,), proj=True, passes=32
    )

    before_rms = float(np.sqrt(np.mean((initial - target) ** 2)))
    once_rms = float(np.sqrt(np.mean((once.clean - target) ** 2)))
    repeated_rms = float(np.sqrt(np.mean((repeated.clean - target) ** 2)))
    assert repeated_rms < once_rms < before_rms
    assert repeated.limited
    np.testing.assert_array_equal(exhausted.clean, repeated.clean)


def test_repeated_refinement_cannot_invent_support_in_another_direction() -> None:
    rows, cols = np.indices((48, 64))
    source = 0.4 + 0.03 * np.sin(2 * np.pi * (rows - cols) / 8)
    # A brightness offset supplies a global residual budget but no stripe evidence.
    initial = source - 0.001

    result = refine_clean(
        gray=source, clean=initial, directions=(0, 2), proj=True, passes=32
    )

    np.testing.assert_array_equal(result.clean, initial)
    assert not result.limited


def test_refinement_reports_an_exactly_exhausted_budget() -> None:
    pattern = np.broadcast_to(np.where(np.arange(8) % 2, -1.0, 1.0), (2, 8))
    source = 0.5 + pattern / 16
    initial = source - pattern / 128

    result = refine_clean(
        gray=source, clean=initial, directions=(0,), proj=True, passes=8
    )

    assert result.limited
    np.testing.assert_allclose(result.clean, initial - pattern / 128, atol=1e-9)


def test_refinement_does_not_report_a_limit_when_no_stripe_remains() -> None:
    pattern = np.broadcast_to(np.where(np.arange(8) % 2, -1.0, 1.0), (2, 8))
    source = 0.5 + pattern / 16
    initial = source - pattern / 32

    result = refine_clean(
        gray=source, clean=initial, directions=(0,), proj=True, passes=8
    )

    assert not result.limited
    np.testing.assert_array_equal(result.clean, np.full_like(source, 0.5))
