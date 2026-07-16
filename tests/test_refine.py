import numpy as np

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
    )

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
    )

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
    )

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
    )

    correction_rms = float(np.sqrt(np.mean((refined - clean) ** 2)))
    residual_rms = float(np.sqrt(np.mean((observed - clean) ** 2)))
    assert correction_rms <= residual_rms
