import numpy as np
import pytest

from destripe.hybrid import (
    MU1_CANDIDATES,
    MU2_CANDIDATES,
    ParameterCandidate,
    _evaluate_candidate,
    _parameter_candidates,
    _select_candidate,
)


def _target(strength: float) -> np.ndarray:
    profile = strength * np.sin(np.linspace(0, 10 * np.pi, 96))[None, :]
    return np.broadcast_to(profile, (80, 96)).copy()


def test_parameter_candidates_are_compact_positive_and_deterministic() -> None:
    gray = np.full((80, 96), 0.45) + _target(0.02)

    first = _parameter_candidates(gray, direction=0, target=_target(0.02))
    second = _parameter_candidates(gray, direction=0, target=_target(0.02))

    assert first == second
    assert {candidate.mu1 for candidate in first} == set(MU1_CANDIDATES)
    mu2_values = {candidate.mu2 for candidate in first}
    assert 1 <= len(mu2_values) <= 3
    assert all(np.isfinite(value) and value > 0.0 for value in mu2_values)
    assert len(first) == len(MU1_CANDIDATES) * len(mu2_values)


def test_parameter_candidates_follow_material_target_strength_change() -> None:
    gray = np.full((80, 96), 0.45)

    weak = _parameter_candidates(gray, direction=0, target=_target(0.005))
    strong = _parameter_candidates(gray, direction=0, target=_target(0.05))

    assert {item.mu2 for item in weak} != {item.mu2 for item in strong}


def test_parameter_candidates_stay_inside_solver_safe_grid() -> None:
    gray = np.full((32, 32), 0.45)

    tiny = _parameter_candidates(gray, direction=0, target=_target(1e-8)[:32, :32])
    huge = _parameter_candidates(gray, direction=0, target=_target(1.0)[:32, :32])

    for candidate in (*tiny, *huge):
        assert candidate.mu2 in MU2_CANDIDATES


@pytest.mark.parametrize("direction", (-1, 5))
def test_parameter_candidates_reject_invalid_direction(direction: int) -> None:
    gray = np.full((16, 16), 0.45)

    with pytest.raises(ValueError, match="direction"):
        _parameter_candidates(gray, direction=direction, target=np.zeros_like(gray))


def test_candidate_evaluation_rejects_nonpositive_target_projection() -> None:
    target = np.ones((4, 5))
    candidate = ParameterCandidate(mu1=0.2, mu2=0.01)

    assert (
        _evaluate_candidate(
            correction=-target,
            target=target,
            protection=np.zeros_like(target),
            candidate=candidate,
            iterations=20,
        )
        is None
    )


def test_candidate_evaluation_analytically_scales_to_target() -> None:
    target = np.ones((4, 5))
    candidate = ParameterCandidate(mu1=0.2, mu2=0.01)

    evaluated = _evaluate_candidate(
        correction=2.0 * target,
        target=target,
        protection=np.zeros_like(target),
        candidate=candidate,
        iterations=20,
    )

    assert evaluated is not None
    assert evaluated.beta == pytest.approx(0.5)
    assert evaluated.explained_fraction == pytest.approx(1.0)
    np.testing.assert_allclose(evaluated.correction, target)


def test_candidate_evaluation_caps_orthogonal_energy_to_target_power() -> None:
    target = np.ones((2, 2))
    correction = np.array([[1.0, 1.0], [1.0, 9.0]])

    evaluated = _evaluate_candidate(
        correction=correction,
        target=target,
        protection=np.zeros_like(target),
        candidate=ParameterCandidate(mu1=0.2, mu2=0.01),
        iterations=20,
    )

    assert evaluated is not None
    assert np.sum(evaluated.correction**2) <= np.sum(target**2) + 1e-12


def test_candidate_evaluation_scales_with_detection_confidence() -> None:
    target = np.ones((2, 2))
    full = _evaluate_candidate(
        correction=target,
        target=target,
        protection=np.zeros_like(target),
        candidate=ParameterCandidate(mu1=0.2, mu2=0.01),
        iterations=20,
    )
    guarded = _evaluate_candidate(
        correction=target,
        target=target,
        protection=np.zeros_like(target),
        candidate=ParameterCandidate(mu1=0.2, mu2=0.01),
        iterations=20,
        confidence=0.25,
    )

    assert full is not None and guarded is not None
    assert guarded.beta == pytest.approx(full.beta * 0.5)


def test_selection_prefers_explanation_then_protection_then_energy_then_mu() -> None:
    target = np.ones((2, 2))
    protection = np.array([[1.0, 0.0], [0.0, 0.0]])
    candidates = (
        (ParameterCandidate(0.25, 0.01), np.full((2, 2), 0.5)),
        (ParameterCandidate(0.25, 0.02), np.array([[2.0, 1.0], [1.0, 0.0]])),
        (
            ParameterCandidate(0.20, 0.01),
            np.array([[0.0, 1.0], [1.0, 2.0]]),
        ),
        (
            ParameterCandidate(1 / 6, 1 / 300),
            np.array([[0.0, 1.0], [1.0, 2.0]]),
        ),
    )
    evaluated = tuple(
        result
        for parameter, correction in candidates
        if (
            result := _evaluate_candidate(
                correction=correction,
                target=target,
                protection=protection,
                candidate=parameter,
                iterations=40,
            )
        )
        is not None
    )

    selected = _select_candidate(evaluated)

    assert selected is not None
    assert selected.candidate == ParameterCandidate(1 / 6, 1 / 300)
