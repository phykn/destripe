import numpy as np
import pytest

from destripe.hybrid import (
    MU1_CANDIDATES,
    MU2_CANDIDATES,
    _parameter_candidates,
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
