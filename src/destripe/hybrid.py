from dataclasses import dataclass
import math

import numpy as np


MU1_CANDIDATES = (1 / 6, 1 / 5, 1 / 4, 1 / 3)
MU2_CANDIDATES = tuple(1 / value for value in (300, 240, 180, 120, 100, 90, 60))

_NORMAL_MAD_SCALE = 0.6744897501960817
_EPS = 1e-9


@dataclass(frozen=True)
class ParameterCandidate:
    mu1: float
    mu2: float


def _parameter_candidates(
    gray: np.ndarray,
    *,
    direction: int,
    target: np.ndarray,
) -> tuple[ParameterCandidate, ...]:
    gray_array = np.asarray(gray, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if gray_array.shape != target_array.shape or gray_array.ndim != 2:
        raise ValueError("gray and target must have the same two-dimensional shape.")
    if not np.isfinite(gray_array).all() or not np.isfinite(target_array).all():
        raise ValueError("gray and target must contain only finite values.")
    if isinstance(direction, bool) or direction not in range(5):
        raise ValueError("direction must be an integer from 0 through 4.")

    strength = _robust_target_strength(target_array)
    center_index = min(
        range(len(MU2_CANDIDATES)),
        key=lambda index: abs(
            math.log(max(strength, _EPS))
            - math.log(MU2_CANDIDATES[index])
        ),
    )
    selected_indices = sorted(
        {
            max(0, center_index - 1),
            center_index,
            min(len(MU2_CANDIDATES) - 1, center_index + 1),
        }
    )
    selected_mu2 = tuple(MU2_CANDIDATES[index] for index in selected_indices)
    return tuple(
        ParameterCandidate(mu1=mu1, mu2=mu2)
        for mu1 in MU1_CANDIDATES
        for mu2 in selected_mu2
    )


def _robust_target_strength(target: np.ndarray) -> float:
    values = np.asarray(target, dtype=np.float64).reshape(-1)
    centered = values - float(np.median(values))
    mad = float(np.median(np.abs(centered)))
    if mad > _EPS:
        return mad / _NORMAL_MAD_SCALE
    standard_deviation = float(np.std(centered))
    if standard_deviation > _EPS:
        return standard_deviation
    return MU2_CANDIDATES[0]
