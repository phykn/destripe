from dataclasses import dataclass

import numpy as np

from . import constants, directions, preprocess, strength


@dataclass(frozen=True)
class AdaptiveParams:
    directions: tuple[int, ...]
    mu1: float
    mu2: float
    confidence: float


def estimate_adaptive_params(
    gray: np.ndarray,
    *,
    fixed_directions: tuple[int, ...] | None = None,
) -> AdaptiveParams:
    fixed = (
        None
        if fixed_directions is None
        else _validate_fixed_modes(fixed_directions)
    )
    analysis = preprocess.make_analysis_tensor(gray)
    high_pass = preprocess.extract_high_pass(analysis)
    scores = directions.score_directions(high_pass)
    score_weights = directions.make_score_weights(scores)
    selection_weights = directions.make_selection_weights(scores)
    selected = (
        directions.select_directions(selection_weights)
        if fixed is None
        else fixed
    )
    mu1, mu2, confidence = strength.estimate_strength(
        score_weights=score_weights,
        selection_weights=selection_weights,
    )
    return AdaptiveParams(
        directions=tuple(selected),
        mu1=mu1,
        mu2=mu2,
        confidence=confidence,
    )


def _validate_fixed_modes(requested: object) -> tuple[int, ...]:
    message = "directions must be a non-empty sequence of unique modes 0..4."
    if not isinstance(requested, (tuple, list)):
        raise ValueError(message)

    normalized: list[int] = []
    seen: set[int] = set()
    for mode in requested:
        if isinstance(mode, bool) or not isinstance(mode, int):
            raise ValueError(message)
        if mode not in constants.ALL_DIRECTIONS or mode in seen:
            raise ValueError(message)
        normalized.append(mode)
        seen.add(mode)

    if not normalized:
        raise ValueError(message)
    return tuple(normalized)
