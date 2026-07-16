from dataclasses import dataclass

import numpy as np

from .constants import ALL_DIRECTIONS, MU1_DENOMINATORS
from .directions import (
    make_score_weights,
    make_selection_weights,
    score_directions,
    select_directions,
)
from .preprocess import extract_high_pass, make_analysis_tensor
from .strength import estimate_strength


@dataclass(frozen=True)
class AdaptiveParams:
    directions: tuple[int, ...]
    mu1: float
    mu2: float
    confidence: float
    stripe_evidence: float
    profile_repetition: float


def estimate_adaptive_params(
    gray: np.ndarray,
    *,
    fixed_directions: tuple[int, ...] | None = None,
) -> AdaptiveParams:
    fixed = (
        None if fixed_directions is None else _validate_fixed_modes(fixed_directions)
    )
    analysis = make_analysis_tensor(gray)
    high_pass = extract_high_pass(analysis)
    scores = score_directions(high_pass)
    supported = tuple(scores)
    score_weights = make_score_weights(scores)
    selection_weights = make_selection_weights(scores)
    selected = select_directions(selection_weights) if fixed is None else fixed
    mu1 = 1 / MU1_DENOMINATORS[2]
    mu2, confidence, stripe_evidence, profile_repetition = estimate_strength(
        high_pass=high_pass,
        selected_directions=tuple(selected),
        supported_directions=supported,
        score_weights=score_weights,
        selection_weights=selection_weights,
    )
    return AdaptiveParams(
        directions=tuple(selected),
        mu1=mu1,
        mu2=mu2,
        confidence=confidence,
        stripe_evidence=stripe_evidence,
        profile_repetition=profile_repetition,
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
        if mode not in ALL_DIRECTIONS or mode in seen:
            raise ValueError(message)
        normalized.append(mode)
        seen.add(mode)

    if not normalized:
        raise ValueError(message)
    return tuple(normalized)
