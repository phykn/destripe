from dataclasses import dataclass

import numpy as np
import torch

from .constants import ALL_DIRECTIONS, EPS, MIN_DIRECTION_SCORE, MU1_DENOMINATORS
from .directions import (
    make_score_weights,
    make_selection_weights,
    measure_direction_coverage,
    score_directions,
    select_directions,
)
from .preprocess import extract_high_pass, make_analysis_tensor
from .strength import estimate_strength
from .stripe import MIN_PROFILE_REPETITION, measure_repetition


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
    if fixed is None:
        selected, stripe_weights, repetitions = _select_repeated_directions(
            analysis,
            high_pass,
            scores,
        )
    else:
        selected = fixed
        stripe_weights = _restrict_weights(selection_weights, selected)
    mu1 = 1 / MU1_DENOMINATORS[2]
    mu2, confidence, stripe_evidence, strength_repetition = estimate_strength(
        high_pass=high_pass,
        selected_directions=tuple(selected),
        supported_directions=supported,
        score_weights=score_weights,
        selection_weights=selection_weights,
        stripe_weights=stripe_weights,
    )
    profile_repetition = (
        min((repetitions[mode] for mode in selected), default=0.0)
        if fixed is None
        else strength_repetition
    )
    return AdaptiveParams(
        directions=tuple(selected),
        mu1=mu1,
        mu2=mu2,
        confidence=confidence,
        stripe_evidence=stripe_evidence,
        profile_repetition=profile_repetition,
    )


def _select_repeated_directions(
    analysis: torch.Tensor,
    high_pass: torch.Tensor,
    scores: dict[int, float],
) -> tuple[tuple[int, ...], np.ndarray, dict[int, float]]:
    repetitions = {}
    repeated_scores = {}
    for mode, score in scores.items():
        if score <= MIN_DIRECTION_SCORE:
            continue
        repetition = min(
            max(
                measure_repetition(analysis, mode),
                measure_repetition(high_pass, mode),
            ),
            measure_direction_coverage(analysis, mode),
        )
        if repetition >= MIN_PROFILE_REPETITION:
            repeated_scores[mode] = score
            repetitions[mode] = repetition

    if not repeated_scores:
        return (), np.zeros(len(ALL_DIRECTIONS), dtype=np.float64), repetitions

    candidate_weights = make_selection_weights(repeated_scores)
    return select_directions(candidate_weights), candidate_weights, repetitions


def _restrict_weights(
    weights: np.ndarray,
    directions: tuple[int, ...],
) -> np.ndarray:
    restricted = np.zeros_like(weights)
    if not directions:
        return restricted

    indices = list(directions)
    total = float(weights[indices].sum())
    if total > EPS:
        restricted[indices] = weights[indices] / total
    else:
        restricted[indices] = 1.0 / len(indices)
    return restricted


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
