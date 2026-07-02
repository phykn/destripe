import math

import numpy as np
import torch

from .constants import (
    EPS,
    MU1_DENOMINATORS,
    MU1_MAX,
    MU1_MIN,
    MU2_DENOMINATORS,
    MU2_MAX,
    MU2_MIN,
)
from .stripe import project


def estimate_strength(
    *,
    high_pass: torch.Tensor,
    selected_directions: tuple[int, ...],
    score_weights: np.ndarray,
    selection_weights: np.ndarray,
) -> tuple[float, float, float]:
    score_strength = _measure_concentration(score_weights)
    selection_strength = _measure_concentration(selection_weights)
    ambiguity = _measure_entropy(score_weights)
    direction_confidence = math.sqrt(score_strength * selection_strength) * (
        1 - ambiguity
    )
    stripe_coherence = _measure_stripe(
        high_pass=high_pass,
        selected_directions=selected_directions,
        selection_weights=selection_weights,
    )
    confidence = math.sqrt(direction_confidence * stripe_coherence)

    stripe_permission = math.sqrt(selection_strength * stripe_coherence)
    target_mu1 = _interpolate_log(
        low=MU1_MIN,
        high=MU1_MAX,
        position=stripe_permission,
    )
    target_mu2 = _interpolate_log(
        low=MU2_MIN,
        high=MU2_MAX,
        position=1 - stripe_permission,
    )
    mu1 = _snap_log(value=target_mu1, denominators=MU1_DENOMINATORS)
    mu2 = _snap_log(value=target_mu2, denominators=MU2_DENOMINATORS)
    return mu1, mu2, confidence


def _measure_concentration(weights: np.ndarray) -> float:
    uniform_power = 1.0 / len(weights)
    power = float(np.sum(weights * weights))
    return min(1.0, max(0.0, (power - uniform_power) / (1.0 - uniform_power)))


def _measure_entropy(weights: np.ndarray) -> float:
    positive = weights[weights > 0.0]
    if positive.size <= 1:
        return 0.0
    entropy = -float(np.sum(positive * np.log(positive)))
    return min(1.0, max(0.0, entropy / math.log(len(weights))))


def _measure_stripe(
    *,
    high_pass: torch.Tensor,
    selected_directions: tuple[int, ...],
    selection_weights: np.ndarray,
) -> float:
    coherences = []
    weights = []
    for mode in selected_directions:
        stripe_img = project(high_pass, mode)
        stripe_amp = float(stripe_img.abs().mean().item())
        residual_amp = float((high_pass - stripe_img).abs().mean().item())
        coherences.append(stripe_amp / (stripe_amp + residual_amp + EPS))
        weights.append(float(selection_weights[mode]))

    if not coherences:
        return 0.0

    weight_array = np.array(weights, dtype=np.float64)
    total_weight = float(weight_array.sum())
    if total_weight > EPS:
        return _average_weighted(coherences, weight_array, total_weight)
    return float(np.mean(coherences))


def _average_weighted(
    values: list[float],
    weights: np.ndarray,
    total_weight: float,
) -> float:
    return float(np.sum(weights * np.array(values, dtype=np.float64)) / total_weight)


def _snap_log(value: float, denominators: tuple[int, ...]) -> float:
    candidates = [1 / denominator for denominator in denominators]
    log_value = math.log(max(value, EPS))
    return min(candidates, key=lambda candidate: abs(math.log(candidate) - log_value))


def _interpolate_log(*, low: float, high: float, position: float) -> float:
    clipped = min(1.0, max(0.0, position))
    if clipped <= 0:
        return float(low)
    if clipped >= 1:
        return float(high)
    return float(
        math.exp(math.log(low) * (1 - clipped) + math.log(high) * clipped)
    )
