import math

import numpy as np
import torch

from .constants import EPS, MU1_MAX, MU1_MIN, MU2_MAX, MU2_MIN
from .stripe import measure_parallel, measure_tv, project


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
        1.0 - ambiguity
    )
    stripe_coherence, stripe_amp, parallel_cost, tv_cost = _measure_stripe(
        high_pass=high_pass,
        selected_directions=selected_directions,
        selection_weights=selection_weights,
    )
    confidence = math.sqrt(direction_confidence * stripe_coherence)

    stripe_permission = math.sqrt(selection_strength * stripe_coherence)
    target_mu2 = _interpolate_log(
        low=MU2_MIN,
        high=MU2_MAX,
        position=1.0 - stripe_permission,
    )
    mu1, mu2 = _fit_cost_boundary(
        target_mu2=target_mu2,
        stripe_amp=stripe_amp,
        parallel_cost=parallel_cost,
        tv_cost=tv_cost,
    )
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
) -> tuple[float, float, float, float]:
    coherences = []
    amplitudes = []
    parallel_costs = []
    tv_costs = []
    weights = []
    for mode in selected_directions:
        stripe_img = project(high_pass, mode)
        stripe_amp = float(stripe_img.abs().mean().item())
        residual_amp = float((high_pass - stripe_img).abs().mean().item())
        coherences.append(stripe_amp / (stripe_amp + residual_amp + EPS))
        amplitudes.append(stripe_amp)
        parallel_costs.append(measure_parallel(stripe_img, mode))
        tv_costs.append(measure_tv(stripe_img))
        weights.append(float(selection_weights[mode]))

    if not coherences:
        return 0.0, 0.0, 0.0, 0.0

    weight_array = np.array(weights, dtype=np.float64)
    total_weight = float(weight_array.sum())
    if total_weight > EPS:
        return (
            _average_weighted(coherences, weight_array, total_weight),
            _average_weighted(amplitudes, weight_array, total_weight),
            _average_weighted(parallel_costs, weight_array, total_weight),
            _average_weighted(tv_costs, weight_array, total_weight),
        )
    return (
        float(np.mean(coherences)),
        float(np.mean(amplitudes)),
        float(np.mean(parallel_costs)),
        float(np.mean(tv_costs)),
    )


def _fit_cost_boundary(
    *,
    target_mu2: float,
    stripe_amp: float,
    parallel_cost: float,
    tv_cost: float,
) -> tuple[float, float]:
    if stripe_amp <= EPS or tv_cost <= EPS:
        return MU1_MIN, MU2_MAX

    # Keep a coherent stripe candidate preferable under the solver cost.
    mu2_limit = (
        MU1_MIN * tv_cost - parallel_cost
    ) / (stripe_amp + EPS)
    if target_mu2 <= mu2_limit:
        return MU1_MIN, target_mu2
    if mu2_limit >= MU2_MIN:
        return MU1_MIN, float(np.nextafter(mu2_limit, 0.0))

    required_mu1 = (
        parallel_cost + MU2_MIN * stripe_amp
    ) / (tv_cost + EPS)
    if required_mu1 <= MU1_MAX:
        return max(MU1_MIN, float(required_mu1)), MU2_MIN
    return MU1_MIN, MU2_MAX


def _average_weighted(
    values: list[float],
    weights: np.ndarray,
    total_weight: float,
) -> float:
    return float(np.sum(weights * np.array(values, dtype=np.float64)) / total_weight)


def _interpolate_log(*, low: float, high: float, position: float) -> float:
    clipped = min(1.0, max(0.0, position))
    if clipped <= 0.0:
        return float(low)
    if clipped >= 1.0:
        return float(high)
    return float(
        math.exp(math.log(low) * (1.0 - clipped) + math.log(high) * clipped)
    )
