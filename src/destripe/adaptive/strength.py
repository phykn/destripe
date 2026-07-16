import math
from dataclasses import dataclass

import numpy as np
import torch

from .constants import (
    EPS,
    MU2_DENOMINATORS,
    NORMAL_MAD_SCALE,
)
from .stripe import measure_repetition, measure_shrinkage, project


@dataclass(frozen=True)
class _StripeStats:
    mode: int
    abs_values: np.ndarray
    sigma: float
    coherence: float
    repetition: float


def estimate_strength(
    *,
    high_pass: torch.Tensor,
    selected_directions: tuple[int, ...],
    supported_directions: tuple[int, ...],
    score_weights: np.ndarray,
    selection_weights: np.ndarray,
) -> tuple[float, float, float, float]:
    supported_indices = list(supported_directions)
    supported_scores = score_weights[supported_indices]
    supported_selection = selection_weights[supported_indices]
    score_strength = _measure_concentration(supported_scores)
    selection_strength = _measure_concentration(supported_selection)
    ambiguity = _measure_entropy(supported_scores)
    direction_confidence = math.sqrt(score_strength * selection_strength) * (
        1 - ambiguity
    )
    stripe_stats = _make_stripe_stats(
        high_pass=high_pass,
        selected_directions=selected_directions,
        direction_confidence=direction_confidence,
    )
    stripe_coherence = _measure_stripe(
        stripe_stats=stripe_stats,
        selection_weights=selection_weights,
    )
    confidence = math.sqrt(direction_confidence * stripe_coherence)
    stripe_amplitude = _measure_stripe_amplitude(
        stripe_stats=stripe_stats,
        selection_weights=selection_weights,
    )
    high_pass_amplitude = float(high_pass.abs().mean().item())
    relative_amplitude = stripe_amplitude / (high_pass_amplitude + EPS)
    repetition = _measure_primary_repetition(
        stripe_stats=stripe_stats,
        selection_weights=selection_weights,
    )

    mu2 = _estimate_mu2(
        stripe_stats=stripe_stats,
        selection_weights=selection_weights,
    )
    return mu2, confidence, confidence * relative_amplitude, repetition


def _make_stripe_stats(
    *,
    high_pass: torch.Tensor,
    selected_directions: tuple[int, ...],
    direction_confidence: float,
) -> list[_StripeStats]:
    stats = []
    for mode in selected_directions:
        stripe_img = project(high_pass, mode)
        values = stripe_img.detach().cpu().numpy().reshape(-1)
        reliability = measure_shrinkage(high_pass, mode)
        sigma = _estimate_sigma(values) * math.sqrt(
            max(0.0, 1 - reliability * direction_confidence)
        )
        stripe_amp = float(stripe_img.abs().mean().item())
        residual_amp = float((high_pass - stripe_img).abs().mean().item())
        coherence = stripe_amp / (stripe_amp + residual_amp + EPS)
        stats.append(
            _StripeStats(
                mode=mode,
                abs_values=np.abs(values),
                sigma=sigma,
                coherence=coherence,
                repetition=measure_repetition(high_pass, mode),
            )
        )
    return stats


def _estimate_mu2(
    *,
    stripe_stats: list[_StripeStats],
    selection_weights: np.ndarray,
) -> float:
    candidates = [1 / denominator for denominator in MU2_DENOMINATORS]
    risks = [
        _measure_mu2_risk(
            stripe_stats=stripe_stats,
            selection_weights=selection_weights,
            threshold=threshold,
        )
        for threshold in candidates
    ]
    best_risk = min(risks)
    # When SURE cannot separate thresholds, avoid inventing a stripe.
    tied = [
        candidate
        for candidate, risk in zip(candidates, risks)
        if risk <= best_risk + EPS
    ]
    return max(tied)


def _measure_mu2_risk(
    *,
    stripe_stats: list[_StripeStats],
    selection_weights: np.ndarray,
    threshold: float,
) -> float:
    risks = []
    weights = []
    for stats in stripe_stats:
        risks.append(
            _measure_sure(
                stats.abs_values,
                threshold=threshold,
                sigma=stats.sigma,
            )
        )
        weights.append(float(selection_weights[stats.mode]))

    if not risks:
        return 0.0

    weight_array = np.array(weights, dtype=np.float64)
    total_weight = float(weight_array.sum())
    if total_weight > EPS:
        return _average_weighted(risks, weight_array, total_weight)
    return float(np.mean(risks))


def _measure_sure(
    abs_values: np.ndarray,
    *,
    threshold: float,
    sigma: float,
) -> float:
    if abs_values.size == 0:
        return 0.0

    sigma2 = sigma * sigma
    bias = np.minimum(abs_values * abs_values, threshold * threshold)
    degrees = abs_values > threshold
    return float(np.mean(bias + 2 * sigma2 * degrees))


def _estimate_sigma(values: np.ndarray) -> float:
    if values.size == 0:
        return EPS

    centered = values - np.median(values)
    mad = float(np.median(np.abs(centered)))
    if mad > EPS:
        return mad / NORMAL_MAD_SCALE

    std = float(np.std(values))
    if std > EPS:
        return std
    return EPS


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
    stripe_stats: list[_StripeStats],
    selection_weights: np.ndarray,
) -> float:
    coherences = [stats.coherence for stats in stripe_stats]
    weights = []
    for stats in stripe_stats:
        weights.append(float(selection_weights[stats.mode]))

    if not coherences:
        return 0.0

    weight_array = np.array(weights, dtype=np.float64)
    total_weight = float(weight_array.sum())
    if total_weight > EPS:
        return _average_weighted(coherences, weight_array, total_weight)
    return float(np.mean(coherences))


def _measure_stripe_amplitude(
    *,
    stripe_stats: list[_StripeStats],
    selection_weights: np.ndarray,
) -> float:
    amplitudes = [float(np.mean(stats.abs_values)) for stats in stripe_stats]
    weights = [float(selection_weights[stats.mode]) for stats in stripe_stats]
    if not amplitudes:
        return 0.0

    weight_array = np.array(weights, dtype=np.float64)
    total_weight = float(weight_array.sum())
    if total_weight > EPS:
        return _average_weighted(amplitudes, weight_array, total_weight)
    return float(np.mean(amplitudes))


def _measure_primary_repetition(
    *,
    stripe_stats: list[_StripeStats],
    selection_weights: np.ndarray,
) -> float:
    if not stripe_stats:
        return 0.0
    primary = max(
        stripe_stats,
        key=lambda stats: (selection_weights[stats.mode], -stats.mode),
    )
    return primary.repetition


def _average_weighted(
    values: list[float],
    weights: np.ndarray,
    total_weight: float,
) -> float:
    return float(np.sum(weights * np.array(values, dtype=np.float64)) / total_weight)
