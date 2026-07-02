import math

import numpy as np
import torch

from .constants import (
    EPS,
    MU2_DENOMINATORS,
    NORMAL_MAD_SCALE,
)
from .stripe import project


def estimate_strength(
    *,
    high_pass: torch.Tensor,
    selected_directions: tuple[int, ...],
    score_weights: np.ndarray,
    selection_weights: np.ndarray,
) -> tuple[float, float]:
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

    mu2 = _estimate_mu2(
        high_pass=high_pass,
        selected_directions=selected_directions,
        selection_weights=selection_weights,
    )
    return mu2, confidence


def _estimate_mu2(
    *,
    high_pass: torch.Tensor,
    selected_directions: tuple[int, ...],
    selection_weights: np.ndarray,
) -> float:
    candidates = [1 / denominator for denominator in MU2_DENOMINATORS]
    risks = [
        _measure_mu2_risk(
            high_pass=high_pass,
            selected_directions=selected_directions,
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
    high_pass: torch.Tensor,
    selected_directions: tuple[int, ...],
    selection_weights: np.ndarray,
    threshold: float,
) -> float:
    risks = []
    weights = []
    for mode in selected_directions:
        stripe_img = project(high_pass, mode)
        sigma = _estimate_sigma(stripe_img)
        risks.append(_measure_sure(stripe_img, threshold=threshold, sigma=sigma))
        weights.append(float(selection_weights[mode]))

    if not risks:
        return 0.0

    weight_array = np.array(weights, dtype=np.float64)
    total_weight = float(weight_array.sum())
    if total_weight > EPS:
        return _average_weighted(risks, weight_array, total_weight)
    return float(np.mean(risks))


def _measure_sure(
    values: torch.Tensor,
    *,
    threshold: float,
    sigma: float,
) -> float:
    arr = values.detach().cpu().numpy().reshape(-1)
    if arr.size == 0:
        return 0.0

    abs_arr = np.abs(arr)
    sigma2 = sigma * sigma
    bias = np.minimum(abs_arr * abs_arr, threshold * threshold)
    degrees = abs_arr > threshold
    return float(np.mean(bias + 2 * sigma2 * degrees))


def _estimate_sigma(values: torch.Tensor) -> float:
    arr = values.detach().cpu().numpy().reshape(-1)
    if arr.size == 0:
        return EPS

    centered = arr - np.median(arr)
    mad = float(np.median(np.abs(centered)))
    if mad > EPS:
        return mad / NORMAL_MAD_SCALE

    std = float(np.std(arr))
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
