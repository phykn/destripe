import math

import numpy as np

from . import constants


def estimate_strength(
    *,
    score_weights: np.ndarray,
    selection_weights: np.ndarray,
) -> tuple[float, float, float]:
    score_strength = _measure_concentration(score_weights)
    selection_strength = _measure_concentration(selection_weights)
    overall_strength = math.sqrt(score_strength * selection_strength)
    ambiguity = _measure_entropy(score_weights)
    mu2_strength = overall_strength * ambiguity

    mu1 = float(
        constants.MU1_MIN * (1.0 - overall_strength)
        + constants.MU1_MAX * overall_strength
    )
    log_mu2 = (
        math.log(constants.MU2_MIN) * (1.0 - mu2_strength)
        + math.log(constants.MU2_MAX) * mu2_strength
    )
    confidence = overall_strength * (1.0 - ambiguity)
    return mu1, float(math.exp(log_mu2)), confidence


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
