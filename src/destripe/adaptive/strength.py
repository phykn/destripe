import math

import numpy as np

from . import constants


def estimate_mu_and_confidence(
    *,
    score_weights: np.ndarray,
    selection_weights: np.ndarray,
) -> tuple[float, float, float]:
    score_strength = _distribution_concentration(score_weights)
    selection_strength = _distribution_concentration(selection_weights)
    strength = math.sqrt(score_strength * selection_strength)
    ambiguity = _distribution_entropy(score_weights)

    mu1 = float(
        constants.MU1_MIN * (1.0 - strength) + constants.MU1_MAX * strength
    )
    log_mu2 = (
        math.log(constants.MU2_MIN) * (1.0 - strength * ambiguity)
        + math.log(constants.MU2_MAX) * strength * ambiguity
    )
    confidence = strength * (1.0 - ambiguity)
    return mu1, float(math.exp(log_mu2)), confidence


def _distribution_concentration(weights: np.ndarray) -> float:
    uniform_power = 1.0 / len(weights)
    power = float(np.sum(weights * weights))
    return min(1.0, max(0.0, (power - uniform_power) / (1.0 - uniform_power)))


def _distribution_entropy(weights: np.ndarray) -> float:
    positive = weights[weights > 0.0]
    if positive.size <= 1:
        return 0.0
    entropy = -float(np.sum(positive * np.log(positive)))
    return min(1.0, max(0.0, entropy / math.log(len(weights))))
