import numpy as np
import torch

from .constants import EPS
from .analysis import extract_high_pass
from .profiles import (
    MIN_PROFILE_SIGN_CHANGES,
    measure_repetition,
    measure_shrinkage,
    project,
)


# A single edge contributes only one significant profile sign change.  Refinement
# uses a lower bar than initial detection because its directions were already
# approved globally, but it must still exclude that single-edge case.
MIN_REFINEMENT_REPETITION = 1.0 / MIN_PROFILE_SIGN_CHANGES


def refine_clean(
    *,
    gray: np.ndarray,
    clean: np.ndarray,
    directions: tuple[int, ...],
    proj: bool,
) -> np.ndarray:
    image = np.asarray(gray, dtype=np.float64)
    refined = np.asarray(clean, dtype=np.float64).copy()
    if not directions or refined.shape != image.shape:
        return refined

    for mode in directions:
        high_pass = extract_high_pass(torch.as_tensor(refined, dtype=torch.float32))
        if measure_repetition(high_pass, mode) <= MIN_REFINEMENT_REPETITION:
            continue

        candidate = project(high_pass, mode)
        alpha = measure_shrinkage(high_pass, mode)

        candidate = candidate.cpu().numpy().astype(np.float64)
        candidate -= float(candidate.mean())
        if alpha <= EPS or float(np.mean(np.abs(candidate))) <= EPS:
            continue

        residual_high_pass = extract_high_pass(
            torch.as_tensor(image - refined, dtype=torch.float32)
        )
        residual_profile = project(residual_high_pass, mode)
        residual_profile = residual_profile.cpu().numpy().astype(np.float64)
        residual_profile -= float(residual_profile.mean())
        residual_support = float(np.mean(candidate * residual_profile))
        candidate_energy = float(np.mean(candidate * candidate))
        if residual_support <= EPS or candidate_energy <= EPS:
            continue

        alpha = min(alpha, residual_support / candidate_energy)

        refined = refined - alpha * candidate

    if proj:
        refined = np.clip(refined, 0.0, 1.0)
    return refined
