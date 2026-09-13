from dataclasses import dataclass

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


@dataclass(frozen=True)
class RefineResult:
    clean: np.ndarray
    limited: bool


def refine_clean(
    *,
    gray: np.ndarray,
    clean: np.ndarray,
    directions: tuple[int, ...],
    proj: bool,
    passes: int = 1,
) -> RefineResult:
    image = np.asarray(gray, dtype=np.float64)
    refined = np.asarray(clean, dtype=np.float64).copy()
    if not directions or refined.shape != image.shape:
        return RefineResult(clean=refined, limited=False)

    # Only the solver's initial correction is evidence. Refinement must not
    # increase its own support or replenish its budget on subsequent passes.
    residual = image - refined
    remaining = float(np.sqrt(np.mean(residual * residual)))
    residual_high_pass = extract_high_pass(
        torch.as_tensor(residual, dtype=torch.float32)
    )
    profiles = {
        mode: _project_centered(residual_high_pass, mode) for mode in directions
    }
    budgets = {
        mode: float(np.sqrt(np.mean(profile * profile)))
        for mode, profile in profiles.items()
    }

    limited = False
    for _ in range(passes):
        changed = False
        for mode in directions:
            budget = min(remaining, budgets[mode])
            high_pass = extract_high_pass(torch.as_tensor(refined, dtype=torch.float32))
            if measure_repetition(high_pass, mode) <= MIN_REFINEMENT_REPETITION:
                continue

            candidate = _project_centered(high_pass, mode)
            alpha = measure_shrinkage(high_pass, mode)
            residual_support = float(np.mean(candidate * profiles[mode]))
            candidate_energy = float(np.mean(candidate * candidate))
            if alpha <= EPS or residual_support <= EPS or candidate_energy <= EPS:
                continue
            if budget <= EPS:
                limited = True
                continue

            candidate_rms = float(np.sqrt(candidate_energy))
            alpha = min(alpha, residual_support / candidate_energy)
            if alpha * candidate_rms > budget:
                alpha = budget / candidate_rms
                limited = True
            refined -= alpha * candidate
            spent = alpha * candidate_rms
            budgets[mode] -= spent
            remaining -= spent
            changed = True
        if not changed:
            break

    if proj:
        refined = np.clip(refined, 0.0, 1.0)
    return RefineResult(clean=refined, limited=limited)


def _project_centered(tensor: torch.Tensor, mode: int) -> np.ndarray:
    profile = project(tensor, mode).cpu().numpy().astype(np.float64)
    profile -= float(profile.mean())
    return profile
