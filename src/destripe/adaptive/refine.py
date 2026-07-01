import numpy as np
import torch

from . import preprocess, stripe
from .constants import EPS


def refine_clean(
    *,
    gray: np.ndarray,
    clean: np.ndarray,
    directions: tuple[int, ...],
    proj: bool,
) -> np.ndarray:
    refined = np.asarray(clean, dtype=np.float64).copy()
    if not directions or refined.shape != np.shape(gray):
        return refined

    for mode in directions:
        high_pass = preprocess.extract_high_pass(
            torch.as_tensor(refined, dtype=torch.float32)
        )
        candidate = stripe.project(high_pass, mode)
        alpha = stripe.measure_shrinkage(high_pass, mode)

        candidate = candidate.cpu().numpy().astype(np.float64)
        candidate -= float(candidate.mean())
        if alpha <= EPS or float(np.mean(np.abs(candidate))) <= EPS:
            continue

        refined = refined - alpha * candidate

    if proj:
        refined = np.clip(refined, 0.0, 1.0)
    return refined
