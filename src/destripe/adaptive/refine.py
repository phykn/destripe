import numpy as np

from .safety import select_clean


def refine_clean(
    *,
    gray: np.ndarray,
    clean: np.ndarray,
    components: tuple[np.ndarray, ...],
    directions: tuple[int, ...],
    proj: bool,
) -> np.ndarray:
    return select_clean(
        gray=gray,
        solver_clean=clean,
        components=components,
        directions=directions,
        proj=proj,
    ).clean
