from dataclasses import dataclass
import math
import time

import numpy as np

from .core import UniversalStripeRemover


MU1_CANDIDATES = (1 / 6, 1 / 5, 1 / 4, 1 / 3)
MU2_CANDIDATES = tuple(1 / value for value in (300, 240, 180, 120, 100, 90, 60))

_NORMAL_MAD_SCALE = 0.6744897501960817
_EPS = 1e-9
_MAX_ITERATIONS = 500
_CONVERGENCE_TOLERANCE = 1e-5


@dataclass(frozen=True)
class ParameterCandidate:
    mu1: float
    mu2: float


@dataclass(frozen=True)
class HybridDiagnostics:
    candidate_count: int
    mu1: float | None
    mu2: float | None
    beta: float
    iterations: int
    solver_seconds: float


@dataclass(frozen=True)
class HybridResult:
    clean: np.ndarray
    diagnostics: HybridDiagnostics


@dataclass(frozen=True)
class _CandidateEvaluation:
    candidate: ParameterCandidate
    correction: np.ndarray
    beta: float
    explained_fraction: float
    protected_energy: float
    total_energy: float
    iterations: int


def _parameter_candidates(
    gray: np.ndarray,
    *,
    direction: int,
    target: np.ndarray,
) -> tuple[ParameterCandidate, ...]:
    gray_array = np.asarray(gray, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if gray_array.shape != target_array.shape or gray_array.ndim != 2:
        raise ValueError("gray and target must have the same two-dimensional shape.")
    if not np.isfinite(gray_array).all() or not np.isfinite(target_array).all():
        raise ValueError("gray and target must contain only finite values.")
    if isinstance(direction, bool) or direction not in range(5):
        raise ValueError("direction must be an integer from 0 through 4.")

    strength = _robust_target_strength(target_array)
    center_index = min(
        range(len(MU2_CANDIDATES)),
        key=lambda index: abs(
            math.log(max(strength, _EPS))
            - math.log(MU2_CANDIDATES[index])
        ),
    )
    selected_indices = sorted(
        {
            max(0, center_index - 1),
            center_index,
            min(len(MU2_CANDIDATES) - 1, center_index + 1),
        }
    )
    selected_mu2 = tuple(MU2_CANDIDATES[index] for index in selected_indices)
    return tuple(
        ParameterCandidate(mu1=mu1, mu2=mu2)
        for mu1 in MU1_CANDIDATES
        for mu2 in selected_mu2
    )


def _robust_target_strength(target: np.ndarray) -> float:
    values = np.asarray(target, dtype=np.float64).reshape(-1)
    centered = values - float(np.median(values))
    mad = float(np.median(np.abs(centered)))
    if mad > _EPS:
        return mad / _NORMAL_MAD_SCALE
    standard_deviation = float(np.std(centered))
    if standard_deviation > _EPS:
        return standard_deviation
    return MU2_CANDIDATES[0]


def _evaluate_candidate(
    *,
    correction: np.ndarray,
    target: np.ndarray,
    protection: np.ndarray,
    candidate: ParameterCandidate,
    iterations: int,
    confidence: float = 1.0,
) -> _CandidateEvaluation | None:
    correction_array = np.asarray(correction, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    protection_array = np.asarray(protection, dtype=np.float64)
    if (
        correction_array.shape != target_array.shape
        or protection_array.shape != target_array.shape
        or not np.isfinite(correction_array).all()
    ):
        return None

    target_power = float(np.sum(target_array * target_array))
    candidate_projection = float(np.sum(correction_array * target_array))
    if (
        not math.isfinite(target_power)
        or target_power <= _EPS
        or not math.isfinite(candidate_projection)
        or candidate_projection <= _EPS
    ):
        return None

    beta = float(np.clip(target_power / candidate_projection, 0.0, 1.0))
    correction_power = float(np.sum(correction_array * correction_array))
    if not math.isfinite(correction_power) or correction_power <= _EPS:
        return None
    confidence_scale = math.sqrt(float(np.clip(confidence, 0.0, 1.0)))
    beta = min(beta, math.sqrt(target_power / correction_power)) * confidence_scale
    scaled = beta * correction_array
    explained = float(np.sum(scaled * target_array) / target_power)
    explained = float(np.clip(explained, 0.0, 1.0))
    protected_energy = float(np.mean((scaled * protection_array) ** 2))
    total_energy = float(np.mean(scaled * scaled))
    return _CandidateEvaluation(
        candidate=candidate,
        correction=scaled,
        beta=beta,
        explained_fraction=explained,
        protected_energy=protected_energy,
        total_energy=total_energy,
        iterations=iterations,
    )


def _select_candidate(
    candidates: tuple[_CandidateEvaluation, ...],
) -> _CandidateEvaluation | None:
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            -item.explained_fraction,
            item.protected_energy,
            item.total_energy,
            item.candidate.mu1,
            item.candidate.mu2,
        ),
    )


def _run_hybrid(
    gray: np.ndarray,
    *,
    direction: int,
    target: np.ndarray,
    protection: np.ndarray,
    reliability: float,
    consistent: bool,
    proj: bool,
) -> HybridResult:
    gray_array = np.asarray(gray, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    protection_array = np.asarray(protection, dtype=np.float64)
    if not consistent or float(np.sum(target_array * target_array)) <= _EPS:
        return _noop_result(gray_array)

    parameters = _parameter_candidates(
        gray_array,
        direction=direction,
        target=target_array,
    )
    evaluated: list[_CandidateEvaluation] = []
    solver_started = time.perf_counter()
    for parameter in parameters:
        remover = UniversalStripeRemover(
            mu1=parameter.mu1,
            mu2=parameter.mu2,
            directions=[direction],
        )
        try:
            solve_result = remover._process_with_info(
                gray_array,
                iterations=_MAX_ITERATIONS,
                tol=_CONVERGENCE_TOLERANCE,
                proj=proj,
            )
        except (RuntimeError, FloatingPointError):
            continue
        clean = solve_result.clean.detach().cpu().numpy().astype(np.float64, copy=False)
        candidate = _evaluate_candidate(
            correction=gray_array - clean,
            target=target_array,
            protection=protection_array,
            candidate=parameter,
            iterations=solve_result.iterations,
            confidence=reliability,
        )
        if candidate is not None:
            evaluated.append(candidate)
    solver_seconds = time.perf_counter() - solver_started

    selected = _select_candidate(tuple(evaluated))
    if selected is None:
        return _noop_result(
            gray_array,
            candidate_count=len(parameters),
            solver_seconds=solver_seconds,
        )

    clean = gray_array - selected.correction
    if proj:
        clean = np.clip(clean, 0.0, 1.0)
    return HybridResult(
        clean=clean,
        diagnostics=HybridDiagnostics(
            candidate_count=len(parameters),
            mu1=selected.candidate.mu1,
            mu2=selected.candidate.mu2,
            beta=selected.beta,
            iterations=selected.iterations,
            solver_seconds=solver_seconds,
        ),
    )


def _noop_result(
    gray: np.ndarray,
    *,
    candidate_count: int = 0,
    solver_seconds: float = 0.0,
) -> HybridResult:
    return HybridResult(
        clean=np.asarray(gray, dtype=np.float64).copy(),
        diagnostics=HybridDiagnostics(
            candidate_count=candidate_count,
            mu1=None,
            mu2=None,
            beta=0.0,
            iterations=0,
            solver_seconds=solver_seconds,
        ),
    )
