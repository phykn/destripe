from dataclasses import dataclass
import math

import cv2
import numpy as np
import torch

from .constants import EPS, NORMAL_MAD_SCALE, PARALLEL_OFFSETS
from .preprocess import extract_high_pass
from .stripe import measure_shrinkage, project_robust


@dataclass(frozen=True)
class DirectionEvidence:
    protection: torch.Tensor
    reliability: float
    input_profile: torch.Tensor


@dataclass(frozen=True)
class SafetyResult:
    clean: np.ndarray
    alphas: tuple[float, ...]


def make_direction_evidence(
    gray: np.ndarray,
    directions: tuple[int, ...],
) -> dict[int, DirectionEvidence]:
    image = np.asarray(gray, dtype=np.float32)
    tensor = torch.as_tensor(image, dtype=torch.float32)
    high_pass = extract_high_pass(tensor)
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    blurred_high_pass = extract_high_pass(torch.as_tensor(blurred))

    evidence = {}
    for mode in directions:
        protection = _make_protection(high_pass, mode)
        safe = 1.0 - protection
        input_profile = project_robust(high_pass, mode, safe)
        blurred_profile = project_robust(blurred_high_pass, mode, safe)
        scale_repeatability = _positive_centered_cosine(
            input_profile,
            blurred_profile,
        )
        split_repeatability = measure_shrinkage(
            high_pass,
            mode,
            weights=safe,
        )
        reliability = math.sqrt(split_repeatability * scale_repeatability)
        evidence[mode] = DirectionEvidence(
            protection=protection,
            reliability=float(np.clip(np.nan_to_num(reliability), 0.0, 1.0)),
            input_profile=torch.nan_to_num(input_profile),
        )
    return evidence


def choose_alpha(
    *,
    input_profile: torch.Tensor,
    proposal_profile: torch.Tensor,
    reliability: float,
    leakage: float,
) -> float:
    normalizer = float(torch.sum(input_profile.square()).item()) + EPS
    a_value = float(torch.sum(proposal_profile.square()).item()) / normalizer
    a_value += max(0.0, float(leakage))
    b_value = float(reliability) * float(
        torch.sum(input_profile * proposal_profile).item()
    ) / normalizer
    return float(np.clip(b_value / (a_value + EPS), 0.0, 1.0))


def select_clean(
    *,
    gray: np.ndarray,
    solver_clean: np.ndarray,
    components: tuple[np.ndarray, ...],
    directions: tuple[int, ...],
    proj: bool,
) -> SafetyResult:
    image = np.asarray(gray, dtype=np.float64)
    solver = np.asarray(solver_clean, dtype=np.float64)
    component_arrays = _validate_components(
        components=components,
        directions=directions,
        shape=image.shape,
    )
    if image.ndim != 2 or solver.shape != image.shape:
        raise ValueError("solver_clean and components must match gray shape.")

    evidence_by_mode = make_direction_evidence(image, directions)
    clean_high_pass = extract_high_pass(torch.as_tensor(solver, dtype=torch.float32))
    accepted_correction = np.zeros_like(image, dtype=np.float64)
    alphas = []

    for mode, component in zip(directions, component_arrays):
        evidence = evidence_by_mode[mode]
        safe = 1.0 - evidence.protection
        residual = project_robust(clean_high_pass, mode, safe)
        proposal = component + residual.numpy()
        proposal_tensor = torch.as_tensor(proposal, dtype=torch.float32)
        proposal_high_pass = extract_high_pass(proposal_tensor)
        proposal_profile = project_robust(proposal_high_pass, mode, safe)
        curvature = _second_parallel_diff(proposal_tensor, mode)
        proposal_power = float(torch.sum(proposal_profile.square()).item())
        leakage = float(
            torch.sum((evidence.protection * curvature).square()).item()
        )
        alpha = choose_alpha(
            input_profile=evidence.input_profile,
            proposal_profile=proposal_profile,
            reliability=evidence.reliability,
            leakage=leakage / (proposal_power + EPS),
        )
        accepted_correction += alpha * proposal
        alphas.append(alpha)

    clean = image - accepted_correction
    if proj:
        clean = np.clip(clean, 0.0, 1.0)
    return SafetyResult(clean=clean, alphas=tuple(alphas))


def _validate_components(
    *,
    components: tuple[np.ndarray, ...],
    directions: tuple[int, ...],
    shape: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    if len(components) != len(directions):
        raise ValueError("components must match directions.")

    arrays = []
    for component in components:
        try:
            array = np.asarray(component, dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError("components must contain finite arrays.") from None
        if array.shape != shape:
            raise ValueError("components must match gray shape.")
        if not np.isfinite(array).all():
            raise ValueError("components must contain only finite values.")
        arrays.append(array)
    return tuple(arrays)


def _make_protection(high_pass: torch.Tensor, mode: int) -> torch.Tensor:
    activity = _parallel_activity(high_pass, mode)
    median = torch.median(activity)
    mad = torch.median((activity - median).abs()) / NORMAL_MAD_SCALE
    scale = mad
    if not math.isfinite(float(scale.item())) or float(scale.item()) <= EPS:
        scale = activity.std(unbiased=False)
    normalized = ((activity - median) / (3.0 * scale + EPS)).clamp(0.0, 1.0)

    array = normalized.numpy()
    array = cv2.dilate(array, np.ones((3, 3), dtype=np.uint8))
    array = cv2.GaussianBlur(array, (5, 5), sigmaX=0)
    return torch.from_numpy(np.nan_to_num(array)).clamp(0.0, 1.0)


def _parallel_activity(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    return _parallel_diff(tensor, mode).abs()


def _parallel_diff(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    row_step, col_step = PARALLEL_OFFSETS[mode]
    source_rows, target_rows = _valid_slices(tensor.shape[0], row_step)
    source_cols, target_cols = _valid_slices(tensor.shape[1], col_step)
    difference = torch.zeros_like(tensor)
    difference[target_rows, target_cols] = (
        tensor[target_rows, target_cols] - tensor[source_rows, source_cols]
    )
    return difference


def _second_parallel_diff(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    row_step, col_step = PARALLEL_OFFSETS[mode]
    source_rows, middle_rows, target_rows = _second_valid_slices(
        tensor.shape[0], row_step
    )
    source_cols, middle_cols, target_cols = _second_valid_slices(
        tensor.shape[1], col_step
    )
    difference = torch.zeros_like(tensor)
    difference[target_rows, target_cols] = (
        tensor[target_rows, target_cols]
        - 2.0 * tensor[middle_rows, middle_cols]
        + tensor[source_rows, source_cols]
    )
    return difference


def _valid_slices(size: int, step: int) -> tuple[slice, slice]:
    if step >= 0:
        span = max(0, size - step)
        start = min(step, size)
        return slice(0, span), slice(start, start + span)
    span = max(0, size + step)
    start = min(-step, size)
    return slice(start, start + span), slice(0, span)


def _second_valid_slices(size: int, step: int) -> tuple[slice, slice, slice]:
    span = max(0, size - 2 * abs(step))
    if step >= 0:
        middle = min(step, size)
        target = min(2 * step, size)
        return (
            slice(0, span),
            slice(middle, middle + span),
            slice(target, target + span),
        )
    source = min(-2 * step, size)
    middle = min(-step, size)
    return (
        slice(source, source + span),
        slice(middle, middle + span),
        slice(0, span),
    )


def _positive_centered_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    centered_first = first - first.mean()
    centered_second = second - second.mean()
    denominator = torch.linalg.vector_norm(centered_first) * torch.linalg.vector_norm(
        centered_second
    )
    if not math.isfinite(float(denominator.item())) or float(denominator.item()) <= EPS:
        return 0.0
    correlation = torch.sum(centered_first * centered_second) / denominator
    return float(torch.nan_to_num(correlation).clamp(0.0, 1.0).item())
