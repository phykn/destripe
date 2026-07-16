import torch

from .constants import EPS, PARALLEL_OFFSETS


PROFILE_BIN_COUNT = 8
PROFILE_SIGNIFICANCE_RMS = 0.25
MIN_PROFILE_SIGN_CHANGES = 3
MIN_PROFILE_ENERGY_PARTICIPATION = 2.5
MIN_PROFILE_REPETITION = 1.0
MIN_PROFILE_DISTRIBUTED_PARTICIPATION = 0.15
MIN_SPARSE_RESIDUAL_CLOSURE = 0.5
SPARSE_EDGE_MAD_MULTIPLIER = 8.0
MIN_REPEATED_EDGE_MAGNITUDE_RATIO = 0.75
MIN_REPEATED_EDGE_PARTICIPATION = 0.15
MAX_REPEATED_EDGE_SPACING_CV = 0.25
MIN_REPEATED_EDGE_COVERAGE = 0.7
MIN_SAWTOOTH_RAMP_PARTICIPATION = 0.8
MAX_SAWTOOTH_RAMP_CLOSURE = 0.1


def project(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    profile, inverse = _make_profile(tensor, mode)
    return profile[inverse].reshape(tensor.shape)


def measure_repetition(tensor: torch.Tensor, mode: int) -> float:
    profile, _ = _make_profile(tensor, mode)
    return _measure_profile_repetition(profile)


def measure_distributed_repetition(tensor: torch.Tensor, mode: int) -> float:
    profile, _ = _make_profile(tensor, mode)
    residual_profile, _, _ = _split_sparse_profile(profile)
    return _measure_profile_repetition(residual_profile)


def _measure_profile_repetition(profile: torch.Tensor) -> float:
    if profile.numel() < MIN_PROFILE_SIGN_CHANGES + 1:
        return 0.0

    centered = profile - profile.mean()
    rms = torch.sqrt(torch.mean(centered * centered))
    if float(rms.item()) <= EPS:
        return 0.0

    significant = centered[centered.abs() >= PROFILE_SIGNIFICANCE_RMS * rms]
    if significant.numel() < 2:
        sign_changes = 0
    else:
        sign_changes = int((significant[1:] * significant[:-1] < 0).sum().item())

    bin_count = min(PROFILE_BIN_COUNT, centered.numel())
    bin_ids = torch.div(
        torch.arange(centered.numel(), device=profile.device) * bin_count,
        centered.numel(),
        rounding_mode="floor",
    )
    bin_energy = torch.zeros(bin_count, dtype=profile.dtype, device=profile.device)
    bin_energy.scatter_add_(0, bin_ids, centered * centered)
    energy = bin_energy.sum()
    if float(energy.item()) <= EPS:
        return 0.0
    energy_shares = bin_energy / energy
    participation = float((1 / torch.sum(energy_shares * energy_shares)).item())

    sign_score = min(1.0, sign_changes / MIN_PROFILE_SIGN_CHANGES)
    participation_score = min(
        1.0,
        participation / MIN_PROFILE_ENERGY_PARTICIPATION,
    )
    return min(sign_score, participation_score)


def measure_profile_distribution(tensor: torch.Tensor, mode: int) -> float:
    profile, _ = _make_profile(tensor, mode)
    _, sparse_edges, repeated_edges = _split_sparse_profile(profile)
    if repeated_edges:
        return 1.0

    residual = torch.diff(profile).masked_fill(sparse_edges, 0.0)
    absolute_sum = residual.abs().sum()
    squared_sum = torch.sum(residual * residual)
    if float(squared_sum.item()) <= EPS:
        return 0.0

    participation = absolute_sum.square() / (residual.numel() * squared_sum + EPS)
    if bool(sparse_edges.any()):
        closure = 1.0 - residual.sum().abs() / (absolute_sum + EPS)
        if float(closure.item()) < MIN_SPARSE_RESIDUAL_CLOSURE:
            return 0.0
    return float(participation.clamp(min=0.0, max=1.0).item())


def find_sparse_profile_edges(profile: torch.Tensor) -> torch.Tensor:
    _, sparse_edges, _ = _split_sparse_profile(profile)
    return sparse_edges


def extract_sparse_profile_structure(
    tensor: torch.Tensor,
    mode: int,
) -> torch.Tensor:
    profile, inverse = _make_profile(tensor, mode)
    sparse_edges = find_sparse_profile_edges(profile)
    if not bool(sparse_edges.any()):
        return torch.zeros_like(tensor)

    structure_deltas = torch.zeros_like(profile[:-1])
    source_deltas = torch.diff(profile)
    structure_deltas[sparse_edges] = source_deltas[sparse_edges]
    structure_profile = torch.cat(
        (
            torch.zeros(1, dtype=tensor.dtype, device=tensor.device),
            torch.cumsum(structure_deltas, dim=0),
        )
    )
    counts = torch.bincount(inverse, minlength=profile.numel()).to(tensor.dtype)
    weighted_mean = torch.sum(structure_profile * counts) / counts.sum().clamp(min=1.0)
    structure_profile -= weighted_mean
    return structure_profile[inverse].reshape(tensor.shape)


def _split_sparse_profile(
    profile: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    deltas = torch.diff(profile)
    if deltas.numel() == 0:
        empty = torch.zeros_like(deltas, dtype=torch.bool)
        return profile, empty, False

    median = torch.median(deltas)
    deviations = (deltas - median).abs()
    maximum = float(deviations.max().item())
    if maximum <= EPS:
        empty = torch.zeros_like(deltas, dtype=torch.bool)
        return profile, empty, False

    mad = float(torch.median(deviations).item())
    tolerance = max(EPS, maximum * 1e-6, abs(float(median.item())) * 2e-3)
    threshold = SPARSE_EDGE_MAD_MULTIPLIER * mad if mad > tolerance else tolerance
    sparse_edges = deviations > threshold
    if _has_repeated_sparse_edges(deltas, sparse_edges):
        empty = torch.zeros_like(sparse_edges)
        return profile, empty, True

    residual_deltas = deltas.masked_fill(sparse_edges, 0.0)
    if _has_distributed_sawtooth_ramp(deltas, sparse_edges):
        empty = torch.zeros_like(sparse_edges)
        return profile, empty, True
    residual_profile = torch.cat(
        (
            torch.zeros(1, dtype=profile.dtype, device=profile.device),
            torch.cumsum(residual_deltas, dim=0),
        )
    )
    return residual_profile, sparse_edges, False


def _has_repeated_sparse_edges(
    deltas: torch.Tensor,
    sparse_edges: torch.Tensor,
) -> bool:
    indices = torch.nonzero(sparse_edges, as_tuple=False).flatten()
    if indices.numel() < 2:
        return False

    magnitudes = deltas[sparse_edges].abs()
    maximum = float(magnitudes.max().item())
    if maximum <= EPS:
        return False
    magnitude_ratio = float((magnitudes.min() / maximum).item())
    if magnitude_ratio < MIN_REPEATED_EDGE_MAGNITUDE_RATIO:
        return False
    participation = magnitudes.sum().square() / (
        deltas.numel() * torch.sum(magnitudes * magnitudes) + EPS
    )
    # Sparse low-frequency square profiles are indistinguishable from scene bars.
    # Only a dense, full-profile edge train is safe to classify automatically.
    if float(participation.item()) < MIN_REPEATED_EDGE_PARTICIPATION:
        return False

    gaps = torch.diff(indices.to(torch.float32))
    mean_gap = gaps.mean().clamp(min=1.0)
    covered_span = indices[-1] - indices[0] + mean_gap
    coverage = float((covered_span / deltas.numel()).item())
    if coverage < MIN_REPEATED_EDGE_COVERAGE:
        return False
    if gaps.numel() <= 1:
        return bool((deltas[sparse_edges][0] * deltas[sparse_edges][1] < 0).item())
    spacing_cv = float((gaps.std(unbiased=False) / mean_gap).item())
    if spacing_cv > MAX_REPEATED_EDGE_SPACING_CV:
        return False

    edge_values = deltas[sparse_edges]
    return bool(torch.all(edge_values[1:] * edge_values[:-1] < 0).item())


def _has_distributed_sawtooth_ramp(
    deltas: torch.Tensor,
    sparse_edges: torch.Tensor,
) -> bool:
    if not bool(sparse_edges.any()):
        return False

    residual = deltas[~sparse_edges]
    if residual.numel() < MIN_PROFILE_SIGN_CHANGES:
        return False
    absolute_sum = residual.abs().sum()
    squared_sum = torch.sum(residual * residual)
    if float(squared_sum.item()) <= EPS:
        return False
    # Same-sign jumps alone describe a staircase. Sawtooth resets must oppose a
    # distributed ramp between those jumps.
    if float((deltas[sparse_edges].sum() * residual.sum()).item()) >= -EPS:
        return False
    participation = absolute_sum.square() / (residual.numel() * squared_sum + EPS)
    closure = 1.0 - residual.sum().abs() / (absolute_sum + EPS)
    return (
        float(participation.item()) >= MIN_SAWTOOTH_RAMP_PARTICIPATION
        and float(closure.item()) <= MAX_SAWTOOTH_RAMP_CLOSURE
    )


def measure_shrinkage(tensor: torch.Tensor, mode: int) -> float:
    line_ids = _make_line_ids(tensor.shape, mode, tensor.device).reshape(-1)
    split_ids = _make_split_ids(tensor.shape, mode, tensor.device).reshape(-1)
    values = tensor.reshape(-1)
    unique, inverse = torch.unique(line_ids, sorted=True, return_inverse=True)

    profiles = []
    masks = []
    for split in (0, 1):
        selected = split_ids.remainder(2) == split
        sums = torch.zeros(unique.numel(), dtype=tensor.dtype, device=tensor.device)
        counts = torch.zeros_like(sums)
        sums.scatter_add_(0, inverse[selected], values[selected])
        counts.scatter_add_(0, inverse[selected], torch.ones_like(values[selected]))
        profiles.append(sums / counts.clamp(min=1.0))
        masks.append(counts > 0)

    usable = masks[0] & masks[1]
    if int(usable.sum().item()) < 2:
        return 0.0

    first = profiles[0][usable]
    second = profiles[1][usable]
    first = first - first.mean()
    second = second - second.mean()
    full = (first + second) / 2
    variance = torch.mean(full * full)
    if float(variance.item()) <= EPS:
        return 0.0
    covariance = torch.mean(first * second)
    return float((covariance / variance).clamp(min=0.0, max=1.0).item())


def _make_profile(
    tensor: torch.Tensor,
    mode: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    line_ids = _make_line_ids(tensor.shape, mode, tensor.device).reshape(-1)
    values = tensor.reshape(-1)
    unique, inverse = torch.unique(line_ids, sorted=True, return_inverse=True)
    sums = torch.zeros(unique.numel(), dtype=tensor.dtype, device=tensor.device)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, inverse, values)
    counts.scatter_add_(0, inverse, torch.ones_like(values))
    return sums / counts.clamp(min=1.0), inverse


def _make_line_ids(
    shape: torch.Size | tuple[int, int],
    mode: int,
    device: torch.device,
) -> torch.Tensor:
    row_step, col_step = PARALLEL_OFFSETS[mode]
    height, width = shape
    rows = torch.arange(height, device=device)[:, None]
    cols = torch.arange(width, device=device)[None, :]
    return col_step * rows - row_step * cols


def _make_split_ids(
    shape: torch.Size | tuple[int, int],
    mode: int,
    device: torch.device,
) -> torch.Tensor:
    row_step, _ = PARALLEL_OFFSETS[mode]
    height, width = shape
    rows = torch.arange(height, device=device)[:, None]
    cols = torch.arange(width, device=device)[None, :]
    if row_step % 2:
        return rows.expand(height, width)
    return cols.expand(height, width)
