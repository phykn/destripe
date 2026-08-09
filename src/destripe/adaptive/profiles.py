import torch

from .constants import EPS, PARALLEL_OFFSETS


PROFILE_BIN_COUNT = 8
PROFILE_SIGNIFICANCE_RMS = 0.25
MIN_PROFILE_SIGN_CHANGES = 3
MIN_PROFILE_ENERGY_PARTICIPATION = 2.5
MIN_PROFILE_REPETITION = 1.0


def project(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    profile, inverse = make_profile(tensor, mode)
    return profile[inverse].reshape(tensor.shape)


def measure_repetition(tensor: torch.Tensor, mode: int) -> float:
    profile, _ = make_profile(tensor, mode)
    return measure_profile_repetition(profile)


def measure_profile_repetition(profile: torch.Tensor) -> float:
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


def measure_shrinkage(tensor: torch.Tensor, mode: int) -> float:
    inverse, line_count = _make_line_index(tensor.shape, mode, tensor.device)
    split_ids = _make_split_ids(tensor.shape, mode, tensor.device).reshape(-1)
    values = tensor.reshape(-1)

    profiles = []
    masks = []
    for split in (0, 1):
        selected = split_ids.remainder(2) == split
        sums = torch.zeros(line_count, dtype=tensor.dtype, device=tensor.device)
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


def make_profile(
    tensor: torch.Tensor,
    mode: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    inverse, line_count = _make_line_index(tensor.shape, mode, tensor.device)
    values = tensor.reshape(-1)
    sums = torch.zeros(line_count, dtype=tensor.dtype, device=tensor.device)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, inverse, values)
    counts.scatter_add_(0, inverse, torch.ones_like(values))
    return sums / counts.clamp(min=1.0), inverse


def _make_line_index(
    shape: torch.Size | tuple[int, int],
    mode: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    line_ids = _make_line_ids(shape, mode, device).reshape(-1)
    if min(shape) < 2:
        _, inverse = torch.unique(line_ids, sorted=True, return_inverse=True)
        return inverse, int(inverse.max().item()) + 1

    inverse = line_ids - line_ids.min()
    return inverse, int(inverse.max().item()) + 1


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
