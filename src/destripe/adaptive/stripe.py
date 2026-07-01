import math

import torch

from .constants import EPS, PARALLEL_OFFSETS


def project(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    line_ids = _make_line_ids(tensor.shape, mode, tensor.device).reshape(-1)
    values = tensor.reshape(-1)

    unique, inverse = torch.unique(line_ids, sorted=True, return_inverse=True)
    sums = torch.zeros(
        unique.numel(),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, inverse, values)
    counts.scatter_add_(0, inverse, torch.ones_like(values))
    return (sums / counts.clamp(min=1.0))[inverse].reshape(tensor.shape)


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
    full = 0.5 * (first + second)
    variance = torch.mean(full * full)
    if float(variance.item()) <= EPS:
        return 0.0
    covariance = torch.mean(first * second)
    return float((covariance / variance).clamp(min=0.0, max=1.0).item())


def measure_parallel(tensor: torch.Tensor, mode: int) -> float:
    step = PARALLEL_OFFSETS[mode]
    diff = _offset_diff(tensor, step=step).abs()
    return float((diff / math.hypot(*step)).mean().item())


def measure_tv(tensor: torch.Tensor) -> float:
    row = torch.zeros_like(tensor)
    col = torch.zeros_like(tensor)
    row[:-1, :] = tensor[1:, :] - tensor[:-1, :]
    col[:, :-1] = tensor[:, 1:] - tensor[:, :-1]
    return float(torch.sqrt(row * row + col * col).mean().item())


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


def _offset_diff(
    tensor: torch.Tensor,
    *,
    step: tuple[int, int],
) -> torch.Tensor:
    row_step, col_step = step
    row_start_a = max(0, -row_step)
    row_start_b = max(0, row_step)
    col_start_a = max(0, -col_step)
    col_start_b = max(0, col_step)
    rows = tensor.shape[0] - abs(row_step)
    cols = tensor.shape[1] - abs(col_step)
    if rows <= 0 or cols <= 0:
        return torch.zeros(1, dtype=tensor.dtype, device=tensor.device)
    a = tensor[
        row_start_a : row_start_a + rows,
        col_start_a : col_start_a + cols,
    ]
    b = tensor[
        row_start_b : row_start_b + rows,
        col_start_b : col_start_b + cols,
    ]
    return b - a
