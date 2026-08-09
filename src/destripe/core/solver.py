from dataclasses import dataclass

import torch

from .operators import adjoint_dir, adjoint_grad, dir_diff, forward_diff


@dataclass(frozen=True)
class SolveResult:
    clean: torch.Tensor
    iterations: int


def solve_pdhg(
    *,
    data: torch.Tensor,
    directions: tuple[int, ...],
    mu1: torch.Tensor | float,
    mu2: torch.Tensor | float,
    device: torch.device | str,
    tau: float,
    sigma: float,
    iterations: int,
    proj: bool,
    verbose: bool,
) -> SolveResult:
    if data.is_floating_point():
        data = data.to(device=device)
    else:
        data = data.to(device=device, dtype=torch.float32)

    # Dual variables are sigma-scaled, so primal updates fold in tau*sigma.
    step_size = tau * sigma
    mu1_tensor = _make_mu_tensor(value=mu1, ref=data)
    mu2_tensor = _make_mu_tensor(value=mu2, ref=data)
    tv_dual_radius = mu1_tensor / sigma
    dir_dual_clip = 1.0 / sigma
    sparse_dual_clip = mu2_tensor / sigma
    negative_sparse_dual_clip = -sparse_dual_clip
    eps = 1e-9

    num_stripes = len(directions)
    num_vars = 1 + num_stripes

    clean = data.clone()
    stripe_components = [torch.zeros_like(input=data) for _ in directions]

    grad_row, grad_row_bar = _make_zero_pair(ref=data)
    grad_col, grad_col_bar = _make_zero_pair(ref=data)

    dir_dual = [torch.zeros_like(input=data) for _ in directions]
    dir_dual_bar = [torch.zeros_like(input=data) for _ in directions]
    sparse_dual = [torch.zeros_like(input=data) for _ in directions]
    sparse_dual_bar = [torch.zeros_like(input=data) for _ in directions]

    scratch = torch.empty_like(input=data)
    directional_diff = torch.empty_like(input=data)
    grad_norm = torch.empty_like(input=data)

    executed_iterations = 0
    with torch.no_grad():
        for iteration_idx in range(iterations):
            executed_iterations = iteration_idx + 1
            if verbose:
                print(f"\rIteration: {iteration_idx + 1} / {iterations}", end="")

            adjoint_grad(
                target=clean,
                p_h=grad_row_bar,
                p_v=grad_col_bar,
                scale=step_size,
            )

            for component_idx, mode in enumerate(directions):
                adjoint_dir(
                    target=stripe_components[component_idx],
                    q=dir_dual_bar[component_idx],
                    mode=mode,
                    scale=step_size,
                )
                stripe_components[component_idx].sub_(
                    sparse_dual_bar[component_idx], alpha=step_size
                )

            # Independent primal updates would drift off u + sum(s_i) = data.
            scratch.copy_(data)
            for stripe_component in stripe_components:
                scratch.sub_(stripe_component)
            scratch.sub_(clean).div_(num_vars)
            clean.add_(scratch)
            for stripe_component in stripe_components:
                stripe_component.add_(scratch)

            if proj:
                # Clamping clean would break equality unless stripes absorb the residual.
                scratch.copy_(clean)
                clean.clamp_(min=0, max=1)
                scratch.sub_(clean).div_(num_stripes)
                for stripe_component in stripe_components:
                    stripe_component.add_(scratch)

            grad_row_bar.copy_(grad_row)
            grad_col_bar.copy_(grad_col)

            forward_diff(x=clean, dim=1, out=scratch)
            grad_row.add_(scratch)
            forward_diff(x=clean, dim=2, out=scratch)
            grad_col.add_(scratch)

            torch.mul(grad_row, grad_row, out=grad_norm)
            grad_norm.addcmul_(grad_col, grad_col)
            grad_norm.sqrt_().clamp_(min=eps)
            torch.div(tv_dual_radius, grad_norm, out=scratch)
            scratch.clamp_(max=1.0)
            grad_row.mul_(scratch)
            grad_col.mul_(scratch)

            grad_row_bar.mul_(-1).add_(grad_row, alpha=2)
            grad_col_bar.mul_(-1).add_(grad_col, alpha=2)

            for component_idx, mode in enumerate(directions):
                dir_dual_bar[component_idx].copy_(dir_dual[component_idx])
                dir_diff(
                    x=stripe_components[component_idx],
                    mode=mode,
                    out=directional_diff,
                )
                dir_dual[component_idx].add_(directional_diff).clamp_(
                    min=-dir_dual_clip,
                    max=dir_dual_clip,
                )
                dir_dual_bar[component_idx].mul_(-1).add_(
                    dir_dual[component_idx], alpha=2
                )

                sparse_dual_bar[component_idx].copy_(sparse_dual[component_idx])
                sparse_dual[component_idx].add_(stripe_components[component_idx])
                torch.maximum(
                    sparse_dual[component_idx],
                    negative_sparse_dual_clip,
                    out=sparse_dual[component_idx],
                )
                torch.minimum(
                    sparse_dual[component_idx],
                    sparse_dual_clip,
                    out=sparse_dual[component_idx],
                )
                sparse_dual_bar[component_idx].mul_(-1).add_(
                    sparse_dual[component_idx], alpha=2
                )

    if verbose:
        print("")

    return SolveResult(clean=clean, iterations=executed_iterations)


def _make_mu_tensor(
    *,
    value: torch.Tensor | float,
    ref: torch.Tensor,
) -> torch.Tensor:
    out = torch.as_tensor(value, dtype=ref.dtype, device=ref.device)
    if out.dim() == 0:
        return out
    if out.dim() == 1 and out.numel() == ref.shape[0]:
        return out.reshape(-1, 1, 1)
    if out.shape == (ref.shape[0], 1, 1):
        return out
    raise ValueError("mu tensor must be scalar or match the batch size.")


def _make_zero_pair(ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    zero = torch.zeros_like(input=ref)
    return zero, zero.clone()
