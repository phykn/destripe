import torch


def forward_diff(
    x: torch.Tensor,
    dim: int,
    out: torch.Tensor,
) -> None:
    n = x.size(dim)
    torch.sub(
        x.narrow(dim=dim, start=1, length=n - 1),
        x.narrow(dim=dim, start=0, length=n - 1),
        out=out.narrow(dim=dim, start=0, length=n - 1),
    )
    out.narrow(dim=dim, start=n - 1, length=1).zero_()


def dir_diff(
    x: torch.Tensor,
    mode: int,
    out: torch.Tensor,
) -> None:
    out.zero_()
    if mode == 0:
        out[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]
    elif mode == 1:
        out[:, :-2, :-1] = x[:, 2:, 1:] - x[:, :-2, :-1]
    elif mode == 2:
        out[:, :-1, :-1] = x[:, 1:, 1:] - x[:, :-1, :-1]
    elif mode == 3:
        out[:, :-2, 1:] = x[:, 2:, :-1] - x[:, :-2, 1:]
    elif mode == 4:
        out[:, :-1, 1:] = x[:, 1:, :-1] - x[:, :-1, 1:]


def adjoint_1d(
    target: torch.Tensor,
    p: torch.Tensor,
    dim: int,
    scale: float,
) -> None:
    idx = [slice(None)] * 3

    idx[dim] = 0
    target[tuple(idx)].add_(p[tuple(idx)], alpha=scale)

    idx[dim] = slice(1, -1)
    idx2 = list(idx)
    idx2[dim] = slice(None, -2)
    target[tuple(idx)].sub_(p[tuple(idx2)], alpha=scale).add_(
        p[tuple(idx)],
        alpha=scale,
    )

    idx[dim] = -1
    idx2 = list(idx)
    idx2[dim] = -2
    target[tuple(idx)].sub_(p[tuple(idx2)], alpha=scale)


def adjoint_grad(
    target: torch.Tensor,
    p_h: torch.Tensor,
    p_v: torch.Tensor,
    scale: float,
) -> None:
    adjoint_1d(target=target, p=p_h, dim=1, scale=scale)
    adjoint_1d(target=target, p=p_v, dim=2, scale=scale)


def adjoint_dir(
    target: torch.Tensor,
    q: torch.Tensor,
    mode: int,
    scale: float,
) -> None:
    if mode == 0:
        target[:, 1:, :].sub_(q[:, :-1, :], alpha=scale)
        target[:, :-1, :].add_(q[:, :-1, :], alpha=scale)
    elif mode == 1:
        target[:, 2:, 1:].sub_(q[:, :-2, :-1], alpha=scale)
        target[:, :-2, :-1].add_(q[:, :-2, :-1], alpha=scale)
    elif mode == 2:
        target[:, 1:, 1:].sub_(q[:, :-1, :-1], alpha=scale)
        target[:, :-1, :-1].add_(q[:, :-1, :-1], alpha=scale)
    elif mode == 3:
        target[:, 2:, :-1].sub_(q[:, :-2, 1:], alpha=scale)
        target[:, :-2, 1:].add_(q[:, :-2, 1:], alpha=scale)
    elif mode == 4:
        target[:, 1:, :-1].sub_(q[:, :-1, 1:], alpha=scale)
        target[:, :-1, 1:].add_(q[:, :-1, 1:], alpha=scale)
