import math

import numpy as np
import torch
import torch.nn.functional as F


_NUM_DIRS = 5  # Vertical + 4 diagonal directions
_NUM_VARS = 1 + _NUM_DIRS  # Clean image u + stripe components s_i


class UniversalStripeRemover:
    """PDHG-based universal stripe noise remover.

    Decomposes an image into a clean component `u` and directional stripe
    components `s_i`, such that `u + sum(s_i) = data`.

    Args:
        mu1: TV regularization weight for the clean image.
        mu2: L2 penalty weight for stripe components.
        device: Computation device. Auto-selects CUDA if available.
    """

    def __init__(
        self,
        mu1: float = 0.33,
        mu2: float = 0.003,
        device: torch.device | str | None = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.mu1 = mu1
        self.mu2 = mu2
        self.tau = 0.35
        self.sigma = 0.35

    def process(
        self,
        image: torch.Tensor | np.ndarray,
        iterations: int = 500,
        tol: float = 1e-5,
        proj: bool = True,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Processes the input image to remove stripe noise.

        Args:
            image: Input image.
            iterations: Maximum number of PDHG iterations.
            tol: Convergence tolerance.
            proj: Whether to project the clean image onto [0, 1].
            verbose: Whether to print iteration progress.

        Returns:
            The destriped clean image.
        """
        x = self._to_tensor(x=image)
        if x.dim() not in {2, 3}:
            raise ValueError("image must have shape (H, W) or (N, H, W).")

        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)

        result = self._solve(
            data=x, iterations=iterations, tol=tol, proj=proj, verbose=verbose
        )
        return result.squeeze(0) if squeeze else result

    def process_tiled(
        self,
        image: torch.Tensor | np.ndarray,
        tiles: int = 1,
        iterations: int = 500,
        tol: float = 1e-5,
        overlap: int = 64,
        proj: bool = True,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Processes the input image tile by tile for large data.

        Args:
            image: Input image.
            tiles: Number of tiles per side.
            iterations: Maximum number of PDHG iterations.
            tol: Convergence tolerance.
            overlap: Number of pixels for overlap blending between tiles.
            proj: Whether to project the clean image onto [0, 1].
            verbose: Whether to print iteration progress.

        Returns:
            The destriped clean image.
        """
        data = self._to_tensor(x=image)
        if data.dim() == 2:
            pass
        elif data.dim() == 3 and data.shape[0] == 1:
            data = data.squeeze(0)
        else:
            raise ValueError("image must have shape (H, W) or (1, H, W).")

        if tiles <= 1:
            return self.process(
                image=data,
                iterations=iterations,
                tol=tol,
                proj=proj,
                verbose=verbose,
            )

        h, w = data.shape
        data = self._pad_reflect(
            t=data,
            pad_bottom=(tiles - h % tiles) % tiles,
            pad_right=(tiles - w % tiles) % tiles,
        )

        padded_h, padded_w = data.shape
        core_h, core_w = padded_h // tiles, padded_w // tiles
        ov = max(min(overlap, core_h // 4, core_w // 4), 0)

        data = self._pad_reflect(
            t=data, pad_top=ov, pad_bottom=ov, pad_left=ov, pad_right=ov
        )

        tile_h, tile_w = core_h + 2 * ov, core_w + 2 * ov
        tile_list = [
            data[i * core_h : i * core_h + tile_h, j * core_w : j * core_w + tile_w]
            for i in range(tiles)
            for j in range(tiles)
        ]
        batch = torch.stack(tensors=tile_list)

        if verbose:
            total = tiles * tiles
            print(
                f"Tiling {tiles}x{tiles}: {total} tiles of "
                f"{tile_h}x{tile_w}, overlap={ov}"
            )

        results = self.process(
            image=batch, iterations=iterations, tol=tol, proj=proj, verbose=verbose
        )

        weight = self._cosine_window(h=tile_h, w=tile_w, margin=ov)
        canvas = torch.zeros(padded_h + 2 * ov, padded_w + 2 * ov)
        weight_sum = torch.zeros_like(input=canvas)

        for idx, (i, j) in enumerate(
            (i, j) for i in range(tiles) for j in range(tiles)
        ):
            y, x = i * core_h, j * core_w
            canvas[y : y + tile_h, x : x + tile_w] += results[idx] * weight
            weight_sum[y : y + tile_h, x : x + tile_w] += weight

        canvas /= weight_sum.clamp(min=1e-9)
        return canvas[ov : ov + padded_h, ov : ov + padded_w][:h, :w]

    def _solve(
        self,
        data: torch.Tensor,
        iterations: int,
        tol: float,
        proj: bool,
        verbose: bool,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=torch.float32)

        # PDHG constants (pre-scaled by sigma)
        #   standard form: u -= tau * K^T p_bar
        #   here: step = tau * sigma is used with sigma-scaled duals
        step = self.tau * self.sigma
        lam = self.mu1 / self.sigma  # TV dual ball radius
        q_clip = 1.0 / self.sigma  # directional sparsity dual bound
        r_clip = self.mu2 / self.sigma  # L2 penalty dual bound
        eps = 1e-9

        # Primal variables
        u = data.clone()
        s = [torch.zeros_like(input=data) for _ in range(_NUM_DIRS)]

        # Dual variables for TV gradient
        p_h, p_h_bar = self._zero_pair(ref=data)
        p_v, p_v_bar = self._zero_pair(ref=data)

        # Dual variables for directional sparsity (q) and L2 penalty (r)
        q_pairs = [self._zero_pair(ref=data) for _ in range(_NUM_DIRS)]
        r_pairs = [self._zero_pair(ref=data) for _ in range(_NUM_DIRS)]
        q = [pair[0] for pair in q_pairs]
        q_bar = [pair[1] for pair in q_pairs]
        r = [pair[0] for pair in r_pairs]
        r_bar = [pair[1] for pair in r_pairs]

        u_prev = u.clone()
        buf = torch.empty_like(input=data)
        dir_buf = torch.empty_like(input=data)
        diff_buf = torch.empty_like(input=data)

        with torch.no_grad():
            for k in range(iterations):
                if verbose:
                    print(f"\rIteration: {k + 1} / {iterations}", end="")

                # --- Primal step ---
                self._adjoint_grad(target=u, p_h=p_h_bar, p_v=p_v_bar, a=step)

                for i in range(_NUM_DIRS):
                    self._adjoint_dir(target=s[i], q=q_bar[i], mode=i, a=step)
                    s[i].sub_(r_bar[i], alpha=step)

                # Enforce constraint: u + sum(s_i) = data
                buf.copy_(data)
                for si in s:
                    buf.sub_(si)
                buf.sub_(u).div_(_NUM_VARS)
                u.add_(buf)
                for si in s:
                    si.add_(buf)

                # Project u onto [0, 1], redistribute excess to s_i
                if proj:
                    torch.clamp(input=u, max=0, out=buf)
                    buf.add_((u - 1).clamp_(min=0))
                    buf.div_(_NUM_DIRS)
                    for si in s:
                        si.add_(buf)
                    u.clamp_(min=0, max=1)

                # --- Dual step: TV (isotropic projection) ---
                p_h_bar.copy_(p_h)
                p_v_bar.copy_(p_v)

                self._forward_diff(x=u, dim=1, out=buf)
                p_h.add_(buf)
                self._forward_diff(x=u, dim=2, out=buf)
                p_v.add_(buf)

                torch.mul(p_h, p_h, out=diff_buf)
                diff_buf.addcmul_(p_v, p_v)
                diff_buf.sqrt_().clamp_(min=eps)
                torch.div(lam, diff_buf, out=buf)
                buf.clamp_(max=1.0)
                p_h.mul_(buf)
                p_v.mul_(buf)

                p_h_bar.mul_(-1).add_(p_h, alpha=2)
                p_v_bar.mul_(-1).add_(p_v, alpha=2)

                # --- Dual step: directional sparsity + L2 ---
                for i in range(_NUM_DIRS):
                    q_bar[i].copy_(q[i])
                    self._dir_diff(x=s[i], mode=i, out=dir_buf)
                    q[i].add_(dir_buf).clamp_(min=-q_clip, max=q_clip)
                    q_bar[i].mul_(-1).add_(q[i], alpha=2)

                    r_bar[i].copy_(r[i])
                    r[i].add_(s[i]).clamp_(min=-r_clip, max=r_clip)
                    r_bar[i].mul_(-1).add_(r[i], alpha=2)

                # Convergence check
                if k > 0 and k % 20 == 0:
                    torch.sub(input=u, other=u_prev, out=buf)
                    rel = buf.norm() / (u_prev.norm() + eps)
                    if rel < tol:
                        if verbose:
                            print(f"\nConverged at iteration {k + 1}.")
                        break
                    u_prev.copy_(u)

        if verbose:
            print("")

        return u.cpu()

    @staticmethod
    def _forward_diff(
        x: torch.Tensor,
        dim: int,
        out: torch.Tensor,
    ) -> None:
        """Forward difference with Neumann BC (last element = 0)."""
        n = x.size(dim)
        torch.sub(
            x.narrow(dim=dim, start=1, length=n - 1),
            x.narrow(dim=dim, start=0, length=n - 1),
            out=out.narrow(dim=dim, start=0, length=n - 1),
        )
        out.narrow(dim=dim, start=n - 1, length=1).zero_()

    @staticmethod
    def _dir_diff(
        x: torch.Tensor,
        mode: int,
        out: torch.Tensor,
    ) -> None:
        """Directional difference operator for the given mode."""
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

    @staticmethod
    def _adjoint_1d(
        target: torch.Tensor,
        p: torch.Tensor,
        dim: int,
        a: float,
    ) -> None:
        """Adjoint of 1D forward difference: adds -div(p)*a to target."""
        idx = [slice(None)] * 3

        idx[dim] = 0
        target[tuple(idx)].add_(p[tuple(idx)], alpha=a)

        idx[dim] = slice(1, -1)
        idx2 = list(idx)
        idx2[dim] = slice(None, -2)
        target[tuple(idx)].sub_(p[tuple(idx2)], alpha=a).add_(p[tuple(idx)], alpha=a)

        idx[dim] = -1
        idx2 = list(idx)
        idx2[dim] = -2
        target[tuple(idx)].sub_(p[tuple(idx2)], alpha=a)

    @classmethod
    def _adjoint_grad(
        cls,
        target: torch.Tensor,
        p_h: torch.Tensor,
        p_v: torch.Tensor,
        a: float,
    ) -> None:
        """Adjoint of 2D gradient operator."""
        cls._adjoint_1d(target=target, p=p_h, dim=1, a=a)
        cls._adjoint_1d(target=target, p=p_v, dim=2, a=a)

    @staticmethod
    def _adjoint_dir(
        target: torch.Tensor,
        q: torch.Tensor,
        mode: int,
        a: float,
    ) -> None:
        """Adjoint of directional difference operator."""
        if mode == 0:
            target[:, 1:, :].sub_(q[:, :-1, :], alpha=a)
            target[:, :-1, :].add_(q[:, :-1, :], alpha=a)
        elif mode == 1:
            target[:, 2:, 1:].sub_(q[:, :-2, :-1], alpha=a)
            target[:, :-2, :-1].add_(q[:, :-2, :-1], alpha=a)
        elif mode == 2:
            target[:, 1:, 1:].sub_(q[:, :-1, :-1], alpha=a)
            target[:, :-1, :-1].add_(q[:, :-1, :-1], alpha=a)
        elif mode == 3:
            target[:, 2:, :-1].sub_(q[:, :-2, 1:], alpha=a)
            target[:, :-2, 1:].add_(q[:, :-2, 1:], alpha=a)
        elif mode == 4:
            target[:, 1:, :-1].sub_(q[:, :-1, 1:], alpha=a)
            target[:, :-1, 1:].add_(q[:, :-1, 1:], alpha=a)

    def _to_tensor(
        self,
        x: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(data=x)
        return x.to(dtype=torch.float32)

    @staticmethod
    def _zero_pair(
        ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros_like(input=ref)
        return z, z.clone()

    @staticmethod
    def _pad_reflect(
        t: torch.Tensor,
        pad_top: int = 0,
        pad_bottom: int = 0,
        pad_left: int = 0,
        pad_right: int = 0,
    ) -> torch.Tensor:
        if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
            return t
        return F.pad(
            input=t.unsqueeze(0),
            pad=(pad_left, pad_right, pad_top, pad_bottom),
            mode="reflect",
        ).squeeze(0)

    @staticmethod
    def _cosine_window(
        h: int,
        w: int,
        margin: int,
    ) -> torch.Tensor:
        win = torch.ones(h, w)
        if margin > 0:
            ramp = 0.5 * (
                1.0
                - torch.cos(input=torch.linspace(start=0, end=math.pi, steps=margin))
            )
            win[:margin, :] *= ramp[:, None]
            win[-margin:, :] *= ramp.flip(dims=(0,))[:, None]
            win[:, :margin] *= ramp[None, :]
            win[:, -margin:] *= ramp.flip(dims=(0,))[None, :]
        return win
