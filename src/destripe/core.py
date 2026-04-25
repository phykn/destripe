import math

import numpy as np
import torch
import torch.nn.functional as F


_NUM_DIRS = 5  # Vertical + 4 diagonal directions
_NUM_VARS = 1 + _NUM_DIRS  # Clean image u + stripe components s_i


class UniversalStripeRemover:
    """Remove stripe noise from grayscale images with a PDHG solver.

    The model decomposes input data into a clean component ``u`` and directional
    stripe components ``s_i`` such that ``u + sum(s_i) = data``.

    Args:
        mu1: TV regularization weight for the clean image.
        mu2: L2 penalty weight for stripe components.
        device: Computation device. If ``None``, CUDA is used when available,
            otherwise CPU.
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

    # --- Public API ---

    def process(
        self,
        image: torch.Tensor | np.ndarray,
        iterations: int = 500,
        tol: float = 1e-5,
        proj: bool = True,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Destripe a grayscale image or a batch of grayscale images.

        Args:
            image: Input tensor/array with shape ``(H, W)`` or ``(N, H, W)``.
            iterations: Maximum number of PDHG iterations. Must be positive.
            tol: Relative convergence tolerance. Must be non-negative.
            proj: Whether to project the clean component onto ``[0, 1]``.
            verbose: Whether to print iteration progress.

        Returns:
            A tensor with the same rank as ``image`` containing the clean
            component estimate. Floating-point input dtypes are preserved
            (fp32 in / fp32 out, fp64 in / fp64 out); integer inputs are
            promoted to fp32.

        Raises:
            ValueError: If ``image`` rank is unsupported, contains non-finite
                values, or if ``iterations``/``tol`` are invalid.

        Note:
            The convergence check at iteration ``20k`` (``k >= 1``) compares
            ``u`` against its snapshot from iteration ``20(k-1)``. On CUDA,
            reductions used by the convergence norm are not bit-deterministic
            across runs by default; iteration count and outputs may differ
            for identical inputs unless ``torch.use_deterministic_algorithms``
            is enabled globally.
        """
        self._validate_solver_params(iterations=iterations, tol=tol)

        input_tensor = self._to_tensor(x=image)
        self._validate_finite_tensor(name="image", x=input_tensor)

        if input_tensor.dim() not in {2, 3}:
            raise ValueError("image must have shape (H, W) or (N, H, W).")

        squeeze_batch = input_tensor.dim() == 2
        if squeeze_batch:
            input_tensor = input_tensor.unsqueeze(0)

        clean = self._solve(
            data=input_tensor,
            iterations=iterations,
            tol=tol,
            proj=proj,
            verbose=verbose,
        )
        return clean.squeeze(0) if squeeze_batch else clean

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
        """Destripe a grayscale image tile-by-tile.

        Args:
            image: Input tensor/array with shape ``(H, W)`` or ``(1, H, W)``.
            tiles: Number of tiles per image side. Must be positive.
            iterations: Maximum number of PDHG iterations per tile. Must be
                positive.
            tol: Relative convergence tolerance. Must be non-negative.
            overlap: Overlap width (in pixels) before cosine blending. Must be
                non-negative.
            proj: Whether to project the clean component onto ``[0, 1]``.
            verbose: Whether to print iteration progress.

        Returns:
            A tensor with shape ``(H, W)``.

        Raises:
            ValueError: If ``image`` shape is unsupported, contains non-finite
                values, or if solver/tile parameters are invalid.
        """
        self._validate_solver_params(iterations=iterations, tol=tol)
        self._validate_tiling_params(tiles=tiles, overlap=overlap)

        input_tensor = self._to_tensor(x=image)
        self._validate_finite_tensor(name="image", x=input_tensor)

        if input_tensor.dim() == 2:
            image_2d = input_tensor
        elif input_tensor.dim() == 3 and input_tensor.shape[0] == 1:
            image_2d = input_tensor.squeeze(0)
        else:
            raise ValueError("image must have shape (H, W) or (1, H, W).")

        if tiles <= 1:
            return self.process(
                image=image_2d,
                iterations=iterations,
                tol=tol,
                proj=proj,
                verbose=verbose,
            )

        orig_h, orig_w = image_2d.shape
        padded_image = self._pad_reflect(
            t=image_2d,
            pad_bottom=(tiles - orig_h % tiles) % tiles,
            pad_right=(tiles - orig_w % tiles) % tiles,
        )

        padded_h, padded_w = padded_image.shape
        core_h, core_w = padded_h // tiles, padded_w // tiles
        overlap_pixels = max(min(overlap, core_h // 4, core_w // 4), 0)

        padded_image = self._pad_reflect(
            t=padded_image,
            pad_top=overlap_pixels,
            pad_bottom=overlap_pixels,
            pad_left=overlap_pixels,
            pad_right=overlap_pixels,
        )

        tile_h, tile_w = core_h + 2 * overlap_pixels, core_w + 2 * overlap_pixels
        indices = [(row, col) for row in range(tiles) for col in range(tiles)]
        tiles_batch = [
            padded_image[
                row * core_h : row * core_h + tile_h,
                col * core_w : col * core_w + tile_w,
            ]
            for row, col in indices
        ]
        tile_tensor = torch.stack(tensors=tiles_batch)

        if verbose:
            total_tiles = tiles * tiles
            print(
                f"Tiling {tiles}x{tiles}: {total_tiles} tiles of "
                f"{tile_h}x{tile_w}, overlap={overlap_pixels}"
            )

        cleaned_tiles = self.process(
            image=tile_tensor,
            iterations=iterations,
            tol=tol,
            proj=proj,
            verbose=verbose,
        )

        blend_weight = self._cosine_window(h=tile_h, w=tile_w, margin=overlap_pixels).to(
            device=cleaned_tiles.device, dtype=cleaned_tiles.dtype
        )
        blended_canvas = torch.zeros(
            padded_h + 2 * overlap_pixels,
            padded_w + 2 * overlap_pixels,
            device=cleaned_tiles.device,
            dtype=cleaned_tiles.dtype,
        )
        blend_sum = torch.zeros_like(input=blended_canvas)

        for idx, (row, col) in enumerate(indices):
            y0, x0 = row * core_h, col * core_w
            blended_canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] += (
                cleaned_tiles[idx] * blend_weight
            )
            blend_sum[y0 : y0 + tile_h, x0 : x0 + tile_w] += blend_weight

        blended_canvas /= blend_sum.clamp(min=1e-9)
        return blended_canvas[
            overlap_pixels : overlap_pixels + padded_h,
            overlap_pixels : overlap_pixels + padded_w,
        ][:orig_h, :orig_w]

    # --- Solver ---

    def _solve(
        self,
        data: torch.Tensor,
        iterations: int,
        tol: float,
        proj: bool,
        verbose: bool,
    ) -> torch.Tensor:
        if data.is_floating_point():
            data = data.to(device=self.device)
        else:
            data = data.to(device=self.device, dtype=torch.float32)

        # PDHG constants (pre-scaled by sigma)
        #   standard form: u -= tau * K^T p_bar
        #   here: step = tau * sigma is used with sigma-scaled duals
        step_size = self.tau * self.sigma
        tv_dual_radius = self.mu1 / self.sigma
        dir_dual_clip = 1.0 / self.sigma
        l2_dual_clip = self.mu2 / self.sigma
        eps = 1e-9

        clean = data.clone()
        stripe_components = [torch.zeros_like(input=data) for _ in range(_NUM_DIRS)]

        grad_row, grad_row_bar = self._zero_pair(ref=data)
        grad_col, grad_col_bar = self._zero_pair(ref=data)

        dir_dual = [torch.zeros_like(input=data) for _ in range(_NUM_DIRS)]
        dir_dual_bar = [torch.zeros_like(input=data) for _ in range(_NUM_DIRS)]
        l2_dual = [torch.zeros_like(input=data) for _ in range(_NUM_DIRS)]
        l2_dual_bar = [torch.zeros_like(input=data) for _ in range(_NUM_DIRS)]

        prev_clean = clean.clone()
        scratch = torch.empty_like(input=data)
        directional_diff = torch.empty_like(input=data)
        grad_norm = torch.empty_like(input=data)

        with torch.no_grad():
            for iteration_idx in range(iterations):
                if verbose:
                    print(f"\rIteration: {iteration_idx + 1} / {iterations}", end="")

                self._adjoint_grad(
                    target=clean,
                    p_h=grad_row_bar,
                    p_v=grad_col_bar,
                    a=step_size,
                )

                for mode in range(_NUM_DIRS):
                    self._adjoint_dir(
                        target=stripe_components[mode],
                        q=dir_dual_bar[mode],
                        mode=mode,
                        a=step_size,
                    )
                    stripe_components[mode].sub_(l2_dual_bar[mode], alpha=step_size)

                # Enforce u + sum(s_i) = data via shared scratch.
                scratch.copy_(data)
                for stripe_component in stripe_components:
                    scratch.sub_(stripe_component)
                scratch.sub_(clean).div_(_NUM_VARS)
                clean.add_(scratch)
                for stripe_component in stripe_components:
                    stripe_component.add_(scratch)

                if proj:
                    # Distribute clamp residual to stripes to maintain
                    # the constraint u + sum(s_i) = data.
                    torch.clamp(input=clean, max=0, out=scratch)
                    scratch.add_((clean - 1).clamp_(min=0))
                    scratch.div_(_NUM_DIRS)
                    for stripe_component in stripe_components:
                        stripe_component.add_(scratch)
                    clean.clamp_(min=0, max=1)

                grad_row_bar.copy_(grad_row)
                grad_col_bar.copy_(grad_col)

                self._forward_diff(x=clean, dim=1, out=scratch)
                grad_row.add_(scratch)
                self._forward_diff(x=clean, dim=2, out=scratch)
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

                for mode in range(_NUM_DIRS):
                    dir_dual_bar[mode].copy_(dir_dual[mode])
                    self._dir_diff(x=stripe_components[mode], mode=mode, out=directional_diff)
                    dir_dual[mode].add_(directional_diff).clamp_(
                        min=-dir_dual_clip,
                        max=dir_dual_clip,
                    )
                    dir_dual_bar[mode].mul_(-1).add_(dir_dual[mode], alpha=2)

                    l2_dual_bar[mode].copy_(l2_dual[mode])
                    l2_dual[mode].add_(stripe_components[mode]).clamp_(
                        min=-l2_dual_clip,
                        max=l2_dual_clip,
                    )
                    l2_dual_bar[mode].mul_(-1).add_(l2_dual[mode], alpha=2)

                if iteration_idx % 20 == 0:
                    if iteration_idx > 0:
                        torch.sub(input=clean, other=prev_clean, out=scratch)
                        rel_change = scratch.norm() / (prev_clean.norm() + eps)
                        if rel_change < tol:
                            if verbose:
                                print(f"\nConverged at iteration {iteration_idx + 1}.")
                            break
                    prev_clean.copy_(clean)

        if verbose:
            print("")

        return clean.cpu()

    # --- Validation ---

    @staticmethod
    def _validate_solver_params(iterations: int, tol: float) -> None:
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError(f"iterations must be a positive integer, got {iterations}.")
        if tol < 0:
            raise ValueError(f"tol must be non-negative, got {tol}.")

    @staticmethod
    def _validate_tiling_params(tiles: int, overlap: int) -> None:
        if not isinstance(tiles, int) or tiles <= 0:
            raise ValueError(f"tiles must be a positive integer, got {tiles}.")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}.")

    @staticmethod
    def _validate_finite_tensor(name: str, x: torch.Tensor) -> None:
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} must not contain NaN or Inf values.")

    # --- Differential operators ---

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

    # --- Tensor utilities ---

    def _to_tensor(
        self,
        x: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(data=x)
        if x.is_floating_point():
            return x
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
