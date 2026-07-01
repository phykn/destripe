import math
import numbers
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F


_NUM_DIRS = 5  # Vertical + 4 diagonal directions
_ALL_DIRECTIONS = tuple(range(_NUM_DIRS))


class UniversalStripeRemover:
    """Remove stripe noise from grayscale images with a PDHG solver.

    The model decomposes input data into a clean component ``u`` and directional
    stripe components ``s_i`` such that ``u + sum(s_i) = data``. By default all
    five stripe directions are active; pass ``directions`` to use a subset.

    Args:
        mu1: TV regularization weight for the clean image.
        mu2: L2 penalty weight for stripe components.
        device: Computation device. If ``None``, CUDA is used when available,
            otherwise CPU.
        directions: Optional sequence of active direction modes. Modes must be
            unique integers in ``0..4``. If ``None``, all modes are used.
    """

    def __init__(
        self,
        mu1: float = 0.33,
        mu2: float = 0.003,
        device: torch.device | str | None = None,
        directions: Sequence[int] | None = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.mu1 = mu1
        self.mu2 = mu2
        self.directions = self._validate_directions(directions)
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
        """Destripe a grayscale image or a batch of grayscale images.

        Uses the active stripe ``directions`` configured on this remover.

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
        tile_mus: Sequence[tuple[float, float]] | None = None,
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
            tile_mus: Optional ``(mu1, mu2)`` values for each tile in row-major
                order. When omitted, tiles use the existing batch path.

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

        validated_tile_mus = None
        if tile_mus is not None:
            validated_tile_mus = self._validate_tile_mus(
                tile_mus=tile_mus,
                expected_count=tiles * tiles,
            )

        orig_h, orig_w = image_2d.shape
        if min(orig_h, orig_w) < 2:
            return image_2d.clone()

        if tiles <= 1:
            return self._process_single_tile(
                image=image_2d,
                iterations=iterations,
                tol=tol,
                proj=proj,
                verbose=verbose,
                tile_mu=validated_tile_mus[0]
                if validated_tile_mus is not None
                else None,
            )

        pad_bottom = (tiles - orig_h % tiles) % tiles
        pad_right = (tiles - orig_w % tiles) % tiles
        padded_h = orig_h + pad_bottom
        padded_w = orig_w + pad_right
        core_h, core_w = padded_h // tiles, padded_w // tiles
        if core_h < 2 or core_w < 2:
            return self.process(
                image=image_2d,
                iterations=iterations,
                tol=tol,
                proj=proj,
                verbose=verbose,
            )

        padded_image = self._pad_reflect(
            t=image_2d,
            pad_bottom=pad_bottom,
            pad_right=pad_right,
        )

        overlap_pixels = min(overlap, core_h // 4, core_w // 4)

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

        if tile_mus is None:
            cleaned_tiles = self.process(
                image=tile_tensor,
                iterations=iterations,
                tol=tol,
                proj=proj,
                verbose=verbose,
            )
        else:
            tile_mu1, tile_mu2 = self._tile_mu_tensors(
                tile_mus=validated_tile_mus,
                ref=tile_tensor,
            )
            cleaned_tiles = self._solve(
                data=tile_tensor,
                iterations=iterations,
                tol=tol,
                proj=proj,
                verbose=verbose,
                mu1=tile_mu1,
                mu2=tile_mu2,
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

    def _process_single_tile(
        self,
        *,
        image: torch.Tensor,
        iterations: int,
        tol: float,
        proj: bool,
        verbose: bool,
        tile_mu: tuple[float, float] | None,
    ) -> torch.Tensor:
        if tile_mu is None:
            return self.process(
                image=image,
                iterations=iterations,
                tol=tol,
                proj=proj,
                verbose=verbose,
            )

        original_mu1, original_mu2 = self.mu1, self.mu2
        try:
            self.mu1 = float(tile_mu[0])
            self.mu2 = float(tile_mu[1])
            return self.process(
                image=image,
                iterations=iterations,
                tol=tol,
                proj=proj,
                verbose=verbose,
            )
        finally:
            self.mu1, self.mu2 = original_mu1, original_mu2

    @staticmethod
    def _tile_mu_tensors(
        tile_mus: Sequence[tuple[float, float]],
        ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mus = torch.as_tensor(tile_mus, dtype=ref.dtype)
        return mus[:, 0], mus[:, 1]

    @staticmethod
    def _validate_tile_mus(
        tile_mus: Sequence[tuple[float, float]],
        expected_count: int,
    ) -> list[tuple[float, float]]:
        try:
            count = len(tile_mus)
        except TypeError as exc:
            raise ValueError("tile_mus must be a sequence of (mu1, mu2) pairs.") from exc
        if count != expected_count:
            raise ValueError("tile_mus length must match the number of tiles.")

        validated = []
        for entry in tile_mus:
            if (
                not isinstance(entry, Sequence)
                or isinstance(entry, (str, bytes))
                or len(entry) != 2
            ):
                raise ValueError("tile_mus entries must be 2-item numeric pairs.")
            mu1, mu2 = entry
            if (
                isinstance(mu1, bool)
                or isinstance(mu2, bool)
                or not isinstance(mu1, numbers.Real)
                or not isinstance(mu2, numbers.Real)
            ):
                raise ValueError("tile_mus entries must be finite numeric pairs.")
            mu1_float = float(mu1)
            mu2_float = float(mu2)
            if not math.isfinite(mu1_float) or not math.isfinite(mu2_float):
                raise ValueError("tile_mus entries must be finite numeric pairs.")
            validated.append((mu1_float, mu2_float))
        return validated

    def _solve(
        self,
        data: torch.Tensor,
        iterations: int,
        tol: float,
        proj: bool,
        verbose: bool,
        mu1: torch.Tensor | None = None,
        mu2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if data.is_floating_point():
            data = data.to(device=self.device)
        else:
            data = data.to(device=self.device, dtype=torch.float32)

        # PDHG constants (pre-scaled by sigma)
        #   standard form: u -= tau * K^T p_bar
        #   here: step = tau * sigma is used with sigma-scaled duals
        step_size = self.tau * self.sigma
        mu1_tensor = self._solver_mu_tensor(value=mu1, fallback=self.mu1, ref=data)
        mu2_tensor = self._solver_mu_tensor(value=mu2, fallback=self.mu2, ref=data)
        tv_dual_radius = mu1_tensor / self.sigma
        dir_dual_clip = 1.0 / self.sigma
        l2_dual_clip = mu2_tensor / self.sigma
        eps = 1e-9

        num_stripes = len(self.directions)
        num_vars = 1 + num_stripes

        clean = data.clone()
        stripe_components = [torch.zeros_like(input=data) for _ in self.directions]

        grad_row, grad_row_bar = self._zero_pair(ref=data)
        grad_col, grad_col_bar = self._zero_pair(ref=data)

        dir_dual = [torch.zeros_like(input=data) for _ in self.directions]
        dir_dual_bar = [torch.zeros_like(input=data) for _ in self.directions]
        l2_dual = [torch.zeros_like(input=data) for _ in self.directions]
        l2_dual_bar = [torch.zeros_like(input=data) for _ in self.directions]

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

                for component_idx, mode in enumerate(self.directions):
                    self._adjoint_dir(
                        target=stripe_components[component_idx],
                        q=dir_dual_bar[component_idx],
                        mode=mode,
                        a=step_size,
                    )
                    stripe_components[component_idx].sub_(
                        l2_dual_bar[component_idx], alpha=step_size
                    )

                # Enforce u + sum(s_i) = data via shared scratch.
                scratch.copy_(data)
                for stripe_component in stripe_components:
                    scratch.sub_(stripe_component)
                scratch.sub_(clean).div_(num_vars)
                clean.add_(scratch)
                for stripe_component in stripe_components:
                    stripe_component.add_(scratch)

                if proj:
                    # Distribute clamp residual to stripes to maintain
                    # the constraint u + sum(s_i) = data.
                    torch.clamp(input=clean, max=0, out=scratch)
                    scratch.add_((clean - 1).clamp_(min=0))
                    scratch.div_(num_stripes)
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

                for component_idx, mode in enumerate(self.directions):
                    dir_dual_bar[component_idx].copy_(dir_dual[component_idx])
                    self._dir_diff(
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

                    l2_dual_bar[component_idx].copy_(l2_dual[component_idx])
                    l2_dual[component_idx].add_(stripe_components[component_idx])
                    torch.maximum(
                        l2_dual[component_idx],
                        -l2_dual_clip,
                        out=l2_dual[component_idx],
                    )
                    torch.minimum(
                        l2_dual[component_idx],
                        l2_dual_clip,
                        out=l2_dual[component_idx],
                    )
                    l2_dual_bar[component_idx].mul_(-1).add_(
                        l2_dual[component_idx], alpha=2
                    )

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

    @staticmethod
    def _solver_mu_tensor(
        *,
        value: torch.Tensor | None,
        fallback: float,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        if value is None:
            return torch.as_tensor(fallback, dtype=ref.dtype, device=ref.device)

        out = torch.as_tensor(value, dtype=ref.dtype, device=ref.device)
        if out.dim() == 0:
            return out
        if out.dim() == 1 and out.numel() == ref.shape[0]:
            return out.reshape(-1, 1, 1)
        if out.shape == (ref.shape[0], 1, 1):
            return out
        raise ValueError("mu tensor must be scalar or match the batch size.")

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
    def _validate_directions(directions: Sequence[int] | None) -> tuple[int, ...]:
        if directions is None:
            return _ALL_DIRECTIONS
        if not isinstance(directions, Sequence):
            raise ValueError("directions must be None or a sequence of integers.")

        try:
            values = tuple(directions)
        except TypeError as exc:
            raise ValueError(
                "directions must be None or a sequence of integers."
            ) from exc

        if not values:
            raise ValueError("directions must not be empty.")
        for mode in values:
            if not isinstance(mode, int) or isinstance(mode, bool):
                raise ValueError("directions must contain integers in the range 0..4.")
            if mode < 0 or mode >= _NUM_DIRS:
                raise ValueError("directions must contain integers in the range 0..4.")
        if len(set(values)) != len(values):
            raise ValueError("directions must not contain duplicates.")
        return values

    @staticmethod
    def _validate_finite_tensor(name: str, x: torch.Tensor) -> None:
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} must not contain NaN or Inf values.")

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

    @staticmethod
    def _to_tensor(
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
