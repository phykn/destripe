import math
import numbers
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .solver import SolveResult, solve_pdhg


DIRECTION_MODES = (0, 1, 2, 3, 4)


class UniversalStripeRemover:
    """Remove stripe noise from grayscale images with a PDHG solver."""

    def __init__(
        self,
        mu1: float = 1 / 3,
        mu2: float = 1 / 300,
        device: torch.device | str | None = None,
        directions: Sequence[int] | None = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.mu1 = self._validate_positive_real(name="mu1", value=mu1)
        self.mu2 = self._validate_positive_real(name="mu2", value=mu2)
        self.directions = self._validate_directions(directions)
        self.tau = 0.35
        self.sigma = 0.35

    def process(
        self,
        image: torch.Tensor | np.ndarray,
        iterations: int = 500,
        proj: bool = True,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Destripe a grayscale image or batch.

        Args:
            image: Input tensor/array with shape ``(H, W)`` or ``(N, H, W)``.
            iterations: Maximum number of PDHG iterations.
            proj: Whether to project the clean component onto ``[0, 1]``.
            verbose: Whether to print iteration progress.

        Returns:
            Clean estimate with the same rank as ``image``.
        """
        return self._process_with_info(
            image,
            iterations=iterations,
            proj=proj,
            verbose=verbose,
        ).clean

    def _process_with_info(
        self,
        image: torch.Tensor | np.ndarray,
        *,
        iterations: int,
        proj: bool,
        verbose: bool = False,
    ) -> SolveResult:
        self._validate_solver_params(iterations=iterations)

        input_tensor = self._convert_to_tensor(x=image)
        self._validate_finite_tensor(name="image", x=input_tensor)

        if input_tensor.dim() not in {2, 3}:
            raise ValueError("image must have shape (H, W) or (N, H, W).")
        if min(input_tensor.shape[-2:]) < 2:
            clean = self._make_solver_bypass_result(image=input_tensor, proj=proj)
            return SolveResult(clean=clean, iterations=0)

        squeeze_batch = input_tensor.dim() == 2
        if squeeze_batch:
            input_tensor = input_tensor.unsqueeze(0)

        result = self._run_solver(
            data=input_tensor,
            iterations=iterations,
            proj=proj,
            verbose=verbose,
        )
        clean = (result.clean.squeeze(0) if squeeze_batch else result.clean).cpu()
        return SolveResult(clean=clean, iterations=result.iterations)

    def process_tiled(
        self,
        image: torch.Tensor | np.ndarray,
        tiles: int = 1,
        iterations: int = 500,
        overlap: int = 64,
        proj: bool = True,
        verbose: bool = False,
        tile_mus: Sequence[tuple[float, float]] | None = None,
    ) -> torch.Tensor:
        """Destripe a grayscale image tile-by-tile.

        Args:
            image: Input tensor/array with shape ``(H, W)`` or ``(1, H, W)``.
            tiles: Number of tiles per image side.
            iterations: Maximum number of PDHG iterations per tile.
            overlap: Overlap width in pixels before cosine blending.
            proj: Whether to project the clean component onto ``[0, 1]``.
            verbose: Whether to print iteration progress.
            tile_mus: Optional ``(mu1, mu2)`` values in row-major tile order.

        Returns:
            A tensor with shape ``(H, W)``.
        """
        self._validate_solver_params(iterations=iterations)
        self._validate_tiling_params(tiles=tiles, overlap=overlap)

        input_tensor = self._convert_to_tensor(x=image)
        self._validate_finite_tensor(name="image", x=input_tensor)

        if input_tensor.dim() == 2:
            image_2d = input_tensor
        elif input_tensor.dim() == 3 and input_tensor.shape[0] == 1:
            image_2d = input_tensor.squeeze(0)
        else:
            raise ValueError("image must have shape (H, W) or (1, H, W).")

        tile_mu_values = None
        if tile_mus is not None:
            tile_mu_values = self._validate_tile_mus(
                tile_mus=tile_mus,
                expected_count=tiles * tiles,
            )

        orig_h, orig_w = image_2d.shape
        if min(orig_h, orig_w) < 2:
            if tile_mu_values is not None:
                raise ValueError(
                    "tile_mus cannot be applied to tiles smaller than 2x2."
                )
            return self._make_solver_bypass_result(image=image_2d, proj=proj)

        if tiles <= 1:
            if tile_mu_values is None:
                return self.process(
                    image=image_2d,
                    iterations=iterations,
                    proj=proj,
                    verbose=verbose,
                )

            tile_mu1, tile_mu2 = tile_mu_values[0]
            return (
                self._run_solver(
                    data=image_2d.unsqueeze(0),
                    iterations=iterations,
                    proj=proj,
                    verbose=verbose,
                    mu1=tile_mu1,
                    mu2=tile_mu2,
                )
                .clean.squeeze(0)
                .cpu()
            )

        pad_bottom = (tiles - orig_h % tiles) % tiles
        pad_right = (tiles - orig_w % tiles) % tiles
        padded_h = orig_h + pad_bottom
        padded_w = orig_w + pad_right
        core_h, core_w = padded_h // tiles, padded_w // tiles
        if core_h < 2 or core_w < 2:
            if tile_mu_values is not None:
                raise ValueError(
                    "tile_mus cannot be applied to tiles smaller than 2x2."
                )
            return self.process(
                image=image_2d,
                iterations=iterations,
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

        if tile_mu_values is None:
            cleaned_tiles = self._run_solver(
                data=tile_tensor,
                iterations=iterations,
                proj=proj,
                verbose=verbose,
            ).clean
        else:
            tile_mu1, tile_mu2 = self._make_tile_mu_tensors(
                tile_mus=tile_mu_values,
                ref=tile_tensor,
            )
            cleaned_tiles = self._run_solver(
                data=tile_tensor,
                iterations=iterations,
                proj=proj,
                verbose=verbose,
                mu1=tile_mu1,
                mu2=tile_mu2,
            ).clean

        blend_weight = self._make_cosine_window(
            h=tile_h,
            w=tile_w,
            margin=overlap_pixels,
        ).to(device=cleaned_tiles.device, dtype=cleaned_tiles.dtype)
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
        ][:orig_h, :orig_w].cpu()

    @staticmethod
    def _make_tile_mu_tensors(
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
            raise ValueError(
                "tile_mus must be a sequence of (mu1, mu2) pairs."
            ) from exc
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
            if (
                not math.isfinite(mu1_float)
                or not math.isfinite(mu2_float)
                or mu1_float <= 0
                or mu2_float <= 0
            ):
                raise ValueError("tile_mus entries must be positive finite pairs.")
            validated.append((mu1_float, mu2_float))
        return validated

    def _run_solver(
        self,
        data: torch.Tensor,
        iterations: int,
        proj: bool,
        verbose: bool,
        mu1: torch.Tensor | float | None = None,
        mu2: torch.Tensor | float | None = None,
    ) -> SolveResult:
        return solve_pdhg(
            data=data,
            directions=self.directions,
            mu1=self.mu1 if mu1 is None else mu1,
            mu2=self.mu2 if mu2 is None else mu2,
            device=self.device,
            tau=self.tau,
            sigma=self.sigma,
            iterations=iterations,
            proj=proj,
            verbose=verbose,
        )

    @staticmethod
    def _make_solver_bypass_result(*, image: torch.Tensor, proj: bool) -> torch.Tensor:
        clean = image.clone()
        if proj:
            clean.clamp_(min=0, max=1)
        return clean.cpu()

    @staticmethod
    def _validate_solver_params(iterations: int) -> None:
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, numbers.Integral)
            or iterations <= 0
        ):
            raise ValueError(
                f"iterations must be a positive integer, got {iterations}."
            )

    @staticmethod
    def _validate_tiling_params(tiles: int, overlap: int) -> None:
        if (
            isinstance(tiles, bool)
            or not isinstance(tiles, numbers.Integral)
            or tiles <= 0
        ):
            raise ValueError(f"tiles must be a positive integer, got {tiles}.")
        if (
            isinstance(overlap, bool)
            or not isinstance(overlap, numbers.Integral)
            or overlap < 0
        ):
            raise ValueError(f"overlap must be a non-negative integer, got {overlap}.")

    @staticmethod
    def _validate_positive_real(*, name: str, value: object) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive finite number, got {value}.")
        return float(value)

    @staticmethod
    def _validate_directions(directions: Sequence[int] | None) -> tuple[int, ...]:
        message = "directions must contain integers in the range 0..4."
        if directions is None:
            return DIRECTION_MODES
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
                raise ValueError(message)
            if mode not in DIRECTION_MODES:
                raise ValueError(message)
        if len(set(values)) != len(values):
            raise ValueError("directions must not contain duplicates.")
        return values

    @staticmethod
    def _validate_finite_tensor(name: str, x: torch.Tensor) -> None:
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} must not contain NaN or Inf values.")

    @staticmethod
    def _convert_to_tensor(
        x: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(data=x)
        if x.is_complex():
            raise ValueError("image must contain real values.")
        if x.is_floating_point():
            return x
        return x.to(dtype=torch.float32)

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
    def _make_cosine_window(
        h: int,
        w: int,
        margin: int,
    ) -> torch.Tensor:
        win = torch.ones(h, w)
        if margin > 0:
            ramp = (
                1 - torch.cos(input=torch.linspace(start=0, end=math.pi, steps=margin))
            ) / 2
            win[:margin, :] *= ramp[:, None]
            win[-margin:, :] *= ramp.flip(dims=(0,))[:, None]
            win[:, :margin] *= ramp[None, :]
            win[:, -margin:] *= ramp.flip(dims=(0,))[None, :]
        return win
