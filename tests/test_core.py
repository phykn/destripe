import numpy as np
import pytest
import torch

import destripe.ops as destripe_ops
from destripe import preprocess
from destripe import UniversalStripeRemover, destripe
from destripe.adaptive import estimate_adaptive_params


@pytest.fixture()
def remover() -> UniversalStripeRemover:
    return UniversalStripeRemover(device="cpu")


@pytest.fixture()
def gray_image() -> np.ndarray:
    """32x32 smooth gradient with synthetic horizontal stripes."""
    rng = np.random.default_rng(42)
    base = np.linspace(0, 1, 32).reshape(1, -1).repeat(32, axis=0)
    stripe = np.zeros_like(base)
    stripe[10, :] = 0.3
    stripe[20, :] = -0.2
    return np.clip(base + stripe + rng.normal(0, 0.01, base.shape), 0, 1)


class TestAdjointConsistency:
    """<Dx, y> must equal <x, D^T y> for all operator pairs."""

    SHAPE = (1, 16, 16)

    def test_forward_diff_adjoint_dim1(self, remover: UniversalStripeRemover) -> None:
        self._check_gradient_adjoint(remover, dim=1)

    def test_forward_diff_adjoint_dim2(self, remover: UniversalStripeRemover) -> None:
        self._check_gradient_adjoint(remover, dim=2)

    @pytest.mark.parametrize("mode", range(5))
    def test_dir_diff_adjoint(self, remover: UniversalStripeRemover, mode: int) -> None:
        torch.manual_seed(mode)
        x = torch.randn(self.SHAPE)
        y = torch.randn(self.SHAPE)

        out = torch.empty_like(x)
        remover._dir_diff(x=x, mode=mode, out=out)
        lhs = (out * y).sum().item()

        target = torch.zeros_like(x)
        remover._adjoint_dir(target=target, q=y, mode=mode, a=1.0)
        rhs = (x * target).sum().item()

        assert lhs == pytest.approx(-rhs, abs=1e-5)

    def _check_gradient_adjoint(self, remover: UniversalStripeRemover, dim: int) -> None:
        torch.manual_seed(dim)
        x = torch.randn(self.SHAPE)
        y = torch.randn(self.SHAPE)

        fwd = torch.empty_like(x)
        remover._forward_diff(x=x, dim=dim, out=fwd)
        lhs = (fwd * y).sum().item()

        target = torch.zeros_like(x)
        remover._adjoint_1d(target=target, p=y, dim=dim, a=1.0)
        rhs = (x * target).sum().item()

        assert lhs == pytest.approx(-rhs, abs=1e-5)


class TestDirections:
    def test_default_directions_are_all_modes(self) -> None:
        remover = UniversalStripeRemover(device="cpu")
        assert remover.directions == (0, 1, 2, 3, 4)

    def test_subset_directions_preserve_shape(self) -> None:
        remover = UniversalStripeRemover(device="cpu", directions=[1, 4])
        img = torch.rand(24, 24)
        result = remover.process(image=img, iterations=5)
        assert remover.directions == (1, 4)
        assert result.shape == img.shape
        assert torch.isfinite(result).all()

    @pytest.mark.parametrize(
        "directions",
        [
            [],
            [0, 0],
            [-1],
            [5],
            [1.5],
            ["0"],
            {0},
            [[0]],
            [True],
        ],
    )
    def test_invalid_directions(self, directions: object) -> None:
        with pytest.raises(ValueError, match="directions"):
            UniversalStripeRemover(device="cpu", directions=directions)


class TestProcess:
    def test_grayscale_2d(self, remover: UniversalStripeRemover) -> None:
        img = torch.rand(32, 32)
        result = remover.process(image=img, iterations=10, proj=True)
        assert result.shape == (32, 32)
        assert result.dtype == torch.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_batch_3d(self, remover: UniversalStripeRemover) -> None:
        img = torch.rand(3, 32, 32)
        result = remover.process(image=img, iterations=10)
        assert result.shape == (3, 32, 32)
        assert result.dtype == torch.float32

    def test_numpy_input(self, remover: UniversalStripeRemover) -> None:
        img = np.random.default_rng(0).random((32, 32), dtype=np.float32)
        result = remover.process(image=img, iterations=10)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (32, 32)

    def test_invalid_shape(self, remover: UniversalStripeRemover) -> None:
        with pytest.raises(ValueError, match="shape"):
            remover.process(image=torch.rand(2, 3, 32, 32))

    def test_invalid_iterations(self, remover: UniversalStripeRemover) -> None:
        with pytest.raises(ValueError, match="iterations"):
            remover.process(image=torch.rand(32, 32), iterations=0)

    def test_invalid_tol(self, remover: UniversalStripeRemover) -> None:
        with pytest.raises(ValueError, match="tol"):
            remover.process(image=torch.rand(32, 32), tol=-1e-3)

    def test_invalid_non_finite(self, remover: UniversalStripeRemover) -> None:
        img = torch.rand(32, 32)
        img[0, 0] = torch.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            remover.process(image=img)

    def test_constant_image(self, remover: UniversalStripeRemover) -> None:
        img = torch.full((32, 32), 0.5)
        result = remover.process(image=img, iterations=20)
        assert torch.allclose(result, img, atol=1e-3)


class TestProcessTiled:
    def test_tiles_1_fallback(self, remover: UniversalStripeRemover) -> None:
        img = torch.rand(32, 32)
        result = remover.process_tiled(image=img, tiles=1, iterations=10)
        assert result.shape == (32, 32)

    def test_tiles_2(self, remover: UniversalStripeRemover) -> None:
        img = torch.rand(64, 64)
        result = remover.process_tiled(image=img, tiles=2, iterations=10, overlap=8)
        assert result.shape == (64, 64)
        assert result.dtype == torch.float32

    def test_tiles_overlap_clamped(self, remover: UniversalStripeRemover) -> None:
        img = torch.rand(32, 32)
        result = remover.process_tiled(image=img, tiles=4, iterations=5, overlap=10_000)
        assert result.shape == (32, 32)

    def test_invalid_batch(self, remover: UniversalStripeRemover) -> None:
        with pytest.raises(ValueError, match="shape"):
            remover.process_tiled(image=torch.rand(3, 32, 32), tiles=2)

    def test_invalid_tiles(self, remover: UniversalStripeRemover) -> None:
        with pytest.raises(ValueError, match="tiles"):
            remover.process_tiled(image=torch.rand(32, 32), tiles=0)

    def test_invalid_overlap(self, remover: UniversalStripeRemover) -> None:
        with pytest.raises(ValueError, match="overlap"):
            remover.process_tiled(image=torch.rand(32, 32), tiles=2, overlap=-1)

    def test_tile_mus_length_error_restores_mus(
        self, remover: UniversalStripeRemover
    ) -> None:
        original_mu1, original_mu2 = remover.mu1, remover.mu2

        with pytest.raises(ValueError, match="tile_mus"):
            remover.process_tiled(
                image=torch.rand(8, 8),
                tiles=2,
                iterations=1,
                overlap=0,
                tile_mus=[(0.1, 0.0017)],
            )

        assert remover.mu1 == original_mu1
        assert remover.mu2 == original_mu2

    @pytest.mark.parametrize(
        "tile_mus",
        [
            [0.1, 0.0017, 0.2, 0.003],
            [(0.1, 0.0017, 0.2)] * 4,
            [("bad", 0.0017)] * 4,
            [(True, 0.0017)] * 4,
            [(np.nan, 0.0017)] * 4,
            [(0.1, np.inf)] * 4,
        ],
    )
    def test_malformed_tile_mus_raise_value_error(
        self,
        remover: UniversalStripeRemover,
        tile_mus: object,
    ) -> None:
        with pytest.raises(ValueError, match="tile_mus"):
            remover.process_tiled(
                image=torch.rand(8, 8),
                tiles=2,
                iterations=1,
                overlap=0,
                tile_mus=tile_mus,  # type: ignore[arg-type]
            )

    def test_tile_mus_validated_when_tiles_one(
        self,
        remover: UniversalStripeRemover,
    ) -> None:
        with pytest.raises(ValueError, match="tile_mus"):
            remover.process_tiled(
                image=torch.rand(8, 8),
                tiles=1,
                iterations=1,
                tile_mus=[(np.nan, 0.0017)],
            )

    @pytest.mark.parametrize(
        ("shape", "tile_mus"),
        [
            ((1, 8), [(float("nan"), 0.1)] * 16),
            ((3, 8), [0.1] * 16),
            ((3, 8), [(True, 0.1)] * 16),
        ],
    )
    def test_malformed_tile_mus_raise_on_fallback_dimensions(
        self,
        remover: UniversalStripeRemover,
        shape: tuple[int, int],
        tile_mus: object,
    ) -> None:
        with pytest.raises(ValueError, match="tile_mus"):
            remover.process_tiled(
                image=torch.rand(shape),
                tiles=4,
                iterations=1,
                overlap=0,
                tile_mus=tile_mus,  # type: ignore[arg-type]
            )

    def test_tile_mus_restore_after_tile_processing_error(
        self,
        remover: UniversalStripeRemover,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_mu1, original_mu2 = remover.mu1, remover.mu2

        def fail_solve(**_: object) -> torch.Tensor:
            raise RuntimeError("forced tile failure")

        monkeypatch.setattr(remover, "_solve", fail_solve)

        with pytest.raises(RuntimeError, match="forced tile failure"):
            remover.process_tiled(
                image=torch.rand(8, 8),
                tiles=2,
                iterations=1,
                overlap=0,
                tile_mus=[(0.1, 0.0017)] * 4,
            )

        assert remover.mu1 == original_mu1
        assert remover.mu2 == original_mu2

    def test_tile_mus_processes_tiles_in_one_batch(
        self,
        remover: UniversalStripeRemover,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[dict[str, object]] = []

        def fake_solve(**kwargs: object) -> torch.Tensor:
            data = kwargs["data"]
            mu1 = kwargs.get("mu1")
            mu2 = kwargs.get("mu2")
            assert isinstance(data, torch.Tensor)
            assert isinstance(mu1, torch.Tensor)
            assert isinstance(mu2, torch.Tensor)
            calls.append(
                {
                    "shape": tuple(data.shape),
                    "mu1": mu1.detach().cpu().clone(),
                    "mu2": mu2.detach().cpu().clone(),
                }
            )
            return data.cpu()

        monkeypatch.setattr(remover, "_solve", fake_solve)

        tile_mus = [
            (0.10, 0.0017),
            (0.20, 0.0030),
            (0.30, 0.0070),
            (0.40, 0.0170),
        ]
        result = remover.process_tiled(
            image=torch.rand(8, 8),
            tiles=2,
            iterations=1,
            overlap=0,
            tile_mus=tile_mus,
        )

        assert result.shape == (8, 8)
        assert len(calls) == 1
        assert calls[0]["shape"][0] == 4
        assert torch.allclose(
            calls[0]["mu1"].reshape(-1),
            torch.tensor([0.10, 0.20, 0.30, 0.40]),
        )
        assert torch.allclose(
            calls[0]["mu2"].reshape(-1),
            torch.tensor([0.0017, 0.0030, 0.0070, 0.0170]),
        )


class TestAdaptiveEstimator:
    def test_vertical_stripes_select_mode_zero(self) -> None:
        img = np.zeros((64, 64), dtype=np.float64)
        img[:, 12] = 1.0
        img[:, 32] = 0.8
        params = estimate_adaptive_params(img)
        assert params.directions[0] == 0
        assert params.mu1 >= 0.33
        assert 0.10 <= params.mu1 <= 0.50
        assert 0.0017 <= params.mu2 <= 0.017

    def test_faint_coherent_stripes_get_default_strength(self) -> None:
        rng = np.random.default_rng(1234)
        h = w = 128
        x = np.linspace(0, 1, w).reshape(1, -1)
        y = np.linspace(0, 1, h).reshape(-1, 1)
        img = (
            0.25
            + 0.35 * x
            + 0.12 * y
            + 0.04 * np.sin(2 * np.pi * x * 3) * np.sin(2 * np.pi * y * 2)
        )
        img += rng.normal(0, 0.003, img.shape)
        img = (img - img.min()) / (img.max() - img.min())
        for col, amp in [(24, 0.015), (57, -0.012), (90, 0.0105), (112, -0.009)]:
            img[:, col] = np.clip(img[:, col] + amp, 0.0, 1.0)

        params = estimate_adaptive_params(img)

        assert params.directions[0] == 0
        assert params.mu1 >= 0.33

    def test_estimator_is_deterministic(self) -> None:
        rng = np.random.default_rng(14)
        img = rng.random((48, 48))
        p1 = estimate_adaptive_params(img)
        p2 = estimate_adaptive_params(img)
        assert p1 == p2

    def test_estimator_normalizes_affine_intensity_scale(self) -> None:
        rng = np.random.default_rng(16)
        img = rng.random((48, 48))
        img[:, 10] += 0.7
        img = np.clip(img, 0.0, 1.0)

        p1 = estimate_adaptive_params(img)
        p2 = estimate_adaptive_params(img * 100.0 + 20.0)

        assert p1.directions == p2.directions
        assert p1.mu1 == pytest.approx(p2.mu1)
        assert p1.mu2 == pytest.approx(p2.mu2)
        assert p1.confidence == pytest.approx(p2.confidence)

    def test_direction_support_uses_relative_scores_without_cutoffs(self) -> None:
        from destripe.adaptive import directions

        scores = {0: 0.20, 1: 0.19, 2: -1.0, 3: -1.0, 4: -1.0}
        weights = directions.selection_weights(scores)

        assert directions.select_directions_from_weights(weights) == (0, 1)

    def test_flat_direction_scores_do_not_select_all_modes(self) -> None:
        from destripe.adaptive import directions

        scores = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
        weights = directions.selection_weights(scores)

        assert directions.select_directions_from_weights(weights) == (0,)

    def test_fixed_directions_do_not_add_count_penalty(self) -> None:
        img = np.zeros((64, 64), dtype=np.float64)
        img[:, 12] = 1.0
        img[:, 32] = 0.8

        single = estimate_adaptive_params(img, fixed_directions=(0,))
        multiple = estimate_adaptive_params(img, fixed_directions=(0, 1))

        assert single.directions == (0,)
        assert multiple.directions == (0, 1)
        assert single.mu1 == pytest.approx(multiple.mu1)
        assert single.mu2 == pytest.approx(multiple.mu2)

    @pytest.mark.parametrize("shape", [(1, 1), (3, 8), (8, 3)])
    def test_estimator_handles_small_arrays(self, shape: tuple[int, int]) -> None:
        img = np.random.default_rng(17).random(shape)
        params = estimate_adaptive_params(img)

        assert params.directions
        assert 0.10 <= params.mu1 <= 0.50
        assert 0.0017 <= params.mu2 <= 0.017

    @pytest.mark.parametrize(
        "fixed_directions",
        [
            (),
            (0, 0),
            (-1,),
            (5,),
            (1.5,),
            ("0",),
            (True,),
        ],
    )
    def test_invalid_fixed_directions(self, fixed_directions: object) -> None:
        with pytest.raises(ValueError, match="directions"):
            estimate_adaptive_params(
                np.random.default_rng(18).random((8, 8)),
                fixed_directions=fixed_directions,  # type: ignore[arg-type]
            )

    def test_tile_mu_smoothing_preserves_shape(self) -> None:
        from destripe.adaptive import smooth_tile_mus

        mus = np.array(
            [
                [[0.10, 0.0017], [0.50, 0.017]],
                [[0.33, 0.0030], [0.40, 0.007]],
            ],
            dtype=np.float64,
        )
        smoothed = smooth_tile_mus(mus)
        assert smoothed.shape == mus.shape
        assert np.all(smoothed[..., 0] >= 0.10)
        assert np.all(smoothed[..., 0] <= 0.50)
        assert np.all(smoothed[..., 1] >= 0.0017)
        assert np.all(smoothed[..., 1] <= 0.017)


class TestDestripe:
    def test_grayscale_float64(self, gray_image: np.ndarray) -> None:
        result = destripe(gray_image, iterations=20)
        assert result.shape == gray_image.shape
        assert result.dtype == gray_image.dtype

    def test_adaptive_uses_estimated_parameters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from destripe import ops
        from destripe.adaptive import AdaptiveParams

        calls = {}

        def fake_estimate(
            gray: np.ndarray, *, fixed_directions=None
        ) -> AdaptiveParams:
            calls["shape"] = gray.shape
            calls["fixed_directions"] = fixed_directions
            return AdaptiveParams(
                directions=(0,), mu1=0.10, mu2=0.0017, confidence=1.0
            )

        monkeypatch.setattr(
            ops, "estimate_adaptive_params", fake_estimate, raising=False
        )
        img = np.random.default_rng(17).random((24, 24))
        result = destripe(img, adaptive=True, iterations=5)

        assert result.shape == img.shape
        assert calls == {"shape": img.shape, "fixed_directions": None}

    def test_adaptive_grayscale_float64(self, gray_image: np.ndarray) -> None:
        result = destripe(gray_image, adaptive=True, iterations=10)
        assert result.shape == gray_image.shape
        assert result.dtype == gray_image.dtype

    def test_adaptive_rgb(self) -> None:
        img = np.random.default_rng(15).random((32, 32, 3)).astype(np.float32)
        result = destripe(img, adaptive=True, iterations=10)
        assert result.shape == img.shape
        assert result.dtype == img.dtype

    def test_adaptive_tiled_passes_tile_mus(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from destripe import ops
        from destripe.adaptive import AdaptiveParams

        seen = {}

        def fake_estimate(
            gray: np.ndarray, *, fixed_directions=None
        ) -> AdaptiveParams:
            return AdaptiveParams(
                directions=(0,), mu1=0.33, mu2=0.003, confidence=1.0
            )

        original = UniversalStripeRemover.process_tiled

        def spy_process_tiled(self, *args, **kwargs):
            seen["tile_mus"] = kwargs.get("tile_mus")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(ops, "estimate_adaptive_params", fake_estimate)
        monkeypatch.setattr(UniversalStripeRemover, "process_tiled", spy_process_tiled)

        img = np.random.default_rng(18).random((32, 32))
        result = destripe(img, adaptive=True, iterations=5, tiles=2, overlap=4)

        assert result.shape == img.shape
        assert seen["tile_mus"] is not None
        assert len(seen["tile_mus"]) == 4

    def test_adaptive_tiled_dtype_shape(self) -> None:
        img = np.random.default_rng(16).random((48, 40, 3)).astype(np.float32)
        result = destripe(img, adaptive=True, iterations=5, tiles=3, overlap=6)
        assert result.shape == img.shape
        assert result.dtype == img.dtype

    def test_process_size_runs_solver_on_resized_grayscale(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = []

        class FakeRemover:
            def __init__(self, **_: object) -> None:
                pass

            def process_tiled(self, image: np.ndarray, **_: object) -> torch.Tensor:
                calls.append(image.shape)
                return torch.as_tensor(image - 0.25)

        monkeypatch.setattr(destripe_ops, "UniversalStripeRemover", FakeRemover)

        img = np.linspace(0.0, 1.0, 16 * 20, dtype=np.float64).reshape(16, 20)
        result = destripe(img, process_size=10, iterations=1, proj=False)

        assert calls == [(8, 10)]
        assert result.shape == img.shape
        assert np.allclose(result, img - 0.25)

    def test_process_size_none_uses_original_resolution(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = []

        class FakeRemover:
            def __init__(self, **_: object) -> None:
                pass

            def process_tiled(self, image: np.ndarray, **_: object) -> torch.Tensor:
                calls.append(image.shape)
                return torch.as_tensor(image)

        monkeypatch.setattr(destripe_ops, "UniversalStripeRemover", FakeRemover)

        img = np.random.default_rng(22).random((100, 100))
        result = destripe(img, process_size=None, iterations=1)

        assert calls == [(100, 100)]
        assert result.shape == img.shape
        assert np.allclose(result, img)

    def test_process_size_larger_than_input_uses_original_resolution(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = []

        class FakeRemover:
            def __init__(self, **_: object) -> None:
                pass

            def process_tiled(self, image: np.ndarray, **_: object) -> torch.Tensor:
                calls.append(image.shape)
                return torch.as_tensor(image)

        monkeypatch.setattr(destripe_ops, "UniversalStripeRemover", FakeRemover)

        img = np.random.default_rng(24).random((80, 100))
        result = destripe(img, process_size=128, iterations=1)

        assert calls == [(80, 100)]
        assert result.shape == img.shape
        assert np.allclose(result, img)

    def test_resize_2d_supports_lanczos_mode(self) -> None:
        img = np.random.default_rng(23).random((11, 13))
        result = preprocess.resize_2d(img, size=(7, 9), mode="lanczos")

        assert result.shape == (7, 9)
        assert result.dtype == np.float64
        assert np.isfinite(result).all()

    def test_process_size_subtracts_resized_rgb_stripe(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class FakeRemover:
            def __init__(self, **_: object) -> None:
                pass

            def process_tiled(self, image: np.ndarray, **_: object) -> torch.Tensor:
                return torch.as_tensor(image - 0.125)

        monkeypatch.setattr(destripe_ops, "UniversalStripeRemover", FakeRemover)

        img = np.linspace(0.0, 1.0, 12 * 18 * 3, dtype=np.float64).reshape(12, 18, 3)
        result = destripe(img, process_size=9, iterations=1, proj=False)

        assert result.shape == img.shape
        assert np.allclose(result, img - 0.125)

    def test_adaptive_process_size_estimates_resized_gray(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from destripe.adaptive import AdaptiveParams

        calls = []

        def fake_estimate(
            gray: np.ndarray, *, fixed_directions=None
        ) -> AdaptiveParams:
            calls.append((gray.shape, fixed_directions))
            return AdaptiveParams(
                directions=(0,), mu1=0.33, mu2=0.003, confidence=1.0
            )

        class FakeRemover:
            directions = (0,)

            def __init__(self, **_: object) -> None:
                pass

            def process_tiled(self, image: np.ndarray, **_: object) -> torch.Tensor:
                return torch.as_tensor(image)

        monkeypatch.setattr(destripe_ops, "estimate_adaptive_params", fake_estimate)
        monkeypatch.setattr(destripe_ops, "UniversalStripeRemover", FakeRemover)

        img = np.random.default_rng(20).random((20, 30))
        result = destripe(img, adaptive=True, process_size=15, iterations=1)

        assert result.shape == img.shape
        assert calls == [((10, 15), None)]

    @pytest.mark.parametrize("process_size", [0, -1, 1.1, float("nan"), True])
    def test_invalid_process_size(self, process_size: object) -> None:
        with pytest.raises(ValueError, match="process_size"):
            destripe(
                np.random.default_rng(21).random((8, 8)),
                process_size=process_size,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("shape", [(1, 8), (3, 8)])
    def test_adaptive_tiled_tiny_images_return_shape_dtype(
        self, shape: tuple[int, int]
    ) -> None:
        img = np.random.default_rng(19).random(shape)
        result = destripe(img, adaptive=True, tiles=4, iterations=1)
        assert result.shape == img.shape
        assert result.dtype == img.dtype

    def test_adaptive_constant_returns_copy(self) -> None:
        img = np.full((16, 16), 12, dtype=np.uint8)
        result = destripe(img, adaptive=True)
        assert np.array_equal(result, img)
        assert result is not img

    def test_grayscale_uint8(self) -> None:
        img = (np.random.default_rng(1).random((32, 32)) * 255).astype(np.uint8)
        result = destripe(img, iterations=20)
        assert result.dtype == np.uint8
        assert result.shape == img.shape

    def test_rgb(self) -> None:
        img = np.random.default_rng(2).random((32, 32, 3)).astype(np.float64)
        result = destripe(img, iterations=20)
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.float64

    def test_rgb_tiled_dtype_shape(self) -> None:
        img = np.random.default_rng(3).random((48, 40, 3)).astype(np.float32)
        result = destripe(img, iterations=10, tiles=3, overlap=6)
        assert result.shape == img.shape
        assert result.dtype == img.dtype

    def test_single_channel(self) -> None:
        img = np.random.default_rng(4).random((32, 32, 1)).astype(np.float32)
        result = destripe(img, iterations=20)
        assert result.shape == (32, 32, 1)
        assert result.dtype == np.float32

    def test_manual_directions(self, gray_image: np.ndarray) -> None:
        result = destripe(gray_image, directions=[0], iterations=10)
        assert result.shape == gray_image.shape
        assert result.dtype == gray_image.dtype

    def test_invalid_manual_directions(self) -> None:
        with pytest.raises(ValueError, match="directions"):
            destripe(np.random.default_rng(12).random((16, 16)), directions=[5])

    def test_adaptive_warns_when_manual_values_are_ignored(self) -> None:
        img = np.random.default_rng(13).random((24, 24))
        with pytest.warns(UserWarning, match="adaptive=True ignores"):
            result = destripe(
                img,
                adaptive=True,
                directions=[0],
                mu1=0.5,
                mu2=0.017,
                iterations=5,
            )
        assert result.shape == img.shape

    def test_manual_arguments_are_forwarded_to_remover(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, object]] = []

        class FakeRemover:
            def __init__(
                self,
                mu1: float,
                mu2: float,
                device: torch.device | str | None = None,
                directions: object = None,
            ) -> None:
                calls.append(
                    {
                        "mu1": mu1,
                        "mu2": mu2,
                        "device": device,
                        "directions": directions,
                    }
                )

            def process_tiled(self, image: np.ndarray, **_: object) -> torch.Tensor:
                return torch.as_tensor(image)

        monkeypatch.setattr(destripe_ops, "UniversalStripeRemover", FakeRemover)

        img = np.random.default_rng(14).random((8, 8))
        result = destripe(
            img,
            mu1=0.5,
            mu2=0.017,
            directions=[1, 4],
            iterations=1,
            device="cpu",
        )

        assert result.shape == img.shape
        assert calls == [
            {
                "mu1": 0.5,
                "mu2": 0.017,
                "device": "cpu",
                "directions": [1, 4],
            }
        ]

    def test_adaptive_ignores_manual_arguments_for_remover(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from destripe.adaptive import AdaptiveParams

        calls: list[dict[str, object]] = []
        estimate_calls: list[dict[str, object]] = []

        def fake_estimate(
            gray: np.ndarray, *, fixed_directions=None
        ) -> AdaptiveParams:
            estimate_calls.append(
                {"shape": gray.shape, "fixed_directions": fixed_directions}
            )
            return AdaptiveParams(
                directions=(0,), mu1=0.10, mu2=0.0017, confidence=1.0
            )

        class FakeRemover:
            def __init__(
                self,
                mu1: float,
                mu2: float,
                device: torch.device | str | None = None,
                directions: object = None,
            ) -> None:
                calls.append(
                    {
                        "mu1": mu1,
                        "mu2": mu2,
                        "device": device,
                        "directions": directions,
                    }
                )

            def process_tiled(self, image: np.ndarray, **_: object) -> torch.Tensor:
                return torch.as_tensor(image)

        monkeypatch.setattr(destripe_ops, "UniversalStripeRemover", FakeRemover)
        monkeypatch.setattr(destripe_ops, "estimate_adaptive_params", fake_estimate)

        img = np.random.default_rng(15).random((8, 8))
        with pytest.warns(UserWarning, match="adaptive=True ignores"):
            result = destripe(
                img,
                adaptive=True,
                mu1=0.5,
                mu2=0.017,
                directions=[1, 4],
                iterations=1,
            )

        assert result.shape == img.shape
        assert calls == [
            {
                "mu1": 0.10,
                "mu2": 0.0017,
                "device": None,
                "directions": (0,),
            }
        ]
        assert estimate_calls == [{"shape": img.shape, "fixed_directions": None}]

    def test_constant_returns_copy(self) -> None:
        img = np.full((32, 32), 128, dtype=np.uint8)
        result = destripe(img)
        assert np.array_equal(result, img)
        assert result is not img

    def test_invalid_channels(self) -> None:
        with pytest.raises(ValueError, match="C in"):
            destripe(np.random.default_rng(5).random((32, 32, 4)))

    def test_invalid_ndim(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            destripe(np.random.default_rng(6).random(32))

    def test_invalid_iterations(self) -> None:
        with pytest.raises(ValueError, match="iterations"):
            destripe(np.random.default_rng(7).random((8, 8)), iterations=0)

    def test_invalid_tol(self) -> None:
        with pytest.raises(ValueError, match="tol"):
            destripe(np.random.default_rng(8).random((8, 8)), tol=-1)

    def test_invalid_tiles(self) -> None:
        with pytest.raises(ValueError, match="tiles"):
            destripe(np.random.default_rng(9).random((8, 8)), tiles=0)

    def test_invalid_overlap(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            destripe(np.random.default_rng(10).random((8, 8)), overlap=-1)

    def test_invalid_non_finite(self) -> None:
        img = np.random.default_rng(11).random((8, 8))
        img[0, 0] = np.inf
        with pytest.raises(ValueError, match="NaN or Inf"):
            destripe(img)

    def test_reproducible_output_for_fixed_input(self, gray_image: np.ndarray) -> None:
        out1 = destripe(gray_image, iterations=15, tiles=2, overlap=4)
        out2 = destripe(gray_image, iterations=15, tiles=2, overlap=4)
        assert np.allclose(out1, out2, atol=1e-8)

    def test_tiled(self, gray_image: np.ndarray) -> None:
        result = destripe(gray_image, iterations=20, tiles=2, overlap=4)
        assert result.shape == gray_image.shape
