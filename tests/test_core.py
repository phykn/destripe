import numpy as np
import pytest
import torch

import destripe.ops as destripe_ops
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


class TestAdaptiveEstimator:
    def test_vertical_stripes_select_mode_zero(self) -> None:
        img = np.zeros((64, 64), dtype=np.float64)
        img[:, 12] = 1.0
        img[:, 32] = 0.8
        params = estimate_adaptive_params(img)
        assert params.directions[0] == 0
        assert 0.10 <= params.mu1 <= 0.50
        assert 0.0017 <= params.mu2 <= 0.017

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

    def test_fixed_directions_drive_mu2_ambiguity(self) -> None:
        img = np.zeros((64, 64), dtype=np.float64)
        img[:, 12] = 1.0
        img[:, 32] = 0.8

        single = estimate_adaptive_params(img, fixed_directions=(0,))
        multiple = estimate_adaptive_params(img, fixed_directions=(0, 1))

        assert single.directions == (0,)
        assert multiple.directions == (0, 1)
        assert single.mu2 == pytest.approx(0.0017)
        assert multiple.mu2 > single.mu2

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
                "mu1": 0.33,
                "mu2": 0.003,
                "device": None,
                "directions": None,
            }
        ]

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
