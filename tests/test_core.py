import numpy as np
import pytest
import torch

from destripe import UniversalStripeRemover, destripe


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
