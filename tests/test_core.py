import numpy as np
import pytest
import torch

from destripe import UniversalStripeRemover, destripe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def remover():
    return UniversalStripeRemover(device="cpu")


@pytest.fixture()
def gray_image():
    """32x32 smooth gradient with synthetic horizontal stripe."""
    rng = np.random.default_rng(42)
    base = np.linspace(0, 1, 32).reshape(1, -1).repeat(32, axis=0)
    stripe = np.zeros_like(base)
    stripe[10, :] = 0.3
    stripe[20, :] = -0.2
    return np.clip(base + stripe + rng.normal(0, 0.01, base.shape), 0, 1)


# ---------------------------------------------------------------------------
# Forward / Adjoint consistency
# ---------------------------------------------------------------------------


class TestAdjointConsistency:
    """<Dx, y> must equal <x, D^T y> for all operator pairs."""

    SHAPE = (1, 16, 16)

    def test_forward_diff_adjoint_dim1(self, remover):
        self._check_gradient_adjoint(remover, dim=1)

    def test_forward_diff_adjoint_dim2(self, remover):
        self._check_gradient_adjoint(remover, dim=2)

    @pytest.mark.parametrize("mode", range(5))
    def test_dir_diff_adjoint(self, remover, mode):
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

    def _check_gradient_adjoint(self, remover, dim: int):
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


# ---------------------------------------------------------------------------
# UniversalStripeRemover.process
# ---------------------------------------------------------------------------


class TestProcess:
    def test_grayscale_2d(self, remover):
        img = torch.rand(32, 32)
        result = remover.process(image=img, iterations=10, proj=True)
        assert result.shape == (32, 32)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_batch_3d(self, remover):
        img = torch.rand(3, 32, 32)
        result = remover.process(image=img, iterations=10)
        assert result.shape == (3, 32, 32)

    def test_numpy_input(self, remover):
        img = np.random.rand(32, 32).astype(np.float32)
        result = remover.process(image=img, iterations=10)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (32, 32)

    def test_invalid_shape(self, remover):
        with pytest.raises(ValueError):
            remover.process(image=torch.rand(2, 3, 32, 32))

    def test_constant_image(self, remover):
        img = torch.full((32, 32), 0.5)
        result = remover.process(image=img, iterations=20)
        assert torch.allclose(result, img, atol=1e-3)


# ---------------------------------------------------------------------------
# UniversalStripeRemover.process_tiled
# ---------------------------------------------------------------------------


class TestProcessTiled:
    def test_tiles_1_fallback(self, remover):
        img = torch.rand(32, 32)
        result = remover.process_tiled(image=img, tiles=1, iterations=10)
        assert result.shape == (32, 32)

    def test_tiles_2(self, remover):
        img = torch.rand(64, 64)
        result = remover.process_tiled(image=img, tiles=2, iterations=10, overlap=8)
        assert result.shape == (64, 64)

    def test_invalid_batch(self, remover):
        with pytest.raises(ValueError):
            remover.process_tiled(image=torch.rand(3, 32, 32), tiles=2)


# ---------------------------------------------------------------------------
# destripe convenience function
# ---------------------------------------------------------------------------


class TestDestripe:
    def test_grayscale_float64(self, gray_image):
        result = destripe(gray_image, iterations=20)
        assert result.shape == gray_image.shape
        assert result.dtype == gray_image.dtype

    def test_grayscale_uint8(self):
        img = (np.random.rand(32, 32) * 255).astype(np.uint8)
        result = destripe(img, iterations=20)
        assert result.dtype == np.uint8
        assert result.shape == img.shape

    def test_rgb(self):
        img = np.random.rand(32, 32, 3).astype(np.float64)
        result = destripe(img, iterations=20)
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.float64

    def test_single_channel(self):
        img = np.random.rand(32, 32, 1).astype(np.float32)
        result = destripe(img, iterations=20)
        assert result.shape == (32, 32, 1)

    def test_constant_returns_copy(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        result = destripe(img)
        assert np.array_equal(result, img)
        assert result is not img

    def test_invalid_channels(self):
        with pytest.raises(ValueError):
            destripe(np.random.rand(32, 32, 4))

    def test_invalid_ndim(self):
        with pytest.raises(ValueError):
            destripe(np.random.rand(32))

    def test_tiled(self, gray_image):
        result = destripe(gray_image, iterations=20, tiles=2, overlap=4)
        assert result.shape == gray_image.shape
