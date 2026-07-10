import inspect

import numpy as np
import pytest
import torch

import destripe.ops as destripe_ops
from destripe import preprocess
from destripe import UniversalStripeRemover, destripe
from destripe.core import operators


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

    def test_forward_diff_adjoint_dim1(self) -> None:
        self._check_gradient_adjoint(dim=1)

    def test_forward_diff_adjoint_dim2(self) -> None:
        self._check_gradient_adjoint(dim=2)

    @pytest.mark.parametrize("mode", range(5))
    def test_dir_diff_adjoint(self, mode: int) -> None:
        torch.manual_seed(mode)
        x = torch.randn(self.SHAPE)
        y = torch.randn(self.SHAPE)

        out = torch.empty_like(x)
        operators.dir_diff(x=x, mode=mode, out=out)
        lhs = (out * y).sum().item()

        target = torch.zeros_like(x)
        operators.adjoint_dir(target=target, q=y, mode=mode, scale=1.0)
        rhs = (x * target).sum().item()

        assert lhs == pytest.approx(-rhs, abs=1e-5)

    def _check_gradient_adjoint(self, dim: int) -> None:
        torch.manual_seed(dim)
        x = torch.randn(self.SHAPE)
        y = torch.randn(self.SHAPE)

        fwd = torch.empty_like(x)
        operators.forward_diff(x=x, dim=dim, out=fwd)
        lhs = (fwd * y).sum().item()

        target = torch.zeros_like(x)
        operators.adjoint_1d(target=target, p=y, dim=dim, scale=1.0)
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


class TestConfiguration:
    @pytest.mark.parametrize("name", ["mu1", "mu2"])
    @pytest.mark.parametrize("value", [0.0, -0.1, np.nan, np.inf, True, "bad"])
    def test_invalid_regularization_weight(self, name: str, value: object) -> None:
        kwargs = {name: value, "device": "cpu"}

        with pytest.raises(ValueError, match=name):
            UniversalStripeRemover(**kwargs)  # type: ignore[arg-type]


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

    @pytest.mark.parametrize("iterations", [True, 1.5])
    def test_invalid_iteration_type(
        self,
        remover: UniversalStripeRemover,
        iterations: object,
    ) -> None:
        with pytest.raises(ValueError, match="iterations"):
            remover.process(
                image=torch.rand(32, 32),
                iterations=iterations,  # type: ignore[arg-type]
            )

    def test_invalid_tol(self, remover: UniversalStripeRemover) -> None:
        with pytest.raises(ValueError, match="tol"):
            remover.process(image=torch.rand(32, 32), tol=-1e-3)

    @pytest.mark.parametrize("tol", [np.nan, np.inf, True, "bad"])
    def test_invalid_tol_value(
        self,
        remover: UniversalStripeRemover,
        tol: object,
    ) -> None:
        with pytest.raises(ValueError, match="tol"):
            remover.process(
                image=torch.rand(32, 32),
                tol=tol,  # type: ignore[arg-type]
            )

    def test_invalid_non_finite(self, remover: UniversalStripeRemover) -> None:
        img = torch.rand(32, 32)
        img[0, 0] = torch.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            remover.process(image=img)

    def test_constant_image(self, remover: UniversalStripeRemover) -> None:
        img = torch.full((32, 32), 0.5)
        result = remover.process(image=img, iterations=20)
        assert torch.allclose(result, img, atol=1e-3)

    @pytest.mark.parametrize("shape", [(1, 8), (8, 1)])
    def test_single_pixel_axis_returns_copy(
        self,
        remover: UniversalStripeRemover,
        shape: tuple[int, int],
    ) -> None:
        img = torch.rand(shape)

        result = remover.process(image=img, iterations=5)

        assert result is not img
        assert torch.equal(result, img)


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

    @pytest.mark.parametrize("overlap", [0.5, True, np.nan])
    def test_invalid_overlap_type(
        self,
        remover: UniversalStripeRemover,
        overlap: object,
    ) -> None:
        with pytest.raises(ValueError, match="overlap"):
            remover.process_tiled(
                image=torch.rand(32, 32),
                tiles=2,
                overlap=overlap,  # type: ignore[arg-type]
            )

class TestDestripe:
    def test_destripe_has_only_automatic_options(self) -> None:
        signature = inspect.signature(destripe)

        assert tuple(signature.parameters) == ("image", "process_size", "proj")

    @pytest.mark.parametrize(
        "old_name",
        [
            "adaptive",
            "mu1",
            "mu2",
            "iterations",
            "tol",
            "tiles",
            "overlap",
            "device",
            "verbose",
            "directions",
        ],
    )
    def test_removed_wrapper_arguments_raise_type_error(self, old_name: str) -> None:
        with pytest.raises(TypeError):
            destripe(np.ones((8, 8)), **{old_name: 1})

    def test_wrapper_calls_automatic_clean_exactly_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[tuple[tuple[int, int], bool]] = []

        def fake_automatic_clean(gray: np.ndarray, *, proj: bool) -> object:
            calls.append((gray.shape, proj))
            return type("Result", (), {"clean": gray - 0.1})()

        monkeypatch.setattr(destripe_ops, "automatic_clean", fake_automatic_clean)

        image = np.linspace(0.0, 1.0, 12 * 18).reshape(12, 18)
        result = destripe(image, process_size=9, proj=False)

        assert calls == [((6, 9), False)]
        assert result.shape == image.shape

    def test_wrapper_normalizes_input_and_restores_float_dtype(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: dict[str, object] = {}

        def fake_automatic_clean(gray: np.ndarray, *, proj: bool) -> object:
            seen["dtype"] = gray.dtype
            seen["minimum"] = float(gray.min())
            seen["maximum"] = float(gray.max())
            return type("Result", (), {"clean": gray.copy()})()

        monkeypatch.setattr(destripe_ops, "automatic_clean", fake_automatic_clean)

        image = np.linspace(20.0, 120.0, 16 * 16, dtype=np.float32).reshape(16, 16)
        result = destripe(image)

        assert seen == {
            "dtype": np.dtype(np.float64),
            "minimum": 0.0,
            "maximum": 1.0,
        }
        assert result.dtype == image.dtype
        assert np.allclose(result, image)

    def test_grayscale_float64(self, gray_image: np.ndarray) -> None:
        result = destripe(gray_image)

        assert result.shape == gray_image.shape
        assert result.dtype == gray_image.dtype

    def test_process_size_resizes_automatic_input_and_correction_back(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[tuple[int, int]] = []

        def fake_automatic_clean(gray: np.ndarray, *, proj: bool) -> object:
            calls.append(gray.shape)
            return type("Result", (), {"clean": gray - 0.25})()

        monkeypatch.setattr(destripe_ops, "automatic_clean", fake_automatic_clean)

        image = np.linspace(0.0, 1.0, 16 * 20, dtype=np.float64).reshape(16, 20)
        result = destripe(image, process_size=10, proj=False)

        assert calls == [(8, 10)]
        assert result.shape == image.shape
        assert np.allclose(result, image - 0.25)

    @pytest.mark.parametrize(
        ("process_size", "expected_shape"),
        [(None, (80, 100)), (128, (80, 100))],
    )
    def test_process_size_uses_original_resolution_when_not_downsampling(
        self,
        monkeypatch: pytest.MonkeyPatch,
        process_size: int | None,
        expected_shape: tuple[int, int],
    ) -> None:
        calls: list[tuple[int, int]] = []

        def fake_automatic_clean(gray: np.ndarray, *, proj: bool) -> object:
            calls.append(gray.shape)
            return type("Result", (), {"clean": gray.copy()})()

        monkeypatch.setattr(destripe_ops, "automatic_clean", fake_automatic_clean)

        image = np.random.default_rng(24).random((80, 100))
        result = destripe(image, process_size=process_size)

        assert calls == [expected_shape]
        assert result.shape == image.shape
        assert np.allclose(result, image)

    def test_resize_to_shape_resizes_2d_images(self) -> None:
        image = np.random.default_rng(23).random((11, 13))
        result = preprocess.resize_to_shape(image, shape=(7, 9))

        assert result.shape == (7, 9)
        assert result.dtype == np.float64
        assert np.isfinite(result).all()

    def test_resize_to_shape_uses_cubic_interpolation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[dict[str, object]] = []

        def fake_resize(
            array: np.ndarray,
            *,
            dsize: tuple[int, int],
            interpolation: int,
        ) -> np.ndarray:
            calls.append({"dsize": dsize, "interpolation": interpolation})
            return np.zeros((dsize[1], dsize[0]), dtype=np.float64)

        monkeypatch.setattr(preprocess.cv2, "resize", fake_resize)

        result = preprocess.resize_to_shape(
            np.random.default_rng(25).random((11, 13)),
            shape=(7, 9),
        )

        assert result.shape == (7, 9)
        assert calls == [
            {"dsize": (9, 7), "interpolation": preprocess.cv2.INTER_CUBIC}
        ]

    def test_rgb_uses_one_shared_resized_correction(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_automatic_clean(gray: np.ndarray, *, proj: bool) -> object:
            return type("Result", (), {"clean": gray - 0.125})()

        monkeypatch.setattr(destripe_ops, "automatic_clean", fake_automatic_clean)

        image = np.linspace(0.0, 1.0, 12 * 18 * 3, dtype=np.float64).reshape(
            12, 18, 3
        )
        result = destripe(image, process_size=9, proj=False)

        assert result.shape == image.shape
        assert np.allclose(result, image - 0.125)

    def test_projection_is_forwarded_and_applied(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[bool] = []

        def fake_automatic_clean(gray: np.ndarray, *, proj: bool) -> object:
            calls.append(proj)
            return type("Result", (), {"clean": gray - 0.75})()

        monkeypatch.setattr(destripe_ops, "automatic_clean", fake_automatic_clean)

        image = np.linspace(0.0, 1.0, 8 * 8).reshape(8, 8)
        projected = destripe(image, proj=True)
        unprojected = destripe(image, proj=False)

        assert calls == [True, False]
        assert projected.min() >= 0.0
        assert projected.max() <= 1.0
        assert unprojected.min() < 0.0

    @pytest.mark.parametrize("process_size", [0, -1, 1.1, float("nan"), True])
    def test_invalid_process_size(self, process_size: object) -> None:
        with pytest.raises(ValueError, match="process_size"):
            destripe(
                np.random.default_rng(21).random((8, 8)),
                process_size=process_size,  # type: ignore[arg-type]
            )

    def test_grayscale_uint8(self) -> None:
        image = (np.random.default_rng(1).random((32, 32)) * 255).astype(np.uint8)
        result = destripe(image)

        assert result.dtype == np.uint8
        assert result.shape == image.shape

    def test_rgb(self) -> None:
        image = np.random.default_rng(2).random((32, 32, 3)).astype(np.float64)
        result = destripe(image)

        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_single_channel(self) -> None:
        image = np.random.default_rng(4).random((32, 32, 1)).astype(np.float32)
        result = destripe(image)

        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_constant_returns_copy_without_running_automatic(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fail_automatic_clean(*_: object, **__: object) -> object:
            raise AssertionError("automatic_clean should not run for a constant image")

        monkeypatch.setattr(destripe_ops, "automatic_clean", fail_automatic_clean)

        image = np.full((32, 32), 128, dtype=np.uint8)
        result = destripe(image)

        assert np.array_equal(result, image)
        assert result is not image

    @pytest.mark.parametrize(
        "image",
        [
            np.random.default_rng(5).random((32, 32, 4)),
            np.random.default_rng(6).random(32),
            np.empty((0, 8)),
        ],
    )
    def test_invalid_shape(self, image: np.ndarray) -> None:
        with pytest.raises(ValueError, match="shape"):
            destripe(image)

    @pytest.mark.parametrize(
        "image",
        [
            np.array([["bad"]]),
            np.array([[1.0 + 2.0j]]),
        ],
    )
    def test_invalid_numeric_type(self, image: np.ndarray) -> None:
        with pytest.raises(ValueError, match="numeric"):
            destripe(image)

    def test_invalid_non_finite(self) -> None:
        image = np.random.default_rng(11).random((8, 8))
        image[0, 0] = np.inf

        with pytest.raises(ValueError, match="NaN or Inf"):
            destripe(image)

    def test_reproducible_output_for_fixed_input(self, gray_image: np.ndarray) -> None:
        first = destripe(gray_image)
        second = destripe(gray_image)

        assert np.allclose(first, second, atol=1e-8)
