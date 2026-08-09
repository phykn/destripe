import inspect
import warnings

import numpy as np
import pytest
import torch

import destripe.ops as destripe_ops
from destripe import preprocess
from destripe import UniversalStripeRemover, destripe
from destripe.core import operators
from destripe.core.solver import SolveResult


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
    def test_pdhg_step_sizes_satisfy_two_dimensional_operator_bound(self) -> None:
        remover = UniversalStripeRemover(device="cpu")

        assert remover.tau * remover.sigma * 8 < 1

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

    def test_solver_always_runs_requested_iterations(
        self,
    ) -> None:
        remover = UniversalStripeRemover(device="cpu", directions=[0])
        image = torch.full((24, 28), 0.4)

        info = remover._process_with_info(
            image,
            iterations=100,
            proj=True,
        )
        public = remover.process(
            image,
            iterations=100,
            proj=True,
        )

        assert torch.is_tensor(public)
        assert torch.is_tensor(info.clean)
        assert info.iterations == 100
        torch.testing.assert_close(public, info.clean)

    def test_manual_solver_does_not_expose_unused_tolerance(self) -> None:
        assert "tol" not in inspect.signature(UniversalStripeRemover.process).parameters
        assert (
            "tol"
            not in inspect.signature(UniversalStripeRemover.process_tiled).parameters
        )

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

    def test_invalid_non_finite(self, remover: UniversalStripeRemover) -> None:
        img = torch.rand(32, 32)
        img[0, 0] = torch.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            remover.process(image=img)

    @pytest.mark.parametrize(
        "image",
        [
            np.ones((8, 8), dtype=np.complex64),
            torch.ones(8, 8, dtype=torch.complex64),
        ],
    )
    def test_complex_input_is_rejected(
        self,
        remover: UniversalStripeRemover,
        image: np.ndarray | torch.Tensor,
    ) -> None:
        with pytest.raises(ValueError, match="real values"):
            remover.process(image=image)

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

    @pytest.mark.parametrize("proj", [False, True])
    def test_single_pixel_axis_honors_projection_and_cpu_output(
        self,
        proj: bool,
    ) -> None:
        input_device = "cuda" if torch.cuda.is_available() else "cpu"
        remover = UniversalStripeRemover(device=input_device)
        image = torch.tensor([[-0.5, 1.5]], device=input_device)

        result = remover.process(image=image, iterations=1, proj=proj)

        expected = image.clamp(0, 1) if proj else image
        assert result.device.type == "cpu"
        torch.testing.assert_close(result, expected.cpu())


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

    @pytest.mark.parametrize("proj", [False, True])
    def test_single_pixel_axis_honors_projection_and_cpu_output(
        self,
        proj: bool,
    ) -> None:
        input_device = "cuda" if torch.cuda.is_available() else "cpu"
        remover = UniversalStripeRemover(device=input_device)
        image = torch.tensor([[-0.5, 1.5]], device=input_device)

        result = remover.process_tiled(image=image, iterations=1, proj=proj)

        expected = image.clamp(0, 1) if proj else image
        assert result.device.type == "cpu"
        torch.testing.assert_close(result, expected.cpu())

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
                tile_mus=[(1 / 6, 1 / 300)],
            )

        assert remover.mu1 == original_mu1
        assert remover.mu2 == original_mu2

    @pytest.mark.parametrize(
        "tile_mus",
        [
            [1 / 6, 1 / 300, 1 / 5, 1 / 240],
            [(1 / 6, 1 / 300, 1 / 5)] * 4,
            [("bad", 1 / 300)] * 4,
            [(True, 1 / 300)] * 4,
            [(np.nan, 1 / 300)] * 4,
            [(0.1, np.inf)] * 4,
            [(0.0, 1 / 300)] * 4,
            [(-0.1, 1 / 300)] * 4,
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
                tile_mus=[(np.nan, 1 / 300)],
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

    @pytest.mark.parametrize("shape", [(1, 8), (3, 8)])
    def test_valid_tile_mus_raise_when_tiles_cannot_be_applied(
        self,
        remover: UniversalStripeRemover,
        shape: tuple[int, int],
    ) -> None:
        with pytest.raises(ValueError, match="cannot be applied"):
            remover.process_tiled(
                image=torch.rand(shape),
                tiles=4,
                iterations=1,
                overlap=0,
                tile_mus=[(1 / 6, 1 / 300)] * 16,
            )

    def test_tile_mus_restore_after_tile_processing_error(
        self,
        remover: UniversalStripeRemover,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_mu1, original_mu2 = remover.mu1, remover.mu2

        def fail_solve(**_: object) -> SolveResult:
            raise RuntimeError("forced tile failure")

        monkeypatch.setattr(remover, "_run_solver", fail_solve)

        with pytest.raises(RuntimeError, match="forced tile failure"):
            remover.process_tiled(
                image=torch.rand(8, 8),
                tiles=2,
                iterations=1,
                overlap=0,
                tile_mus=[(1 / 6, 1 / 300)] * 4,
            )

        assert remover.mu1 == original_mu1
        assert remover.mu2 == original_mu2

    def test_tile_mus_processes_tiles_in_one_batch(
        self,
        remover: UniversalStripeRemover,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[dict[str, object]] = []

        def fake_solve(**kwargs: object) -> SolveResult:
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
            return SolveResult(clean=data.cpu(), iterations=1)

        monkeypatch.setattr(remover, "_run_solver", fake_solve)

        tile_mus = [
            (1 / 6, 1 / 300),
            (1 / 5, 1 / 240),
            (1 / 4, 1 / 180),
            (1 / 3, 1 / 60),
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
            torch.tensor([1 / 6, 1 / 5, 1 / 4, 1 / 3]),
        )
        assert torch.allclose(
            calls[0]["mu2"].reshape(-1),
            torch.tensor([1 / 300, 1 / 240, 1 / 180, 1 / 60]),
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

    def test_wrapper_forwards_native_gray_and_process_size_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[tuple[tuple[int, int], int | None, bool]] = []

        def fake_automatic_clean(
            gray: np.ndarray,
            *,
            process_size: int | None,
            proj: bool,
        ) -> object:
            calls.append((gray.shape, process_size, proj))
            return type("Result", (), {"clean": gray - 0.1})()

        monkeypatch.setattr(destripe_ops, "automatic_clean", fake_automatic_clean)

        image = np.linspace(0.0, 1.0, 12 * 18).reshape(12, 18)
        result = destripe(image, process_size=9, proj=False)

        assert calls == [((12, 18), 9, False)]
        assert result.shape == image.shape

    def test_wrapper_normalizes_input_and_restores_float_dtype(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: dict[str, object] = {}

        def fake_automatic_clean(
            gray: np.ndarray, *, process_size: int | None, proj: bool
        ) -> object:
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

    @pytest.mark.parametrize(
        ("dtype", "values"),
        [
            (
                np.int64,
                (np.iinfo(np.int64).min, np.iinfo(np.int64).max),
            ),
            (
                np.uint64,
                (np.iinfo(np.uint64).min, np.iinfo(np.uint64).max),
            ),
        ],
    )
    def test_identity_correction_preserves_64_bit_integer_endpoints_without_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dtype: type[np.integer],
        values: tuple[int, int],
    ) -> None:
        def fake_automatic_clean(
            gray: np.ndarray, *, process_size: int | None, proj: bool
        ) -> object:
            return type("Result", (), {"clean": gray.copy()})()

        monkeypatch.setattr(destripe_ops, "automatic_clean", fake_automatic_clean)
        image = np.array([values, values], dtype=dtype)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = destripe(image)

        assert result.dtype == image.dtype
        assert np.array_equal(result, image)

    @pytest.mark.parametrize("dtype", [np.int64, np.uint64])
    def test_nonzero_correction_does_not_wrap_64_bit_integer_maximum(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dtype: type[np.integer],
    ) -> None:
        info = np.iinfo(dtype)

        def fake_automatic_clean(
            gray: np.ndarray, *, process_size: int | None, proj: bool
        ) -> object:
            clean = gray.copy()
            clean[1, 0] = max(0.0, clean[1, 0] - 0.1)
            return type("Result", (), {"clean": clean})()

        monkeypatch.setattr(destripe_ops, "automatic_clean", fake_automatic_clean)
        image = np.array(
            [[info.min, info.max], [info.max // 2, info.min]],
            dtype=dtype,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = destripe(image)

        assert result[0, 1] == info.max
        assert result[0, 1] != info.min

    def test_cross_zero_extreme_float_range_has_finite_normalization_and_inverse(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: dict[str, float] = {}

        def fake_automatic_clean(
            gray: np.ndarray, *, process_size: int | None, proj: bool
        ) -> object:
            assert np.isfinite(gray).all()
            seen["minimum"] = float(gray.min())
            seen["maximum"] = float(gray.max())
            clean = gray.copy()
            clean[1, 0] -= 0.1
            return type("Result", (), {"clean": clean})()

        monkeypatch.setattr(destripe_ops, "automatic_clean", fake_automatic_clean)
        image = np.array(
            [[-1e308, 1e308], [-5e307, 5e307]],
            dtype=np.float64,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = destripe(image)

        assert seen == {"minimum": 0.0, "maximum": 1.0}
        assert np.isfinite(result).all()
        assert result[0, 0] == image[0, 0]
        assert result[0, 1] == image[0, 1]
        assert result[1, 0] == pytest.approx(-7e307)

    def test_grayscale_float64(self, gray_image: np.ndarray) -> None:
        result = destripe(gray_image)

        assert result.shape == gray_image.shape
        assert result.dtype == gray_image.dtype

    def test_process_size_is_forwarded_without_resizing_wrapper_input(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[tuple[tuple[int, int], int | None]] = []

        def fake_automatic_clean(
            gray: np.ndarray,
            *,
            process_size: int | None,
            proj: bool,
        ) -> object:
            calls.append((gray.shape, process_size))
            return type("Result", (), {"clean": gray - 0.25})()

        monkeypatch.setattr(destripe_ops, "automatic_clean", fake_automatic_clean)

        image = np.linspace(0.0, 1.0, 16 * 20, dtype=np.float64).reshape(16, 20)
        result = destripe(image, process_size=10, proj=False)

        assert calls == [((16, 20), 10)]
        assert result.shape == image.shape
        assert np.allclose(result, image - 0.25)

    @pytest.mark.parametrize("process_size", [None, 128])
    def test_process_size_is_forwarded_when_no_resize_is_needed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        process_size: int | None,
    ) -> None:
        calls: list[int | None] = []

        def fake_automatic_clean(
            gray: np.ndarray,
            *,
            process_size: int | None,
            proj: bool,
        ) -> object:
            assert gray.shape == (80, 100)
            calls.append(process_size)
            return type("Result", (), {"clean": gray.copy()})()

        monkeypatch.setattr(destripe_ops, "automatic_clean", fake_automatic_clean)

        image = np.random.default_rng(24).random((80, 100))
        result = destripe(image, process_size=process_size)

        assert calls == [process_size]
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
        assert calls == [{"dsize": (9, 7), "interpolation": preprocess.cv2.INTER_CUBIC}]

    def test_rgb_uses_one_shared_luma_correction(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_automatic_clean(
            gray: np.ndarray, *, process_size: int | None, proj: bool
        ) -> object:
            return type("Result", (), {"clean": gray - 0.125})()

        monkeypatch.setattr(destripe_ops, "automatic_clean", fake_automatic_clean)

        image = np.linspace(0.0, 1.0, 12 * 18 * 3, dtype=np.float64).reshape(12, 18, 3)
        result = destripe(image, process_size=9, proj=False)

        assert result.shape == image.shape
        assert np.allclose(result, image - 0.125)

    def test_projection_is_forwarded_and_applied(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[bool] = []

        def fake_automatic_clean(
            gray: np.ndarray, *, process_size: int | None, proj: bool
        ) -> object:
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

    def test_nonconstant_subpicounit_range_runs_automatic_detection(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: dict[str, float] = {}

        def fake_automatic_clean(
            gray: np.ndarray, *, process_size: int | None, proj: bool
        ) -> object:
            seen["minimum"] = float(gray.min())
            seen["maximum"] = float(gray.max())
            return type("Result", (), {"clean": gray.copy()})()

        monkeypatch.setattr(destripe_ops, "automatic_clean", fake_automatic_clean)
        image = np.linspace(2e-13, 7e-13, 8 * 8, dtype=np.float64).reshape(8, 8)

        result = destripe(image)

        assert seen == {"minimum": 0.0, "maximum": 1.0}
        assert np.array_equal(result, image)

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
