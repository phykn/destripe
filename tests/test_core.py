import numpy as np
import pytest
import torch

import destripe.ops as destripe_ops
from destripe import preprocess
from destripe import UniversalStripeRemover, destripe
from destripe.adaptive import estimate_adaptive_params
from destripe.core import operators
from destripe.core.result import StripeResult


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


@pytest.mark.parametrize("tiles", [1, 2])
def test_process_tiled_components_reconstructs_input(tiles: int) -> None:
    remover = UniversalStripeRemover(device="cpu", directions=[0, 2])
    image = torch.rand((32, 36), generator=torch.Generator().manual_seed(92))

    result = remover.process_tiled_components(
        image=image,
        tiles=tiles,
        overlap=4,
        iterations=8,
        proj=False,
    )

    assert result.clean.shape == image.shape
    assert len(result.components) == 2
    assert all(component.shape == image.shape for component in result.components)
    reconstructed = result.clean + torch.stack(result.components).sum(dim=0)
    assert torch.allclose(reconstructed, image, atol=2e-5, rtol=2e-5)


def test_process_tiled_components_handles_tiny_image() -> None:
    remover = UniversalStripeRemover(device="cpu", directions=[0, 3])
    image = torch.rand((1, 8), generator=torch.Generator().manual_seed(93))

    result = remover.process_tiled_components(image=image, tiles=3, iterations=1)

    assert torch.equal(result.clean, image)
    assert len(result.components) == 2
    assert all(torch.count_nonzero(component) == 0 for component in result.components)


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

    def test_tile_mus_restore_after_tile_processing_error(
        self,
        remover: UniversalStripeRemover,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_mu1, original_mu2 = remover.mu1, remover.mu2

        def fail_solve(**_: object) -> StripeResult:
            raise RuntimeError("forced tile failure")

        monkeypatch.setattr(remover, "_solve", fail_solve)

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

        def fake_solve(**kwargs: object) -> StripeResult:
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
            return StripeResult(clean=data.cpu(), components=())

        monkeypatch.setattr(remover, "_solve", fake_solve)

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


class TestAdaptiveEstimator:
    def test_constant_high_pass_scores_are_direction_neutral(self) -> None:
        from destripe.adaptive import directions

        scores = directions.score_directions(torch.zeros(64, 64))
        values = np.array([scores[mode] for mode in range(5)])

        assert np.isfinite(values).all()
        assert np.allclose(values, values[0])

    def test_vertical_stripes_select_mode_zero(self) -> None:
        from destripe.adaptive import constants

        img = np.zeros((64, 64), dtype=np.float64)
        img[:, 12] = 1.0
        img[:, 32] = 0.8
        params = estimate_adaptive_params(img, level=3)
        assert params.directions[0] == 0
        assert params.mu1 == pytest.approx(1 / 3)
        assert params.mu2 == pytest.approx(constants.MU2_MIN)
        assert 1 / 6 <= params.mu1 <= 1 / 3
        assert 1 / 300 <= params.mu2 <= 1 / 60

    def test_sparse_vertical_stripes_select_mode_zero(self) -> None:
        img = np.zeros((96, 96), dtype=np.float64)
        img[:, 20] = 1.0
        img[:, 60] = 0.8

        params = estimate_adaptive_params(img, level=2)

        assert params.directions[0] == 0

    @pytest.mark.parametrize(
        ("level", "mu1"),
        [
            (0, 1 / 6),
            (1, 1 / 5),
            (2, 1 / 4),
            (3, 1 / 3),
        ],
    )
    def test_level_sets_mu1(self, level: int, mu1: float) -> None:
        img = np.zeros((48, 48), dtype=np.float64)
        img[:, 12] = 1.0

        params = estimate_adaptive_params(img, level=level)

        assert params.mu1 == pytest.approx(mu1)

    def test_faint_coherent_stripes_choose_low_mu2(self) -> None:
        from destripe.adaptive import constants

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

        params = estimate_adaptive_params(img, level=2)

        assert params.directions[0] == 0
        assert params.mu1 == pytest.approx(1 / 4)
        assert any(
            params.mu2 == pytest.approx(1 / denominator)
            for denominator in constants.MU2_DENOMINATORS
        )
        assert params.mu2 <= 1 / 120

    def test_coherent_curtain_chooses_low_mu2(self) -> None:
        from destripe.adaptive import constants

        rng = np.random.default_rng(4321)
        h = w = 128
        x = np.linspace(0, 1, w).reshape(1, -1)
        y = np.linspace(0, 1, h).reshape(-1, 1)
        base = 0.35 + 0.25 * x + 0.15 * y
        texture = 0.08 * np.sin(2 * np.pi * x * 7) * np.sin(2 * np.pi * y * 5)
        profile = rng.normal(0, 1, w)
        profile = np.convolve(profile, np.ones(5) / 5, mode="same")
        profile = (profile - profile.mean()) / profile.std()
        curtain = 0.02 * profile.reshape(1, -1)
        img = base + texture + curtain
        img = (img - img.min()) / (img.max() - img.min())

        params = estimate_adaptive_params(img, level=2)

        assert params.directions[0] == 0
        assert params.mu2 == pytest.approx(constants.MU2_MIN)

    def test_directional_texture_uses_sparse_stripe_guard(self) -> None:
        h = w = 128
        x = np.linspace(0, 1, w).reshape(1, -1)
        y = np.linspace(0, 1, h).reshape(-1, 1)
        img = 0.5 + 0.05 * np.sin(2 * np.pi * x * 16) * np.sin(
            2 * np.pi * y * 16
        )

        params = estimate_adaptive_params(img, level=2)

        assert params.mu1 == pytest.approx(1 / 4)
        assert params.mu2 > 1 / 120

    def test_mu2_uses_data_with_same_level(self) -> None:
        h = w = 96
        strong = np.zeros((h, w), dtype=np.float64)
        strong[:, 20] = 1.0
        strong[:, 60] = 0.8

        x = np.linspace(0, 1, w).reshape(1, -1)
        y = np.linspace(0, 1, h).reshape(-1, 1)
        texture = 0.5 + 0.05 * np.sin(2 * np.pi * x * 12) * np.sin(
            2 * np.pi * y * 12
        )

        strong_params = estimate_adaptive_params(strong, level=2)
        texture_params = estimate_adaptive_params(texture, level=2)

        assert strong_params.mu1 == pytest.approx(texture_params.mu1)
        assert strong_params.mu2 < texture_params.mu2

    def test_estimator_is_deterministic(self) -> None:
        rng = np.random.default_rng(14)
        img = rng.random((48, 48))
        p1 = estimate_adaptive_params(img, level=2)
        p2 = estimate_adaptive_params(img, level=2)
        assert p1 == p2

    def test_estimator_normalizes_affine_intensity_scale(self) -> None:
        rng = np.random.default_rng(16)
        img = rng.random((48, 48))
        img[:, 10] += 0.7
        img = np.clip(img, 0.0, 1.0)

        p1 = estimate_adaptive_params(img, level=2)
        p2 = estimate_adaptive_params(img * 100.0 + 20.0, level=2)

        assert p1.directions == p2.directions
        assert p1.mu1 == pytest.approx(p2.mu1)
        assert p1.mu2 == pytest.approx(p2.mu2)
        assert p1.confidence == pytest.approx(p2.confidence)

    def test_strength_reuses_per_direction_statistics(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from destripe.adaptive import strength

        calls = {"project": 0, "measure_shrinkage": 0}
        original_project = strength.project
        original_measure_shrinkage = strength.measure_shrinkage

        def counted_project(tensor: torch.Tensor, mode: int) -> torch.Tensor:
            calls["project"] += 1
            return original_project(tensor, mode)

        def counted_measure_shrinkage(tensor: torch.Tensor, mode: int) -> float:
            calls["measure_shrinkage"] += 1
            return original_measure_shrinkage(tensor, mode)

        monkeypatch.setattr(strength, "project", counted_project)
        monkeypatch.setattr(
            strength,
            "measure_shrinkage",
            counted_measure_shrinkage,
        )

        high_pass = torch.randn(
            (48, 48),
            generator=torch.Generator().manual_seed(19),
        )
        strength.estimate_strength(
            high_pass=high_pass,
            selected_directions=(0, 2),
            score_weights=np.array([0.6, 0.0, 0.4, 0.0, 0.0]),
            selection_weights=np.array([0.6, 0.0, 0.4, 0.0, 0.0]),
        )

        assert calls == {"project": 2, "measure_shrinkage": 2}

    def test_direction_support_uses_relative_scores_without_cutoffs(self) -> None:
        from destripe.adaptive import directions

        scores = {0: 0.20, 1: 0.19, 2: -1.0, 3: -1.0, 4: -1.0}
        weights = directions.make_selection_weights(scores)

        assert directions.select_directions(weights) == (0, 1)

    def test_flat_direction_scores_do_not_select_all_modes(self) -> None:
        from destripe.adaptive import directions

        scores = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
        weights = directions.make_selection_weights(scores)

        assert directions.select_directions(weights) == (0,)

    def test_fixed_directions_do_not_add_count_penalty(self) -> None:
        img = np.zeros((64, 64), dtype=np.float64)
        img[:, 12] = 1.0
        img[:, 32] = 0.8

        single = estimate_adaptive_params(img, level=2, fixed_directions=(0,))
        multiple = estimate_adaptive_params(img, level=2, fixed_directions=(0, 1))

        assert single.directions == (0,)
        assert multiple.directions == (0, 1)
        assert single.mu1 == pytest.approx(multiple.mu1)

    @pytest.mark.parametrize("shape", [(1, 1), (3, 8), (8, 3)])
    def test_estimator_handles_small_arrays(self, shape: tuple[int, int]) -> None:
        img = np.random.default_rng(17).random(shape)
        params = estimate_adaptive_params(img, level=2)

        assert params.directions
        assert params.mu1 == pytest.approx(1 / 4)
        assert 1 / 300 <= params.mu2 <= 1 / 60

    @pytest.mark.parametrize("level", [-1, 4, 1.5, True, None])
    def test_invalid_adaptive_level(self, level: object) -> None:
        with pytest.raises(ValueError, match="adaptive"):
            estimate_adaptive_params(
                np.random.default_rng(18).random((8, 8)),
                level=level,  # type: ignore[arg-type]
            )

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
                level=2,
                fixed_directions=fixed_directions,  # type: ignore[arg-type]
            )

    def test_tile_mu_smoothing_preserves_shape(self) -> None:
        from destripe.adaptive import constants
        from destripe.adaptive import smooth_tile_mus

        mus = np.array(
            [
                [[1 / 6, 1 / 300], [1 / 2, 1 / 60]],
                [[1 / 3, 1 / 300], [1 / 4, 1 / 120]],
            ],
            dtype=np.float64,
        )
        smoothed = smooth_tile_mus(mus)
        assert smoothed.shape == mus.shape
        assert np.all(smoothed[..., 0] >= constants.MU1_MIN)
        assert np.all(smoothed[..., 0] <= constants.MU1_MAX)
        assert np.all(smoothed[..., 1] >= constants.MU2_MIN)
        assert np.all(smoothed[..., 1] <= constants.MU2_MAX)


def test_robust_projection_ignores_protected_local_structure() -> None:
    from destripe.adaptive.stripe import project_robust

    tensor = torch.zeros((32, 12), dtype=torch.float32)
    tensor[:, 5] = 0.02
    tensor[10:14, 5] = 1.0
    weights = torch.ones_like(tensor)
    weights[9:15, 5] = 0.0

    projected = project_robust(tensor, mode=0, weights=weights)

    assert torch.allclose(projected[:, 5], torch.full((32,), 0.02), atol=1e-4)


def test_weighted_shrinkage_ignores_protected_mismatch() -> None:
    from destripe.adaptive.stripe import measure_shrinkage

    tensor = torch.zeros((32, 8), dtype=torch.float32)
    tensor[:, 3] = 0.02
    tensor[[0, 2], 3] += 0.5
    weights = torch.ones_like(tensor)
    weights[[0, 2], 3] = 0.0

    assert measure_shrinkage(tensor, 0, weights=weights) > 0.9


class TestAdaptiveRefine:
    def test_measure_shrinkage_uses_full_line_reliability(self) -> None:
        from destripe.adaptive import stripe

        tensor = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [-0.5, 0.0, 0.5],
                [-1.0, 0.0, 1.0],
                [-0.5, 0.0, 0.5],
            ],
            dtype=torch.float32,
        )

        assert stripe.measure_shrinkage(tensor, 0) == pytest.approx(8 / 9)

    def test_measure_shrinkage_clamps_unreliable_profiles(self) -> None:
        from destripe.adaptive import stripe

        tensor = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [1.0, 0.0, -1.0],
                [-1.0, 0.0, 1.0],
                [1.0, 0.0, -1.0],
            ],
            dtype=torch.float32,
        )

        assert stripe.measure_shrinkage(tensor, 0) == 0.0

    def test_refine_clean_moves_residual_stripe(self) -> None:
        from destripe.adaptive.refine import refine_clean

        h = w = 64
        x = np.linspace(0.25, 0.75, w).reshape(1, -1)
        gray = np.repeat(x, h, axis=0)
        gray[:, 20] += 0.08
        gray[:, 42] -= 0.06
        gray = np.clip(gray, 0.0, 1.0)

        refined = refine_clean(
            gray=gray,
            clean=gray,
            directions=(0,),
            proj=False,
        )

        before = _column_artifact(gray, columns=(20, 42))
        after = _column_artifact(refined, columns=(20, 42))
        assert after < before

    def test_refine_clean_keeps_unhelpful_candidate(self) -> None:
        from destripe.adaptive.refine import refine_clean

        h = w = 64
        x = np.linspace(0, 1, w).reshape(1, -1)
        y = np.linspace(0, 1, h).reshape(-1, 1)
        clean = 0.5 + 0.05 * np.sin(2 * np.pi * x * 12) * np.sin(
            2 * np.pi * y * 12
        )

        refined = refine_clean(
            gray=clean,
            clean=clean,
            directions=(0,),
            proj=False,
        )

        assert np.allclose(refined, clean)


def _column_artifact(image: np.ndarray, *, columns: tuple[int, ...]) -> float:
    values = [
        np.mean(np.abs(image[:, col] - 0.5 * (image[:, col - 1] + image[:, col + 1])))
        for col in columns
    ]
    return float(np.mean(values))


class TestDestripe:
    def test_grayscale_float64(self, gray_image: np.ndarray) -> None:
        result = destripe(gray_image, iterations=20)
        assert result.shape == gray_image.shape
        assert result.dtype == gray_image.dtype

    def test_solver_receives_float32_and_restores_input_dtype(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen = {}

        class FakeRemover:
            def __init__(self, **_: object) -> None:
                pass

            def process_tiled(self, image: np.ndarray, **_: object) -> torch.Tensor:
                seen["dtype"] = image.dtype
                return torch.as_tensor(image)

        monkeypatch.setattr(destripe_ops, "UniversalStripeRemover", FakeRemover)

        img = np.random.default_rng(31).random((16, 16)).astype(np.float64)
        result = destripe(img, iterations=1)

        assert seen["dtype"] == np.float32
        assert result.dtype == img.dtype
        assert np.allclose(result, img, atol=1e-7)

    def test_adaptive_uses_estimated_parameters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from destripe import ops
        from destripe.adaptive import AdaptiveParams

        calls = {}

        def fake_estimate(
            gray: np.ndarray, *, level: int, fixed_directions=None
        ) -> AdaptiveParams:
            calls["shape"] = gray.shape
            calls["level"] = level
            calls["fixed_directions"] = fixed_directions
            return AdaptiveParams(
                directions=(0,), mu1=1 / 6, mu2=1 / 300, confidence=1.0
            )

        monkeypatch.setattr(
            ops, "estimate_adaptive_params", fake_estimate, raising=False
        )
        img = np.random.default_rng(17).random((24, 24))
        result = destripe(img, adaptive=0, iterations=5)

        assert result.shape == img.shape
        assert calls == {"shape": img.shape, "level": 0, "fixed_directions": None}

    def test_adaptive_grayscale_float64(self, gray_image: np.ndarray) -> None:
        result = destripe(gray_image, adaptive=2, iterations=10)
        assert result.shape == gray_image.shape
        assert result.dtype == gray_image.dtype

    def test_adaptive_rgb(self) -> None:
        img = np.random.default_rng(15).random((32, 32, 3)).astype(np.float32)
        result = destripe(img, adaptive=2, iterations=10)
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
            gray: np.ndarray, *, level: int, fixed_directions=None
        ) -> AdaptiveParams:
            assert level == 3
            return AdaptiveParams(
                directions=(0,), mu1=1 / 3, mu2=1 / 300, confidence=1.0
            )

        original = UniversalStripeRemover.process_tiled

        def spy_process_tiled(self, *args, **kwargs):
            seen["tile_mus"] = kwargs.get("tile_mus")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(ops, "estimate_adaptive_params", fake_estimate)
        monkeypatch.setattr(UniversalStripeRemover, "process_tiled", spy_process_tiled)

        img = np.random.default_rng(18).random((32, 32))
        result = destripe(img, adaptive=3, iterations=5, tiles=2, overlap=4)

        assert result.shape == img.shape
        assert seen["tile_mus"] is not None
        assert len(seen["tile_mus"]) == 4

    def test_adaptive_tiled_dtype_shape(self) -> None:
        img = np.random.default_rng(16).random((48, 40, 3)).astype(np.float32)
        result = destripe(img, adaptive=2, iterations=5, tiles=3, overlap=6)
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

    def test_resize_to_shape_resizes_2d_images(self) -> None:
        img = np.random.default_rng(23).random((11, 13))
        result = preprocess.resize_to_shape(img, shape=(7, 9))

        assert result.shape == (7, 9)
        assert result.dtype == np.float64
        assert np.isfinite(result).all()

    def test_resize_to_shape_uses_cubic_interpolation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = []

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
            gray: np.ndarray, *, level: int, fixed_directions=None
        ) -> AdaptiveParams:
            calls.append((gray.shape, level, fixed_directions))
            return AdaptiveParams(
                directions=(0,), mu1=1 / 3, mu2=1 / 300, confidence=1.0
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
        result = destripe(img, adaptive=3, process_size=15, iterations=1)

        assert result.shape == img.shape
        assert calls == [((10, 15), 3, None)]

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
        result = destripe(img, adaptive=2, tiles=4, iterations=1)
        assert result.shape == img.shape
        assert result.dtype == img.dtype

    def test_adaptive_constant_returns_copy(self) -> None:
        img = np.full((16, 16), 12, dtype=np.uint8)
        result = destripe(img, adaptive=2)
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
        with pytest.warns(UserWarning, match="adaptive level ignores"):
            result = destripe(
                img,
                adaptive=2,
                directions=[0],
                mu1=1 / 2,
                mu2=1 / 60,
                iterations=5,
            )
        assert result.shape == img.shape

    @pytest.mark.parametrize("adaptive", [True, False, -1, 4, 1.5, "2"])
    def test_invalid_adaptive_value(self, adaptive: object) -> None:
        with pytest.raises(ValueError, match="adaptive"):
            destripe(
                np.random.default_rng(13).random((8, 8)),
                adaptive=adaptive,  # type: ignore[arg-type]
                iterations=1,
            )

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
            mu1=1 / 2,
            mu2=1 / 60,
            directions=[1, 4],
            iterations=1,
            device="cpu",
        )

        assert result.shape == img.shape
        assert calls == [
            {
                "mu1": 1 / 2,
                "mu2": 1 / 60,
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
            gray: np.ndarray, *, level: int, fixed_directions=None
        ) -> AdaptiveParams:
            estimate_calls.append(
                {
                    "shape": gray.shape,
                    "level": level,
                    "fixed_directions": fixed_directions,
                }
            )
            return AdaptiveParams(
                directions=(0,), mu1=1 / 4, mu2=1 / 300, confidence=1.0
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
        with pytest.warns(UserWarning, match="adaptive level ignores"):
            result = destripe(
                img,
                adaptive=2,
                mu1=1 / 2,
                mu2=1 / 60,
                directions=[1, 4],
                iterations=1,
            )

        assert result.shape == img.shape
        assert calls == [
            {
                "mu1": 1 / 4,
                "mu2": 1 / 300,
                "device": None,
                "directions": (0,),
            }
        ]
        assert estimate_calls == [
            {"shape": img.shape, "level": 2, "fixed_directions": None}
        ]

    def test_adaptive_refines_clean_after_solver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from destripe.adaptive import AdaptiveParams

        seen: dict[str, object] = {}

        def fake_estimate(
            gray: np.ndarray, *, level: int, fixed_directions=None
        ) -> AdaptiveParams:
            assert level == 2
            return AdaptiveParams(
                directions=(0,), mu1=1 / 4, mu2=1 / 300, confidence=1.0
            )

        def fake_refine(
            *,
            gray: np.ndarray,
            clean: np.ndarray,
            directions: tuple[int, ...],
            proj: bool,
        ) -> np.ndarray:
            seen["directions"] = directions
            seen["proj"] = proj
            return clean - 0.1

        class FakeRemover:
            def __init__(
                self,
                mu1: float,
                mu2: float,
                device: torch.device | str | None = None,
                directions: object = None,
            ) -> None:
                pass

            def process_tiled(self, image: np.ndarray, **_: object) -> torch.Tensor:
                return torch.as_tensor(image)

        monkeypatch.setattr(destripe_ops, "estimate_adaptive_params", fake_estimate)
        monkeypatch.setattr(destripe_ops, "refine_clean", fake_refine)
        monkeypatch.setattr(destripe_ops, "UniversalStripeRemover", FakeRemover)

        img = np.random.default_rng(21).random((8, 8))
        result = destripe(img, adaptive=2, iterations=1, proj=False)

        assert seen == {
            "directions": (0,),
            "proj": False,
        }
        assert np.allclose(result, img - 0.1 * (img.max() - img.min()))

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
