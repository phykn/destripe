from pathlib import Path

import destripe.adaptive as adaptive


def test_adaptive_is_package_with_public_exports() -> None:
    assert hasattr(adaptive, "__path__")
    assert callable(adaptive.estimate_adaptive_params)
    assert callable(adaptive.smooth_tile_mus)
    assert callable(adaptive.estimate_tile_mus)


def test_adaptive_local_module_owns_tile_local_estimation() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "destripe" / "adaptive"

    assert (root / "local.py").exists()
    assert not (root / "tiles.py").exists()


def test_adaptive_local_module_avoids_manual_nested_window_loops() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "destripe" / "adaptive"
    source = (root / "local.py").read_text(encoding="utf-8")

    assert "row_offset" not in source
    assert "col_offset" not in source
    assert "for row in range(tiles)" not in source
    assert "for col in range(tiles)" not in source


def test_adaptive_constants_live_in_constants_module() -> None:
    from destripe.adaptive import constants, estimate

    assert constants.ALL_DIRECTIONS == (0, 1, 2, 3, 4)
    assert constants.MU1_MIN == 0.10
    assert constants.MU1_MAX == 0.50
    assert constants.MU2_MIN == 0.0017
    assert constants.MU2_MAX == 0.017
    assert constants.EPS == 1e-9
    assert not hasattr(estimate, "MU1_MIN")
    assert not hasattr(estimate, "_MU1_MIN")


def test_adaptive_estimate_module_has_no_top_docstring_or_trivial_wrappers() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "destripe" / "adaptive"
    source = (root / "estimate.py").read_text(encoding="utf-8")

    assert not source.lstrip().startswith('"""')
    assert "def _score_values" not in source
    assert "def _stripe_evidence_weights" not in source
    assert "def _estimate_mu1" not in source
    assert "def _estimate_mu2" not in source


def test_adaptive_estimate_delegates_preprocess_directions_and_strength() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "destripe" / "adaptive"
    estimate_source = (root / "estimate.py").read_text(encoding="utf-8")

    assert (root / "preprocess.py").exists()
    assert (root / "directions.py").exists()
    assert (root / "strength.py").exists()
    assert "def _analysis_tensor" not in estimate_source
    assert "def _high_pass" not in estimate_source
    assert "def _direction_score" not in estimate_source
    assert "def _select_directions" not in estimate_source
    assert "def _distribution_concentration" not in estimate_source
    assert "def _adaptive_strength" not in estimate_source


def test_adaptive_estimate_does_not_shadow_imported_modules() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "destripe" / "adaptive"
    estimate_source = (root / "estimate.py").read_text(encoding="utf-8")

    assert "def _validate_fixed_directions(directions:" not in estimate_source


def test_adaptive_direction_names_describe_their_inputs_and_roles() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "destripe" / "adaptive"
    directions_source = (root / "directions.py").read_text(encoding="utf-8")
    estimate_source = (root / "estimate.py").read_text(encoding="utf-8")
    strength_source = (root / "strength.py").read_text(encoding="utf-8")

    assert "def score_directions(high_pass:" in directions_source
    assert "def score_weights(" in directions_source
    assert "def selection_weights(" in directions_source
    assert "def select_directions_from_weights(" in directions_source
    assert "def evidence_weights(" not in directions_source
    assert "def support_weights(" not in directions_source
    assert "def select_from_weights(" not in directions_source
    assert "def select_directions(" not in directions_source
    assert "evidence_weights" not in estimate_source
    assert "support_weights" not in estimate_source
    assert "evidence_weights" not in strength_source
    assert "support_weights" not in strength_source


def test_adaptive_direction_score_has_no_self_normalizing_factor() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "destripe" / "adaptive"
    directions_source = (root / "directions.py").read_text(encoding="utf-8")

    assert "_robust_contrast" not in directions_source
    assert "power_q" not in directions_source
    assert "contrast" not in directions_source


def test_adaptive_preprocess_validates_shape_before_tensor_conversion() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "destripe" / "adaptive"
    preprocess_source = (root / "preprocess.py").read_text(encoding="utf-8")

    assert "if t.dim() != 2:" not in preprocess_source
    assert 'raise ValueError("gray must have shape (H, W).")' in preprocess_source


def test_preprocess_module_exposes_image_preprocess_helpers() -> None:
    from destripe import preprocess

    root = Path(__file__).resolve().parents[1] / "src" / "destripe"

    assert (root / "preprocess.py").exists()
    assert not (root / "image_ops.py").exists()
    assert preprocess.validate_process_size(None) is None
    assert preprocess.process_shape((20, 30), 15) == (10, 15)
    assert callable(preprocess.solver_gray)
    assert callable(preprocess.resize_2d)
    assert callable(preprocess.rgb_to_luma)


def test_source_files_do_not_use_future_imports() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "destripe"
    offenders = [
        path.relative_to(root)
        for path in root.rglob("*.py")
        if "from __future__" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_source_files_do_not_use_all_exports() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "destripe"
    offenders = [
        path.relative_to(root)
        for path in root.rglob("*.py")
        if "__all__" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_package_init_files_do_not_start_with_docstrings() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "destripe"
    offenders = [
        path.relative_to(root)
        for path in root.rglob("__init__.py")
        if path.read_text(encoding="utf-8").lstrip().startswith('"""')
    ]

    assert offenders == []
