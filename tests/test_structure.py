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


def test_image_ops_module_exposes_process_size_helpers() -> None:
    from destripe import image_ops

    assert image_ops.validate_process_size(None) is None
    assert image_ops.process_shape((20, 30), 15) == (10, 15)
    assert callable(image_ops.solver_gray)
    assert callable(image_ops.resize_2d)
    assert callable(image_ops.rgb_to_luma)


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
