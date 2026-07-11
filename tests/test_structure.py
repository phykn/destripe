from pathlib import Path


def test_simple_automatic_wrapper_structure() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "destripe"
    ops_source = (root / "ops.py").read_text(encoding="utf-8")

    assert (root / "automatic.py").exists()
    assert (root / "adaptive").is_dir()
    assert not (root / "hybrid.py").exists()
    assert "from .automatic import automatic_clean" in ops_source
    assert "UniversalStripeRemover" not in ops_source
    assert "adaptive_level" not in ops_source
    assert "estimate_adaptive_params" not in ops_source
    assert "estimate_tile_mus" not in ops_source
    assert "refine_clean" not in ops_source


def test_source_modules_use_direct_relative_imports() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "destripe"

    for path in root.rglob("*.py"):
        if path.name == "constants.py":
            continue
        source = path.read_text(encoding="utf-8")
        assert "from . import " not in source
        assert "from . import constants" not in source


def test_preprocess_module_exposes_image_preprocess_helpers() -> None:
    from destripe import preprocess

    root = Path(__file__).resolve().parents[1] / "src" / "destripe"
    source = (root / "preprocess.py").read_text(encoding="utf-8")

    assert (root / "preprocess.py").exists()
    assert not (root / "image_ops.py").exists()
    assert preprocess.validate_process_size(None) is None
    assert preprocess.compute_solver_shape((20, 30), 15) == (10, 15)
    assert callable(preprocess.prepare_solver_gray)
    assert callable(preprocess.resize_to_shape)
    assert callable(preprocess.rgb_to_luma)
    assert "def process_shape(" not in source
    assert "def solver_gray(" not in source
    assert "def solver_shape(" not in source
    assert "def _scaled_dim(" not in source
    assert "def _scale_dim(" in source
    assert "def resize_lanczos(" not in source
    assert "INTER_LANCZOS4" not in source
    assert "INTER_CUBIC" in source
    assert "def resize_2d(" not in source
    assert "mode:" not in source
    assert "unsupported resize mode" not in source


def test_ops_module_keeps_automatic_flow_direct_and_docstring_concise() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "destripe"
    source = (root / "ops.py").read_text(encoding="utf-8")

    assert "def _run_grayscale(" not in source
    assert "def _destripe_grayscale(" not in source
    assert "process_size: Long-edge analysis size; None keeps original resolution." in source
    assert "preserving aspect ratio, then upsample" not in source
    assert "Returns:\n        Destriped image with the same shape and dtype." in source


def test_core_is_package_with_remover_and_operators() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "destripe"
    core_root = root / "core"
    remover_source = (core_root / "remover.py").read_text(encoding="utf-8")
    operator_source = (core_root / "operators.py").read_text(encoding="utf-8")

    assert core_root.is_dir()
    assert (core_root / "__init__.py").exists()
    assert (core_root / "remover.py").exists()
    assert (core_root / "operators.py").exists()
    assert not (core_root / "result.py").exists()
    assert not (root / "core.py").exists()
    assert "class UniversalStripeRemover" in remover_source
    assert "StripeResult" not in remover_source
    assert "process_tiled_components" not in remover_source
    assert "keep_components" not in remover_source
    assert "def _process_tiled(" not in remover_source
    assert "def forward_diff(" in operator_source
    assert "def dir_diff(" in operator_source
    assert "def adjoint_1d(" in operator_source
    assert "def adjoint_grad(" in operator_source
    assert "def adjoint_dir(" in operator_source
    assert "scale:" in operator_source
    assert "_NUM_DIRS" not in remover_source
    assert "_ALL_DIRECTIONS" not in remover_source
    assert "DIRECTION_MODES = (0, 1, 2, 3, 4)" in remover_source
    assert "def _forward_diff(" not in remover_source
    assert "def _dir_diff(" not in remover_source
    assert "def _adjoint_" not in remover_source
    assert "def _process_single_tile(" not in remover_source
    assert "original_mu1" not in remover_source
    assert "original_mu2" not in remover_source
    assert "validated_tile_mus" not in remover_source
    assert "l2_dual" not in remover_source
    assert "sparse_dual" in remover_source
    assert "On CUDA" not in remover_source
    assert "def _make_tile_mu_tensors(" in remover_source
    assert "def _make_solver_mu_tensor(" in remover_source
    assert "def _convert_to_tensor(" in remover_source
    assert "def _make_zero_pair(" in remover_source
    assert "def _make_cosine_window(" in remover_source
    assert "def _tile_mu_tensors(" not in remover_source
    assert "def _solver_mu_tensor(" not in remover_source
    assert "def _to_tensor(" not in remover_source
    assert "def _zero_pair(" not in remover_source
    assert "def _cosine_window(" not in remover_source


def test_core_package_does_not_use_section_marker_comments() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "destripe" / "core"
    offenders = [
        path.relative_to(root)
        for path in root.rglob("*.py")
        if "# ---" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


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
