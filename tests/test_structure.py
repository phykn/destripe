import inspect
from pathlib import Path

from destripe import UniversalStripeRemover, destripe, preprocess


ROOT = Path(__file__).resolve().parents[1] / "src" / "destripe"


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_declared_python_version_matches_annotation_syntax() -> None:
    pyproject = (ROOT.parents[1] / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.10"' in pyproject


def test_public_api_stays_small_and_stable() -> None:
    assert tuple(inspect.signature(destripe).parameters) == (
        "image",
        "process_size",
        "proj",
    )
    assert tuple(inspect.signature(UniversalStripeRemover).parameters) == (
        "mu1",
        "mu2",
        "device",
        "directions",
    )


def test_public_wrapper_owns_image_io_not_solver_resolution() -> None:
    source = _read("ops.py")

    assert "from .automatic import automatic_clean" in source
    assert "UniversalStripeRemover" not in source
    assert "prepare_solver_gray" not in source
    assert "resize_to_shape" not in source
    assert "process_size=process_size_value" in source


def test_automatic_pipeline_owns_working_resolution_and_solver() -> None:
    source = _read("automatic.py")

    assert "from .core import UniversalStripeRemover" in source
    assert "prepare_solver_gray" in source
    assert "resize_to_shape" in source
    assert "estimate_adaptive_params(values)" in source
    assert "process_size=process_size_value" in source


def test_adaptive_modules_have_distinct_analysis_profile_and_structure_owners() -> None:
    adaptive = ROOT / "adaptive"
    analysis = (adaptive / "analysis.py").read_text(encoding="utf-8")
    profiles = (adaptive / "profiles.py").read_text(encoding="utf-8")
    structure = (adaptive / "structure.py").read_text(encoding="utf-8")

    assert not (adaptive / "preprocess.py").exists()
    assert not (adaptive / "stripe.py").exists()
    assert "def extract_high_pass(" in analysis
    assert "def make_profile(" in profiles
    assert "def extract_sparse_profile_structure(" in structure
    assert "from .profiles import" in structure
    assert "from .structure import" not in profiles


def test_core_dependency_direction_is_remover_to_solver_to_operators() -> None:
    core = ROOT / "core"
    remover = (core / "remover.py").read_text(encoding="utf-8")
    solver = (core / "solver.py").read_text(encoding="utf-8")
    operators = (core / "operators.py").read_text(encoding="utf-8")

    assert "from .solver import SolveResult, solve_pdhg" in remover
    assert "from .operators import" in solver
    assert "remover" not in solver
    assert "solver" not in operators
    assert "automatic" not in remover + solver + operators


def test_preprocess_helpers_keep_shape_and_resize_contracts() -> None:
    assert preprocess.validate_process_size(None) is None
    assert preprocess.compute_solver_shape((20, 30), 15) == (10, 15)
    assert callable(preprocess.prepare_solver_gray)
    assert callable(preprocess.resize_to_shape)
    assert callable(preprocess.rgb_to_luma)


def test_source_modules_use_direct_relative_imports() -> None:
    for path in ROOT.rglob("*.py"):
        if path.name == "constants.py":
            continue
        source = path.read_text(encoding="utf-8")
        assert "from . import " not in source
        assert "from . import constants" not in source


def test_source_files_avoid_future_imports_and_all_exports() -> None:
    for path in ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "from __future__" not in source
        assert "__all__" not in source
