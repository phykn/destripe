import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from benchmarks.performance import main as performance_main
from benchmarks.synthetic import (
    PatternSpec,
    RESULT_FIELDS,
    default_pattern_specs,
    diagnostic_summary_lines,
    inject_stripe,
    load_samples,
    main,
    make_stripe_pattern,
    run_benchmark,
    structural_similarity,
)
from destripe.automatic import PARALLEL_OFFSETS


@pytest.mark.parametrize("mode", range(5))
def test_curtain_pattern_is_constant_along_requested_direction(mode: int) -> None:
    pattern = make_stripe_pattern(
        shape=(33, 35),
        kind="curtain",
        mode=mode,
        rng=np.random.default_rng(123),
    )
    row_step, col_step = PARALLEL_OFFSETS[mode]
    rows = pattern.shape[0] - abs(row_step)
    cols = pattern.shape[1] - abs(col_step)
    row_a = max(0, -row_step)
    row_b = max(0, row_step)
    col_a = max(0, -col_step)
    col_b = max(0, col_step)

    first = pattern[row_a : row_a + rows, col_a : col_a + cols]
    second = pattern[row_b : row_b + rows, col_b : col_b + cols]

    assert np.allclose(first, second)
    assert float(pattern.mean()) == pytest.approx(0.0, abs=1e-12)
    assert float(pattern.std()) == pytest.approx(1.0)


def test_load_samples_reserves_sample_one_for_real_stripe(tmp_path: Path) -> None:
    for index in range(1, 6):
        suffix = ".jpeg" if index == 1 else ".png"
        image = np.full((12, 16), index * 20, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"sample_{index:02d}{suffix}"), image)

    samples = load_samples(tmp_path)

    assert [sample.name for sample in samples] == [
        "sample_01.jpeg",
        "sample_02.png",
        "sample_03.png",
        "sample_04.png",
        "sample_05.png",
    ]
    assert [sample.has_ground_truth for sample in samples] == [
        False,
        True,
        True,
        True,
        True,
    ]


def test_inject_stripe_reports_clipped_actual_delta() -> None:
    clean = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
    pattern = np.array([[-1.0, 1.0, 1.0]], dtype=np.float64)

    observed, actual = inject_stripe(clean, pattern, strength=0.2)

    assert np.allclose(observed, [[0.0, 0.7, 1.0]])
    assert np.allclose(actual, observed - clean)


def test_multiplicative_injection_reports_actual_stripe() -> None:
    clean = np.full((12, 14), 0.5)
    pattern = np.tile(np.linspace(-1, 1, 14), (12, 1))
    observed, actual = inject_stripe(
        clean, pattern, strength=0.02, carrier="multiplicative"
    )
    assert np.allclose(actual, observed - clean)
    assert not np.allclose(actual, 0.02 * pattern)


def make_acceptance_fixture(
    *,
    weak_projection_left: float,
    clean_psnr: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    samples = [f"sample_{index:02d}.png" for index in range(2, 6)]
    for sample in samples:
        rows.append(
            {
                "seed": 1234,
                "sample": sample,
                "case_type": "clean",
                "pattern": "none",
                "mode": None,
                "strength": 0.0,
                "carrier": "additive",
                "profile_scale": 9,
                "angle_offset": 0.0,
                "input_psnr": float("inf"),
                "output_psnr": clean_psnr,
                "input_ssim": 1.0,
                "output_ssim": 1.0,
                "stripe_projection_left_pct": 0.0,
            }
        )
        for spec in default_pattern_specs():
            for strength, psnr_gain, ssim_gain in (
                (0.01, 0.2, 0.002),
                (0.03, 1.0, 0.003),
                (0.06, 4.0, 0.004),
            ):
                rows.append(
                    {
                        "seed": 1234,
                        "sample": sample,
                        "case_type": "synthetic",
                        "pattern": spec.name,
                        "mode": spec.mode,
                        "strength": strength,
                        "carrier": spec.carrier,
                        "profile_scale": spec.profile_scale,
                        "angle_offset": spec.angle_offset,
                        "input_psnr": 40.0,
                        "output_psnr": 40.0 + psnr_gain,
                        "input_ssim": 0.95,
                        "output_ssim": 0.95 + ssim_gain,
                        "stripe_projection_left_pct": (
                            weak_projection_left if strength == 0.01 else 50.0
                        ),
                    }
                )
    return rows


def test_result_fields_drop_removed_adaptive_metadata() -> None:
    assert {
        "level",
        "selected_directions",
        "mu1",
        "mu2",
        "confidence",
    }.isdisjoint(RESULT_FIELDS)


def test_acceptance_rejects_noop_weak_rows() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=100.0, clean_psnr=100.0)
    failures = evaluate_acceptance(rows)
    assert any("projection" in failure for failure in failures)


def test_acceptance_rejects_clean_fidelity_below_absolute_gate() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=39.9)
    failures = evaluate_acceptance(rows)
    assert any("clean PSNR" in failure for failure in failures)


def test_structural_similarity_is_one_for_identical_images() -> None:
    image = np.random.default_rng(5).random((32, 32))

    assert structural_similarity(image, image) == pytest.approx(1.0)


def test_default_patterns_cover_all_directions_and_two_vertical_guards() -> None:
    specs = default_pattern_specs()

    assert [
        (
            spec.name,
            spec.kind,
            spec.mode,
            spec.carrier,
            spec.profile_scale,
            spec.angle_offset,
        )
        for spec in specs
    ] == [
        ("curtain_m0", "curtain", 0, "additive", 9, 0.0),
        ("curtain_m1", "curtain", 1, "additive", 9, 0.0),
        ("curtain_m2", "curtain", 2, "additive", 9, 0.0),
        ("curtain_m3", "curtain", 3, "additive", 9, 0.0),
        ("curtain_m4", "curtain", 4, "additive", 9, 0.0),
        ("sparse_m0", "sparse", 0, "additive", 9, 0.0),
        ("nonstationary_m0", "nonstationary", 0, "additive", 9, 0.0),
        ("curtain_narrow_m0", "curtain", 0, "additive", 3, 0.0),
        ("curtain_broad_m0", "curtain", 0, "additive", 15, 0.0),
        ("curtain_multiplicative_m0", "curtain", 0, "multiplicative", 9, 0.0),
        ("curtain_multiplicative_m1", "curtain", 1, "multiplicative", 9, 0.0),
        ("curtain_multiplicative_m2", "curtain", 2, "multiplicative", 9, 0.0),
        ("curtain_multiplicative_m3", "curtain", 3, "multiplicative", 9, 0.0),
        ("curtain_multiplicative_m4", "curtain", 4, "multiplicative", 9, 0.0),
        ("curtain_offgrid_m0", "curtain", 0, "additive", 9, 7.5),
        ("curtain_offgrid_m1", "curtain", 1, "additive", 9, 7.5),
        ("curtain_offgrid_m2", "curtain", 2, "additive", 9, 7.5),
        ("curtain_offgrid_m3", "curtain", 3, "additive", 9, 7.5),
        ("curtain_offgrid_m4", "curtain", 4, "additive", 9, 7.5),
    ]


def test_seed_1234_canonical_pattern_bytes_remain_compatible() -> None:
    expected_hashes = {
        "curtain_m0": "da03ba59da31f251370cf5f7d450b84fea3dbc7c498f5fc5d4ac44f3fb45885e",
        "curtain_m1": "096e01b4781a91b57cac3f0d4fd886bc10db6d419894af0b0b7184eb3fd6fa68",
        "curtain_m2": "8113b899b55f360846c5c99e83f798945ed87e434c50f332e17585ea303d0d40",
        "curtain_m3": "110e96fbaa45ccfa4f0f05d3d32101b8ac46dd89504f44ebf22da698e6991576",
        "curtain_m4": "bc67c65f1d32f75a27d913017e320962f6991b9f42a67b6d473cd8a09a88f30a",
        "sparse_m0": "ff38d52c8bf2dc96fef9d2cd3afd0e1d4021615a8471b0b3f3c0fe39552ee274",
        "nonstationary_m0": "7a053650f96161a878a648ac695aa9a1b9d74d417d38d85324bf05fd93db827a",
    }

    actual_hashes = {}
    for spec in default_pattern_specs()[:7]:
        pattern = make_stripe_pattern(
            shape=(33, 35),
            kind=spec.kind,
            mode=spec.mode,
            rng=np.random.default_rng(1234),
            profile_scale=spec.profile_scale,
            angle_offset=spec.angle_offset,
        )
        actual_hashes[spec.name] = hashlib.sha256(pattern.tobytes()).hexdigest()

    assert actual_hashes == expected_hashes


def test_offgrid_and_profile_scale_variants_change_pattern_geometry() -> None:
    canonical = make_stripe_pattern(
        shape=(33, 35),
        kind="curtain",
        mode=0,
        rng=np.random.default_rng(1234),
    )
    narrow = make_stripe_pattern(
        shape=(33, 35),
        kind="curtain",
        mode=0,
        rng=np.random.default_rng(1234),
        profile_scale=3,
    )
    broad = make_stripe_pattern(
        shape=(33, 35),
        kind="curtain",
        mode=0,
        rng=np.random.default_rng(1234),
        profile_scale=15,
    )
    offgrid = make_stripe_pattern(
        shape=(33, 35),
        kind="curtain",
        mode=0,
        rng=np.random.default_rng(1234),
        angle_offset=7.5,
    )

    assert not np.allclose(narrow, canonical)
    assert not np.allclose(broad, canonical)
    assert not np.allclose(offgrid, canonical)
    assert float(offgrid.mean()) == pytest.approx(0.0, abs=1e-12)
    assert float(offgrid.std()) == pytest.approx(1.0)


def test_offgrid_pattern_interpolates_rotated_continuous_coordinates() -> None:
    class DeterministicProfile:
        def normal(self, *, size: int) -> np.ndarray:
            positions = np.arange(size, dtype=np.float64)
            return positions**2 - 0.25 * positions

    shape = (7, 9)
    mode = 1
    angle_offset = 7.5
    profile_scale = 3
    row_step, col_step = PARALLEL_OFFSETS[mode]
    normal = np.array([col_step, -row_step], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    angle = np.deg2rad(angle_offset)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    rotated_normal = rotation @ normal
    rows, cols = np.indices(shape, dtype=np.float64)
    coordinates = rotated_normal[0] * rows + rotated_normal[1] * cols
    coordinates -= float(coordinates.min())
    line_count = int(np.ceil(coordinates.max())) + 1
    positions = np.arange(line_count, dtype=np.float64)
    profile = positions**2 - 0.25 * positions
    kernel = np.ones(profile_scale, dtype=np.float64) / profile_scale
    smoothed = np.convolve(profile, kernel, mode="same")
    expected = np.interp(coordinates, positions, smoothed)
    expected -= float(expected.mean())
    expected /= float(expected.std())

    actual = make_stripe_pattern(
        shape=shape,
        kind="curtain",
        mode=mode,
        rng=DeterministicProfile(),  # type: ignore[arg-type]
        profile_scale=profile_scale,
        angle_offset=angle_offset,
    )

    assert np.any(np.abs(coordinates - np.round(coordinates)) > 1e-6)
    assert np.allclose(actual, expected)


def test_run_benchmark_keeps_real_stripe_sample_out_of_gt_metrics(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(8)
    for index in range(1, 6):
        suffix = ".jpeg" if index == 1 else ".png"
        image = np.clip(
            100 + 30 * rng.normal(size=(16, 18)),
            0,
            255,
        ).astype(np.uint8)
        cv2.imwrite(str(tmp_path / f"sample_{index:02d}{suffix}"), image)

    rows = run_benchmark(
        load_samples(tmp_path),
        pattern_specs=(PatternSpec(kind="curtain", mode=0),),
        strengths=(0.02,),
        process_size=None,
        seed=17,
    )

    assert len(rows) == 9
    real = rows[0]
    assert real["sample"] == "sample_01.jpeg"
    assert real["case_type"] == "real"
    assert real["pattern"] == "existing"
    assert real["seed"] == 17
    assert real["carrier"] == "additive"
    assert real["profile_scale"] == 9
    assert real["angle_offset"] == 0.0
    assert real["input_psnr"] is None
    assert real["output_psnr"] is None
    assert real["input_ssim"] is None
    assert real["output_ssim"] is None

    clean = [row for row in rows if row["case_type"] == "clean"]
    assert {row["sample"] for row in clean} == {
        "sample_02.png",
        "sample_03.png",
        "sample_04.png",
        "sample_05.png",
    }
    assert all(row["pattern"] == "none" for row in clean)
    assert all(row["strength"] == 0.0 for row in clean)
    assert all(row["input_psnr"] == float("inf") for row in clean)
    assert all(row["input_ssim"] == pytest.approx(1.0) for row in clean)
    assert all(row["output_psnr"] is not None for row in clean)
    assert all(row["output_ssim"] is not None for row in clean)

    synthetic = [row for row in rows if row["case_type"] == "synthetic"]
    assert {row["sample"] for row in synthetic} == {
        "sample_02.png",
        "sample_03.png",
        "sample_04.png",
        "sample_05.png",
    }
    assert all(row["case_type"] == "synthetic" for row in synthetic)
    assert all(row["input_psnr"] is not None for row in synthetic)
    assert all(row["output_psnr"] is not None for row in synthetic)
    assert all(row["input_ssim"] is not None for row in synthetic)
    assert all(row["output_ssim"] is not None for row in synthetic)
    assert all(row["seed"] == 17 for row in synthetic)
    assert all(row["carrier"] == "additive" for row in synthetic)
    assert all(row["profile_scale"] == 9 for row in synthetic)
    assert all(row["angle_offset"] == 0.0 for row in synthetic)
    assert all(set(row) == set(RESULT_FIELDS) for row in rows)


def test_cli_writes_csv_without_saving_generated_images(tmp_path: Path) -> None:
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    rng = np.random.default_rng(12)
    for index in range(1, 6):
        suffix = ".jpeg" if index == 1 else ".png"
        image = np.clip(
            110 + 25 * rng.normal(size=(16, 18)),
            0,
            255,
        ).astype(np.uint8)
        cv2.imwrite(str(asset_dir / f"sample_{index:02d}{suffix}"), image)
    output = tmp_path / "results.csv"

    result = main(
        [
            "--asset-dir",
            str(asset_dir),
            "--output",
            str(output),
            "--patterns",
            "curtain_m0",
            "--strengths",
            "0.02",
            "--process-size",
            "16",
        ]
    )

    assert result == 0
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10
    with output.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        assert {"seed", "carrier", "profile_scale", "angle_offset"} <= set(
            reader.fieldnames
        )
        assert {
            "level",
            "selected_directions",
            "mu1",
            "mu2",
            "confidence",
        }.isdisjoint(reader.fieldnames)
    assert not list(tmp_path.rglob("*.tif"))


def test_cli_rejects_removed_levels_option() -> None:
    with pytest.raises(SystemExit) as error:
        main(["--levels", "0"])

    assert error.value.code == 2


def test_performance_cli_rejects_removed_iterations_option() -> None:
    with pytest.raises(SystemExit) as error:
        performance_main(["--iterations", "1"])

    assert error.value.code == 2


def test_performance_uses_only_the_automatic_wrapper_options() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "benchmarks" / "performance.py"
    ).read_text(encoding="utf-8")

    assert "destripe(image, process_size=args.process_size)" in source
    assert "adaptive=" not in source
    assert "iterations=" not in source
    assert "device=" not in source


def test_readme_documents_simple_automatic_and_manual_paths() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )

    assert "clean = destripe(image, process_size=256)" in readme
    assert "UniversalStripeRemover" in readme
    assert "weak oblique" in readme.lower()
    assert "automatic robust directional profiles" in readme.lower()
    for stale in ("adaptive=", "Adaptive Mode", "tile-mu", "tiles="):
        assert stale not in readme


def test_project_metadata_and_ruff_notebook_scope_match_automatic_api() -> None:
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )

    assert "automatic robust directional profiles" in pyproject.lower()
    assert '[tool.ruff.lint.per-file-ignores]' in pyproject
    assert '"notebooks/*.ipynb" = ["E402"]' in pyproject


def test_cli_check_acceptance_returns_one_after_writing_failures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    rng = np.random.default_rng(13)
    for index in range(1, 6):
        suffix = ".jpeg" if index == 1 else ".png"
        image = np.clip(
            110 + 25 * rng.normal(size=(16, 18)),
            0,
            255,
        ).astype(np.uint8)
        cv2.imwrite(str(asset_dir / f"sample_{index:02d}{suffix}"), image)
    output = tmp_path / "results.csv"

    result = main(
        [
            "--asset-dir",
            str(asset_dir),
            "--output",
            str(output),
            "--patterns",
            "curtain_m0",
            "--strengths",
            "0.01",
            "--process-size",
            "16",
            "--check-acceptance",
        ]
    )

    assert result == 1
    assert output.exists()
    assert "acceptance:" in capsys.readouterr().out


def _canonical_rows_at(
    rows: list[dict[str, object]],
    strength: float,
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if (
            row["case_type"] == "synthetic"
            and row["strength"] == strength
            and row["carrier"] == "additive"
            and row["profile_scale"] == 9
            and row["angle_offset"] == 0.0
            and row["pattern"] == f"curtain_m{row['mode']}"
        )
    ]


def _vertical_weak_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [row for row in _canonical_rows_at(rows, 0.01) if row["mode"] == 0]


def test_acceptance_passes_complete_canonical_fixture() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)

    assert evaluate_acceptance(rows) == []


def test_acceptance_rejects_missing_robustness_pattern() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    rows = [row for row in rows if row["pattern"] != "curtain_offgrid_m4"]

    failures = evaluate_acceptance(rows)
    assert any(
        "missing" in failure and "curtain_offgrid_m4" in failure
        for failure in failures
    )


def test_acceptance_rejects_unexpected_robustness_pattern() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    unexpected = dict(
        next(row for row in rows if row["pattern"] == "curtain_offgrid_m4")
    )
    unexpected["pattern"] = "curtain_offgrid_m5"
    unexpected["mode"] = 5
    rows.append(unexpected)

    assert any(
        "unexpected pattern curtain_offgrid_m5" in failure
        for failure in evaluate_acceptance(rows)
    )


def test_acceptance_rejects_cross_combined_robustness_metadata() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    narrow = next(row for row in rows if row["pattern"] == "curtain_narrow_m0")
    narrow["carrier"] = "multiplicative"

    assert any(
        "metadata" in failure and "curtain_narrow_m0" in failure
        for failure in evaluate_acceptance(rows)
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mode", 0.5, "metadata mismatch"),
        ("profile_scale", 9.5, "metadata mismatch"),
        ("mode", False, "metadata mismatch"),
        ("profile_scale", float("inf"), "metadata mismatch"),
    ],
)
def test_acceptance_rejects_non_lossless_integer_metadata(
    field: str,
    value: object,
    message: str,
) -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    row = next(
        row
        for row in rows
        if row["pattern"] == "curtain_m0"
        and row["strength"] == 0.01
        and row["sample"] == "sample_02.png"
    )
    row[field] = value

    try:
        failures = evaluate_acceptance(rows)
    except (OverflowError, TypeError, ValueError) as error:
        pytest.fail(f"acceptance must report invalid {field}, not raise {error!r}")
    assert any(message in failure for failure in failures)


@pytest.mark.parametrize("mode", ["junk", float("nan"), float("inf")])
def test_acceptance_rejects_invalid_clean_mode(mode: object) -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    clean = next(row for row in rows if row["case_type"] == "clean")
    clean["mode"] = mode

    try:
        failures = evaluate_acceptance(rows)
    except (OverflowError, TypeError, ValueError) as error:
        pytest.fail(f"acceptance must report invalid clean mode, not raise {error!r}")
    assert any("metadata mismatch for clean row" in failure for failure in failures)


def test_acceptance_accepts_csv_integer_strings_and_blank_clean_mode() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    for row in rows:
        row["seed"] = str(row["seed"])
        row["profile_scale"] = str(row["profile_scale"])
        row["mode"] = "" if row["case_type"] == "clean" else str(row["mode"])

    assert evaluate_acceptance(rows) == []


@pytest.mark.parametrize(
    ("metric", "gain", "message"),
    [
        ("output_psnr", 0.09, "weak mean PSNR"),
        ("output_ssim", 0.0009, "weak mean SSIM"),
    ],
)
def test_acceptance_rejects_low_weak_mean_gain(
    metric: str,
    gain: float,
    message: str,
) -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    for row in _vertical_weak_rows(rows):
        input_metric = metric.replace("output", "input")
        row[metric] = float(row[input_metric]) + gain

    assert any(message in failure for failure in evaluate_acceptance(rows))


def test_acceptance_excludes_duplicate_identity_from_gate_weighting() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    weak = _vertical_weak_rows(rows)
    for row in weak:
        row["output_psnr"] = float(row["input_psnr"]) + 0.09
    duplicate = dict(weak[0])
    duplicate["output_psnr"] = float(duplicate["input_psnr"]) + 10.0
    rows.extend(dict(duplicate) for _ in range(20))

    failures = evaluate_acceptance(rows)
    assert any("duplicate row" in failure for failure in failures)
    assert any("weak mean PSNR" in failure for failure in failures)


def test_acceptance_rejects_low_weak_case_coverage() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    for row in _vertical_weak_rows(rows)[:2]:
        row["output_psnr"] = float(row["input_psnr"]) + 0.01
        row["output_ssim"] = float(row["input_ssim"]) + 0.00001

    assert any("weak coverage" in failure for failure in evaluate_acceptance(rows))


def test_acceptance_rejects_any_weak_loss_worse_than_one_db() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    weak = _vertical_weak_rows(rows)
    weak[0]["output_psnr"] = float(weak[0]["input_psnr"]) - 1.01

    assert any("weak PSNR loss" in failure for failure in evaluate_acceptance(rows))


def test_acceptance_rejects_negative_vertical_weak_mean() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    for row in _vertical_weak_rows(rows):
        row["output_psnr"] = float(row["input_psnr"]) - 0.1

    assert any("weak mean PSNR" in failure for failure in evaluate_acceptance(rows))


def test_acceptance_reports_but_does_not_gate_weak_oblique_rows() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    for row in _canonical_rows_at(rows, 0.01):
        if row["mode"] != 0:
            row["output_psnr"] = float(row["input_psnr"]) - 10.0
            row["output_ssim"] = float(row["input_ssim"]) - 0.1
            row["stripe_projection_left_pct"] = 100.0

    assert evaluate_acceptance(rows) == []


def test_diagnostic_summary_reports_weak_oblique_and_robustness_metrics() -> None:
    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)

    lines = diagnostic_summary_lines(rows)

    weak_oblique = [line for line in lines if "weak-oblique" in line]
    robustness = [line for line in lines if "robustness" in line]
    assert len(weak_oblique) == 4
    assert len(robustness) == 4
    assert all("psnr_gain=+0.200000" in line for line in weak_oblique)
    assert all("ssim_gain=+0.002000" in line for line in weak_oblique)
    assert all("projection_left=60.000%" in line for line in weak_oblique)
    assert all("joint_coverage=100.0%" in line for line in weak_oblique)
    assert all("worst_gain=+0.200000" in line for line in weak_oblique)
    assert any("robustness weak" in line and "cases=48" in line for line in lines)
    assert any("robustness pooled" in line and "cases=144" in line for line in lines)


def test_acceptance_rejects_low_projection_case_coverage() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    for index, row in enumerate(_vertical_weak_rows(rows)):
        row["stripe_projection_left_pct"] = 86.0 if index < 2 else 50.0

    failures = evaluate_acceptance(rows)
    assert any("projection coverage" in failure for failure in failures)
    assert not any("mean projection" in failure for failure in failures)


@pytest.mark.parametrize(
    ("strength", "gain", "message"),
    [
        (0.03, 0.668, "medium mean PSNR"),
        (0.06, 3.532, "strong mean PSNR"),
    ],
)
def test_acceptance_rejects_canonical_baseline_regression(
    strength: float,
    gain: float,
    message: str,
) -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    for row in _canonical_rows_at(rows, strength):
        row["output_psnr"] = float(row["input_psnr"]) + gain

    assert any(message in failure for failure in evaluate_acceptance(rows))


def test_acceptance_rejects_clean_ssim_below_absolute_gate() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    clean = next(row for row in rows if row["case_type"] == "clean")
    clean["output_ssim"] = 0.989

    assert any("clean SSIM" in failure for failure in evaluate_acceptance(rows))


@pytest.mark.parametrize("input_psnr", [100.0, float("inf")])
def test_acceptance_allows_finite_or_positive_infinite_clean_input_psnr(
    input_psnr: float,
) -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    for row in rows:
        if row["case_type"] == "clean":
            row["input_psnr"] = input_psnr

    assert evaluate_acceptance(rows) == []


@pytest.mark.parametrize("output_psnr", [100.0, float("inf")])
def test_acceptance_allows_finite_or_positive_infinite_clean_output_psnr(
    output_psnr: float,
) -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    for row in rows:
        if row["case_type"] == "clean":
            row["output_psnr"] = output_psnr

    assert evaluate_acceptance(rows) == []


@pytest.mark.parametrize(
    "input_psnr",
    [float("nan"), float("-inf"), None, "not-a-number"],
)
def test_acceptance_rejects_invalid_clean_input_psnr(input_psnr: object) -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    clean = next(row for row in rows if row["case_type"] == "clean")
    clean["input_psnr"] = input_psnr

    assert any(
        "clean input_psnr" in failure for failure in evaluate_acceptance(rows)
    )


def test_acceptance_failure_order_is_independent_of_row_order() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=100.0, clean_psnr=39.9)

    assert evaluate_acceptance(rows) == evaluate_acceptance(list(reversed(rows)))


@pytest.mark.parametrize(
    ("omission", "message"),
    [
        ("sample", "missing sample"),
        ("strength", "missing strength"),
        ("mode", "missing mode"),
    ],
)
def test_acceptance_rejects_incomplete_canonical_matrix(
    omission: str,
    message: str,
) -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    if omission == "sample":
        rows = [
            row
            for row in rows
            if not (
                row["sample"] == "sample_02.png"
                and row["pattern"] == "curtain_m0"
                and row["strength"] == 0.01
            )
        ]
    elif omission == "strength":
        rows = [row for row in rows if row["strength"] != 0.03]
    else:
        rows = [
            row
            for row in rows
            if not (row["case_type"] == "synthetic" and row["mode"] == 4)
        ]

    assert any(message in failure for failure in evaluate_acceptance(rows))


def test_acceptance_rejects_missing_sample_identity() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    rows = [
        row
        for row in rows
        if not (
            row["pattern"] == "curtain_m0"
            and row["strength"] == 0.01
            and row["sample"] == "sample_02.png"
        )
    ]

    assert any(
        "missing row" in failure and "sample_02.png" in failure
        for failure in evaluate_acceptance(rows)
    )


def test_acceptance_rejects_nonfinite_synthetic_metric() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    synthetic = next(row for row in rows if row["case_type"] == "synthetic")
    synthetic["output_psnr"] = float("nan")

    assert any("non-finite" in failure for failure in evaluate_acceptance(rows))


def _robustness_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row["case_type"] == "synthetic"
        and (
            row["profile_scale"] in {3, 15}
            or row["carrier"] == "multiplicative"
            or row["angle_offset"] != 0.0
        )
    ]


def test_acceptance_does_not_apply_canonical_baseline_to_robustness_rows() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    for row in _robustness_rows(rows):
        row["output_psnr"] = row["input_psnr"]

    assert evaluate_acceptance(rows) == []


def test_acceptance_pools_robustness_mean_across_all_variants() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    for row in _robustness_rows(rows):
        gain = -0.5 if row["pattern"] == "curtain_narrow_m0" else 0.1
        row["output_psnr"] = float(row["input_psnr"]) + gain

    assert evaluate_acceptance(rows) == []


def test_acceptance_does_not_gate_negative_pooled_robustness_mean() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    for row in _robustness_rows(rows):
        row["output_psnr"] = float(row["input_psnr"]) - 0.1

    assert evaluate_acceptance(rows) == []


def test_acceptance_does_not_gate_robustness_worst_case() -> None:
    from benchmarks.acceptance import evaluate_acceptance

    rows = make_acceptance_fixture(weak_projection_left=60.0, clean_psnr=100.0)
    robustness = _robustness_rows(rows)
    for row in robustness:
        row["output_psnr"] = float(row["input_psnr"]) + 0.1
    robustness[0]["output_psnr"] = float(robustness[0]["input_psnr"]) - 1.1

    assert evaluate_acceptance(rows) == []
