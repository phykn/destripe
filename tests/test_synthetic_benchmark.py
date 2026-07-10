from pathlib import Path

import cv2
import numpy as np
import pytest

from benchmarks.synthetic import (
    PatternSpec,
    default_pattern_specs,
    inject_stripe,
    load_samples,
    main,
    make_stripe_pattern,
    run_benchmark,
    structural_similarity,
)
from destripe.adaptive.constants import PARALLEL_OFFSETS


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


def test_structural_similarity_is_one_for_identical_images() -> None:
    image = np.random.default_rng(5).random((32, 32))

    assert structural_similarity(image, image) == pytest.approx(1.0)


def test_default_patterns_cover_all_directions_and_two_vertical_guards() -> None:
    specs = default_pattern_specs()

    assert [(spec.kind, spec.mode) for spec in specs] == [
        ("curtain", 0),
        ("curtain", 1),
        ("curtain", 2),
        ("curtain", 3),
        ("curtain", 4),
        ("sparse", 0),
        ("nonstationary", 0),
    ]


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
        levels=(0,),
        iterations=1,
        process_size=None,
        seed=17,
        device="cpu",
    )

    assert len(rows) == 5
    real = rows[0]
    assert real["sample"] == "sample_01.jpeg"
    assert real["case_type"] == "real"
    assert real["pattern"] == "existing"
    assert real["input_psnr"] is None
    assert real["output_psnr"] is None
    assert real["input_ssim"] is None
    assert real["output_ssim"] is None

    synthetic = rows[1:]
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
            "--levels",
            "0",
            "--iterations",
            "1",
            "--process-size",
            "16",
        ]
    )

    assert result == 0
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    assert lines[0].startswith("sample,case_type,pattern,")
    assert not list(tmp_path.rglob("*.tif"))
