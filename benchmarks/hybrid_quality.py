import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from benchmarks.synthetic import (
    inject_stripe,
    load_samples,
    make_stripe_pattern,
    make_support_mask,
    structural_similarity,
)
from destripe import destripe
from destripe.automatic import automatic_clean
from destripe.preprocess import prepare_solver_gray


REGRESSION_SEEDS = (1234, 20260710, 20260711)
HELD_BACK_INTERRUPTION_SEED = 20260712
STRENGTHS = (0.01, 0.03, 0.06)
SUPPORT_KINDS = ("outer_quarters", "first_half", "center", "segments")


@dataclass(frozen=True)
class QualityRow:
    sample: str
    case: str
    mode: int | None
    strength: float
    input_psnr: float
    output_psnr: float
    input_ssim: float
    output_ssim: float
    unsupported_input_mse: float
    unsupported_output_mse: float


@dataclass(frozen=True)
class TimingRow:
    sample: str
    detection_seconds: float
    solver_seconds: float
    total_seconds: float
    candidate_count: int
    iterations: int


def run_quality(
    *,
    asset_dir: str | Path,
    process_size: int | None,
    seed: int,
) -> tuple[list[QualityRow], TimingRow]:
    samples = load_samples(asset_dir)
    clean_samples = [sample for sample in samples if sample.has_ground_truth]
    if not clean_samples:
        raise ValueError("quality suite requires at least one clean sample.")

    rows: list[QualityRow] = []
    for sample_index, sample in enumerate(clean_samples):
        clean_output = destripe(sample.image, process_size=process_size)
        rows.append(
            _quality_row(
                sample=sample.name,
                case="clean",
                clean=sample.image,
                observed=sample.image,
                output=clean_output,
                mode=None,
                strength=0.0,
            )
        )

        for mode in range(5):
            for strength_index, strength in enumerate(STRENGTHS):
                rng = np.random.default_rng(
                    seed + 100_000 * sample_index + 1_000 * mode + strength_index
                )
                pattern = make_stripe_pattern(
                    shape=sample.image.shape,
                    kind="curtain",
                    mode=mode,
                    rng=rng,
                )
                observed, _ = inject_stripe(
                    sample.image,
                    pattern,
                    strength=strength,
                )
                output = destripe(observed, process_size=process_size)
                rows.append(
                    _quality_row(
                        sample=sample.name,
                        case="continuous",
                        clean=sample.image,
                        observed=observed,
                        output=output,
                        mode=mode,
                        strength=strength,
                    )
                )

            interruption_rng = np.random.default_rng(
                seed + 100_000 * sample_index + 1_000 * mode + 99
            )
            interruption_pattern = make_stripe_pattern(
                shape=sample.image.shape,
                kind="curtain",
                mode=mode,
                rng=interruption_rng,
            )
            for support_index, support_kind in enumerate(SUPPORT_KINDS):
                support = make_support_mask(
                    sample.image.shape,
                    kind=support_kind,
                    mode=mode,
                    rng=np.random.default_rng(
                        seed
                        + 100_000 * sample_index
                        + 1_000 * mode
                        + 200
                        + support_index
                    ),
                )
                observed, _ = inject_stripe(
                    sample.image,
                    interruption_pattern * support,
                    strength=0.03,
                )
                output = destripe(observed, process_size=process_size)
                rows.append(
                    _quality_row(
                        sample=sample.name,
                        case=f"interrupted:{support_kind}",
                        clean=sample.image,
                        observed=observed,
                        output=output,
                        mode=mode,
                        strength=0.03,
                        unsupported=support == 0.0,
                    )
                )

    timing_sample = next(
        (sample for sample in samples if not sample.has_ground_truth),
        clean_samples[0],
    )
    normalized = timing_sample.image.astype(np.float64)
    low = float(normalized.min())
    scale = float(normalized.max()) - low
    if scale > 1e-12:
        normalized = (normalized - low) / scale
    processed = prepare_solver_gray(gray=normalized, process_size=process_size)
    total_started = time.perf_counter()
    automatic = automatic_clean(processed, proj=True)
    total_seconds = time.perf_counter() - total_started
    timing = TimingRow(
        sample=timing_sample.name,
        detection_seconds=automatic.detection_seconds,
        solver_seconds=automatic.solver_seconds,
        total_seconds=total_seconds,
        candidate_count=automatic.candidate_count,
        iterations=automatic.iterations,
    )
    return rows, timing


def quality_failures(rows: list[QualityRow]) -> list[str]:
    failures: list[str] = []
    clean_rows = [row for row in rows if row.case == "clean"]
    for row in clean_rows:
        if row.output_psnr < 40.0:
            failures.append(f"{row.sample}: clean PSNR {row.output_psnr:.3f} < 40")
        if row.output_ssim < 0.99:
            failures.append(f"{row.sample}: clean SSIM {row.output_ssim:.6f} < 0.99")

    for strength in STRENGTHS:
        selected = [
            row
            for row in rows
            if row.case == "continuous" and row.strength == strength
        ]
        if selected:
            input_psnr = float(np.mean([row.input_psnr for row in selected]))
            output_psnr = float(np.mean([row.output_psnr for row in selected]))
            input_ssim = float(np.mean([row.input_ssim for row in selected]))
            output_ssim = float(np.mean([row.output_ssim for row in selected]))
            psnr_failed = (
                output_psnr < input_psnr
                if strength == STRENGTHS[0]
                else output_psnr <= input_psnr
            )
            ssim_failed = (
                output_ssim < input_ssim
                if strength == STRENGTHS[0]
                else output_ssim <= input_ssim
            )
            if psnr_failed:
                failures.append(
                    f"strength {strength}: mean PSNR did not improve or preserve "
                    f"({input_psnr:.3f} -> {output_psnr:.3f})"
                )
            if ssim_failed:
                failures.append(
                    f"strength {strength}: mean SSIM did not improve or preserve "
                    f"({input_ssim:.6f} -> {output_ssim:.6f})"
                )

    for row in rows:
        if not row.case.startswith("interrupted:"):
            continue
        tolerance = 1e-12
        if row.unsupported_output_mse > row.unsupported_input_mse + tolerance:
            failures.append(
                f"{row.sample} {row.case} m{row.mode}: unsupported MSE "
                f"{row.unsupported_input_mse:.3e} -> "
                f"{row.unsupported_output_mse:.3e}"
            )
    return failures


def _quality_row(
    *,
    sample: str,
    case: str,
    clean: np.ndarray,
    observed: np.ndarray,
    output: np.ndarray,
    mode: int | None,
    strength: float,
    unsupported: np.ndarray | None = None,
) -> QualityRow:
    if unsupported is None:
        unsupported = np.zeros(clean.shape, dtype=bool)
    return QualityRow(
        sample=sample,
        case=case,
        mode=mode,
        strength=strength,
        input_psnr=_psnr(clean, observed),
        output_psnr=_psnr(clean, output),
        input_ssim=structural_similarity(clean, observed),
        output_ssim=structural_similarity(clean, output),
        unsupported_input_mse=_masked_mse(observed, clean, unsupported),
        unsupported_output_mse=_masked_mse(output, clean, unsupported),
    )


def _psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = float(np.mean((candidate - reference) ** 2))
    if mse <= 1e-15:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _masked_mse(
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> float:
    if not np.any(mask):
        return 0.0
    difference = candidate[mask] - reference[mask]
    return float(np.mean(difference * difference))


def _summary(rows: list[QualityRow], timing: TimingRow) -> list[str]:
    lines: list[str] = []
    for row in rows:
        if row.case == "clean":
            lines.append(
                f"clean {row.sample}: PSNR={row.output_psnr:.3f}, "
                f"SSIM={row.output_ssim:.6f}"
            )
    for strength in STRENGTHS:
        selected = [
            row
            for row in rows
            if row.case == "continuous" and row.strength == strength
        ]
        lines.append(
            f"continuous {strength:.2f}: mean PSNR "
            f"{np.mean([row.input_psnr for row in selected]):.3f} -> "
            f"{np.mean([row.output_psnr for row in selected]):.3f}, "
            f"SSIM {np.mean([row.input_ssim for row in selected]):.6f} -> "
            f"{np.mean([row.output_ssim for row in selected]):.6f}"
        )
    interrupted = [row for row in rows if row.case.startswith("interrupted:")]
    lines.append(
        "interrupted worst unsupported MSE: "
        f"{max(row.unsupported_output_mse for row in interrupted):.3e}"
    )
    lines.append(
        f"timing {timing.sample}: detect={timing.detection_seconds:.3f}s, "
        f"PDHG={timing.solver_seconds:.3f}s, total={timing.total_seconds:.3f}s, "
        f"candidates={timing.candidate_count}, iterations={timing.iterations}"
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run H3-guided PDHG quality gates.")
    parser.add_argument("--asset-dir", default="asset")
    parser.add_argument("--process-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=REGRESSION_SEEDS[0])
    parser.add_argument("--held-back", action="store_true")
    args = parser.parse_args(argv)
    seed = HELD_BACK_INTERRUPTION_SEED if args.held_back else args.seed
    rows, timing = run_quality(
        asset_dir=args.asset_dir,
        process_size=args.process_size,
        seed=seed,
    )
    for line in _summary(rows, timing):
        print(line)
    failures = quality_failures(rows)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("quality: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
