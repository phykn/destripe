import argparse
import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from benchmarks.acceptance import evaluate_acceptance
from destripe import destripe
from destripe.adaptive import AdaptiveParams, estimate_adaptive_params
from destripe.adaptive.constants import PARALLEL_OFFSETS


RESULT_FIELDS = (
    "seed",
    "sample",
    "case_type",
    "pattern",
    "mode",
    "strength",
    "carrier",
    "profile_scale",
    "angle_offset",
    "level",
    "selected_directions",
    "mu1",
    "mu2",
    "confidence",
    "input_psnr",
    "output_psnr",
    "input_ssim",
    "output_ssim",
    "stripe_projection_left_pct",
    "removed_rmse",
)


@dataclass(frozen=True)
class BenchmarkImage:
    name: str
    image: np.ndarray
    has_ground_truth: bool


@dataclass(frozen=True)
class PatternSpec:
    kind: str
    mode: int
    carrier: str = "additive"
    profile_scale: int = 9
    angle_offset: float = 0.0

    @property
    def name(self) -> str:
        parts = [self.kind]
        if self.profile_scale == 3:
            parts.append("narrow")
        elif self.profile_scale == 15:
            parts.append("broad")
        elif self.profile_scale != 9:
            parts.append(f"scale{self.profile_scale}")
        if self.carrier != "additive":
            parts.append(self.carrier)
        if self.angle_offset != 0.0:
            parts.append("offgrid")
        return f"{'_'.join(parts)}_m{self.mode}"


def default_pattern_specs() -> tuple[PatternSpec, ...]:
    canonical = tuple(
        PatternSpec(kind="curtain", mode=mode) for mode in range(5)
    ) + (
        PatternSpec(kind="sparse", mode=0),
        PatternSpec(kind="nonstationary", mode=0),
    )
    robustness = (
        PatternSpec(kind="curtain", mode=0, profile_scale=3),
        PatternSpec(kind="curtain", mode=0, profile_scale=15),
    ) + tuple(
        PatternSpec(kind="curtain", mode=mode, carrier="multiplicative")
        for mode in range(5)
    ) + tuple(
        PatternSpec(kind="curtain", mode=mode, angle_offset=7.5)
        for mode in range(5)
    )
    return canonical + robustness


def make_stripe_pattern(
    *,
    shape: tuple[int, int],
    kind: str,
    mode: int,
    rng: np.random.Generator,
    profile_scale: int = 9,
    angle_offset: float = 0.0,
) -> np.ndarray:
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("shape must contain two dimensions of at least 2 pixels.")
    if mode not in PARALLEL_OFFSETS:
        raise ValueError("mode must be an integer from 0 through 4.")
    if kind not in {"curtain", "sparse", "nonstationary"}:
        raise ValueError("kind must be curtain, sparse, or nonstationary.")
    if not isinstance(profile_scale, int) or profile_scale <= 0:
        raise ValueError("profile_scale must be a positive integer.")
    if not np.isfinite(angle_offset):
        raise ValueError("angle_offset must be finite.")

    rows, cols = np.indices(shape, dtype=np.int64)
    row_step, col_step = PARALLEL_OFFSETS[mode]
    shifted_ids: np.ndarray | None
    continuous_coordinates: np.ndarray | None
    if angle_offset == 0.0:
        line_ids = col_step * rows - row_step * cols
        shifted_ids = line_ids - int(line_ids.min())
        continuous_coordinates = None
        line_count = int(shifted_ids.max()) + 1
    else:
        normal = np.array([col_step, -row_step], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        angle = np.deg2rad(angle_offset)
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        rotated_normal = np.array(
            [
                cosine * normal[0] - sine * normal[1],
                sine * normal[0] + cosine * normal[1],
            ]
        )
        coordinates = rotated_normal[0] * rows + rotated_normal[1] * cols
        continuous_coordinates = coordinates - float(coordinates.min())
        shifted_ids = None
        line_count = int(np.ceil(continuous_coordinates.max())) + 1

    profile = rng.normal(size=line_count)
    if kind == "sparse":
        sparse_profile = np.zeros(line_count, dtype=np.float64)
        selected = rng.choice(
            line_count,
            size=max(2, line_count // 16),
            replace=False,
        )
        sparse_profile[selected] = profile[selected]
        profile = sparse_profile
    else:
        kernel_size = min(profile_scale, line_count)
        if kernel_size % 2 == 0:
            kernel_size -= 1
        kernel = np.ones(kernel_size, dtype=np.float64) / kernel_size
        profile = np.convolve(profile, kernel, mode="same")

    if continuous_coordinates is None:
        assert shifted_ids is not None
        pattern = profile[shifted_ids]
    else:
        pattern = np.interp(
            continuous_coordinates,
            np.arange(line_count, dtype=np.float64),
            profile,
        )
    if kind == "nonstationary":
        envelope = 0.2 + 0.8 * np.sin(
            np.pi * (rows + 0.5) / shape[0]
        ) ** 2
        pattern = pattern * envelope

    pattern = pattern - float(pattern.mean())
    scale = float(pattern.std())
    if scale <= 1e-12:
        raise ValueError("stripe pattern has no variation.")
    return pattern / scale


def load_samples(asset_dir: str | Path) -> list[BenchmarkImage]:
    root = Path(asset_dir)
    extensions = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    paths = sorted(
        path
        for path in root.glob("sample_*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not paths:
        raise ValueError(f"no sample images found in {root}.")

    samples = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 2:
            raise ValueError(f"sample must be a grayscale image: {path}.")
        if np.issubdtype(image.dtype, np.integer):
            scale = float(np.iinfo(image.dtype).max)
            normalized = image.astype(np.float64) / scale
        else:
            normalized = np.asarray(image, dtype=np.float64)
        if not np.isfinite(normalized).all():
            raise ValueError(f"sample must contain finite values: {path}.")
        samples.append(
            BenchmarkImage(
                name=path.name,
                image=np.clip(normalized, 0.0, 1.0),
                has_ground_truth=path.stem != "sample_01",
            )
        )
    return samples


def inject_stripe(
    clean: np.ndarray,
    pattern: np.ndarray,
    *,
    strength: float,
    carrier: str = "additive",
) -> tuple[np.ndarray, np.ndarray]:
    clean_array = np.asarray(clean, dtype=np.float64)
    pattern_array = np.asarray(pattern, dtype=np.float64)
    if clean_array.shape != pattern_array.shape:
        raise ValueError("clean and pattern must have the same shape.")
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError("strength must be a positive finite number.")
    if carrier == "additive":
        proposed = clean_array + strength * pattern_array
    elif carrier == "multiplicative":
        proposed = clean_array * (1.0 + strength * pattern_array)
    else:
        raise ValueError("carrier must be additive or multiplicative.")
    observed = np.clip(proposed, 0.0, 1.0)
    return observed, observed - clean_array


def structural_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    first = np.asarray(reference, dtype=np.float64)
    second = np.asarray(candidate, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("images must have the same two-dimensional shape.")

    mu_first = cv2.GaussianBlur(first, (11, 11), 1.5)
    mu_second = cv2.GaussianBlur(second, (11, 11), 1.5)
    first_sq = mu_first * mu_first
    second_sq = mu_second * mu_second
    mean_product = mu_first * mu_second
    variance_first = cv2.GaussianBlur(first * first, (11, 11), 1.5) - first_sq
    variance_second = cv2.GaussianBlur(second * second, (11, 11), 1.5) - second_sq
    covariance = cv2.GaussianBlur(first * second, (11, 11), 1.5) - mean_product
    c1 = 0.01**2
    c2 = 0.03**2
    score = (
        (2 * mean_product + c1) * (2 * covariance + c2)
        / ((first_sq + second_sq + c1) * (variance_first + variance_second + c2))
    )
    return float(np.mean(score))


def run_benchmark(
    samples: list[BenchmarkImage],
    *,
    pattern_specs: tuple[PatternSpec, ...] | list[PatternSpec],
    strengths: tuple[float, ...] | list[float],
    levels: tuple[int, ...] | list[int],
    iterations: int,
    process_size: int | None,
    seed: int,
    device: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_index, sample in enumerate(samples):
        if not sample.has_ground_truth:
            for level in levels:
                params = estimate_adaptive_params(sample.image, level=level)
                output = destripe(
                    sample.image,
                    adaptive=level,
                    iterations=iterations,
                    process_size=process_size,
                    device=device,
                )
                rows.append(
                    _make_row(
                        seed=seed,
                        sample=sample.name,
                        case_type="real",
                        pattern="existing",
                        mode=None,
                        strength=None,
                        carrier="additive",
                        profile_scale=9,
                        angle_offset=0.0,
                        level=level,
                        params=params,
                        input_psnr=None,
                        output_psnr=None,
                        input_ssim=None,
                        output_ssim=None,
                        stripe_projection_left_pct=None,
                        removed_rmse=_rmse(sample.image, output),
                    )
                )
            continue

        for level in levels:
            params = estimate_adaptive_params(sample.image, level=level)
            output = destripe(
                sample.image,
                adaptive=level,
                iterations=iterations,
                process_size=process_size,
                device=device,
            )
            rows.append(
                _make_row(
                    seed=seed,
                    sample=sample.name,
                    case_type="clean",
                    pattern="none",
                    mode=None,
                    strength=0.0,
                    carrier="additive",
                    profile_scale=9,
                    angle_offset=0.0,
                    level=level,
                    params=params,
                    input_psnr=math.inf,
                    output_psnr=_psnr(sample.image, output),
                    input_ssim=1.0,
                    output_ssim=structural_similarity(sample.image, output),
                    stripe_projection_left_pct=0.0,
                    removed_rmse=_rmse(sample.image, output),
                )
            )

        for pattern_index, spec in enumerate(pattern_specs):
            for strength_index, strength in enumerate(strengths):
                rng = np.random.default_rng(
                    seed + sample_index * 10_000 + pattern_index * 100 + strength_index
                )
                pattern = make_stripe_pattern(
                    shape=sample.image.shape,
                    kind=spec.kind,
                    mode=spec.mode,
                    rng=rng,
                    profile_scale=spec.profile_scale,
                    angle_offset=spec.angle_offset,
                )
                observed, actual_stripe = inject_stripe(
                    sample.image,
                    pattern,
                    strength=strength,
                    carrier=spec.carrier,
                )
                for level in levels:
                    params = estimate_adaptive_params(observed, level=level)
                    output = destripe(
                        observed,
                        adaptive=level,
                        iterations=iterations,
                        process_size=process_size,
                        device=device,
                    )
                    rows.append(
                        _make_row(
                            seed=seed,
                            sample=sample.name,
                            case_type="synthetic",
                            pattern=spec.name,
                            mode=spec.mode,
                            strength=strength,
                            carrier=spec.carrier,
                            profile_scale=spec.profile_scale,
                            angle_offset=spec.angle_offset,
                            level=level,
                            params=params,
                            input_psnr=_psnr(sample.image, observed),
                            output_psnr=_psnr(sample.image, output),
                            input_ssim=structural_similarity(sample.image, observed),
                            output_ssim=structural_similarity(sample.image, output),
                            stripe_projection_left_pct=_stripe_projection_left(
                                clean=sample.image,
                                output=output,
                                actual_stripe=actual_stripe,
                            ),
                            removed_rmse=_rmse(observed, output),
                        )
                    )
    return rows


def _make_row(
    *,
    seed: int,
    sample: str,
    case_type: str,
    pattern: str,
    mode: int | None,
    strength: float | None,
    carrier: str,
    profile_scale: int,
    angle_offset: float,
    level: int,
    params: AdaptiveParams,
    input_psnr: float | None,
    output_psnr: float | None,
    input_ssim: float | None,
    output_ssim: float | None,
    stripe_projection_left_pct: float | None,
    removed_rmse: float,
) -> dict[str, object]:
    return {
        "seed": seed,
        "sample": sample,
        "case_type": case_type,
        "pattern": pattern,
        "mode": mode,
        "strength": strength,
        "carrier": carrier,
        "profile_scale": profile_scale,
        "angle_offset": angle_offset,
        "level": level,
        "selected_directions": params.directions,
        "mu1": params.mu1,
        "mu2": params.mu2,
        "confidence": params.confidence,
        "input_psnr": input_psnr,
        "output_psnr": output_psnr,
        "input_ssim": input_ssim,
        "output_ssim": output_ssim,
        "stripe_projection_left_pct": stripe_projection_left_pct,
        "removed_rmse": removed_rmse,
    }


def _rmse(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(first) - np.asarray(second)) ** 2)))


def _psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    error = _rmse(reference, candidate)
    if error <= 0.0:
        return math.inf
    return -20.0 * math.log10(error)


def _stripe_projection_left(
    *,
    clean: np.ndarray,
    output: np.ndarray,
    actual_stripe: np.ndarray,
) -> float:
    denominator = float(np.sum(actual_stripe * actual_stripe))
    if denominator <= 1e-18:
        return 0.0
    residual = np.asarray(output) - np.asarray(clean)
    return 100.0 * abs(float(np.sum(residual * actual_stripe))) / denominator


def write_results(path: str | Path, rows: list[dict[str, object]]) -> None:
    output = Path(path)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            directions = serialized["selected_directions"]
            serialized["selected_directions"] = "|".join(
                str(mode) for mode in directions
            )
            writer.writerow(serialized)


def main(argv: Sequence[str] | None = None) -> int:
    specs = default_pattern_specs()
    specs_by_name = {spec.name: spec for spec in specs}
    parser = argparse.ArgumentParser(
        description="Evaluate adaptive destriping with synthetic stripe pairs.",
    )
    parser.add_argument("--asset-dir", type=Path, default=Path("asset"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("synthetic_benchmark.csv"),
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=tuple(specs_by_name),
        default=list(specs_by_name),
    )
    parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        default=[0.01, 0.03, 0.06],
    )
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--process-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--check-acceptance", action="store_true")
    args = parser.parse_args(argv)

    rows = run_benchmark(
        load_samples(args.asset_dir),
        pattern_specs=tuple(specs_by_name[name] for name in args.patterns),
        strengths=tuple(args.strengths),
        levels=tuple(args.levels),
        iterations=args.iterations,
        process_size=args.process_size,
        seed=args.seed,
        device=args.device,
    )
    write_results(args.output, rows)
    real_count = sum(row["case_type"] == "real" for row in rows)
    clean_count = sum(row["case_type"] == "clean" for row in rows)
    synthetic_count = sum(row["case_type"] == "synthetic" for row in rows)
    print(
        f"Wrote {len(rows)} rows to {args.output} "
        f"({real_count} real-only, {clean_count} clean, "
        f"{synthetic_count} synthetic)."
    )
    if args.check_acceptance:
        failures = evaluate_acceptance(rows)
        for failure in failures:
            print(f"acceptance: {failure}")
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
