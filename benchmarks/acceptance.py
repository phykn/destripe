import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


EXPECTED_SAMPLE_STEMS = tuple(f"sample_{index:02d}" for index in range(2, 6))
EXPECTED_STRENGTHS = (0.01, 0.03, 0.06)
SUPPORTED_MODES = tuple(range(5))
CANONICAL_BASELINES = {
    0.03: ("medium", 0.919),
    0.06: ("strong", 3.783),
}

GroupKey = tuple[object, str, str, float | None, int | None]


def evaluate_acceptance(rows: list[dict[str, object]]) -> list[str]:
    """Return deterministic benchmark gate failures; an empty list means pass."""
    failures: list[str] = []
    groups: dict[GroupKey, list[dict[str, object]]] = defaultdict(list)
    seeds: set[object] = set()

    for row in rows:
        seed = _seed(row)
        seeds.add(seed)
        key = (
            seed,
            str(row.get("case_type", "")),
            str(row.get("pattern", "")),
            _optional_float(row.get("strength")),
            _optional_int(row.get("mode")),
        )
        groups[key].append(row)
        failures.extend(_nonfinite_failures(row, key))

    if not seeds:
        return [
            "missing sample data: no benchmark rows",
            "missing strength data: no benchmark rows",
            "missing mode data: no benchmark rows",
        ]

    for seed in sorted(seeds, key=str):
        seed_rows = [row for row in rows if _seed(row) == seed]
        canonical = [row for row in seed_rows if _is_canonical_curtain(row)]
        clean = [row for row in seed_rows if row.get("case_type") == "clean"]
        robustness = [row for row in seed_rows if _is_robustness(row)]

        failures.extend(_completeness_failures(seed, canonical, clean, groups))
        failures.extend(_clean_failures(seed, clean))
        failures.extend(_canonical_failures(seed, canonical))
        failures.extend(_robustness_failures(seed, robustness, groups))

    return sorted(failures)


def _completeness_failures(
    seed: object,
    canonical: list[dict[str, object]],
    clean: list[dict[str, object]],
    groups: dict[GroupKey, list[dict[str, object]]],
) -> list[str]:
    failures: list[str] = []
    clean_samples = {_sample_stem(row) for row in clean}
    for sample in EXPECTED_SAMPLE_STEMS:
        if sample not in clean_samples:
            failures.append(f"seed {seed}: missing sample {sample} from clean rows")

    present_strengths = {_optional_float(row.get("strength")) for row in canonical}
    for strength in EXPECTED_STRENGTHS:
        if strength not in present_strengths:
            failures.append(
                f"seed {seed}: missing strength {strength:.2f} from canonical rows"
            )

    present_modes = {_optional_int(row.get("mode")) for row in canonical}
    for mode in SUPPORTED_MODES:
        if mode not in present_modes:
            failures.append(f"seed {seed}: missing mode {mode} from canonical rows")

    for mode in SUPPORTED_MODES:
        pattern = f"curtain_m{mode}"
        for strength in EXPECTED_STRENGTHS:
            key = (seed, "synthetic", pattern, strength, mode)
            present_samples = {_sample_stem(row) for row in groups.get(key, [])}
            for sample in EXPECTED_SAMPLE_STEMS:
                if sample not in present_samples:
                    failures.append(
                        f"seed {seed}: missing sample {sample} from "
                        f"{pattern} strength {strength:.2f}"
                    )
    return failures


def _clean_failures(
    seed: object,
    clean: list[dict[str, object]],
) -> list[str]:
    failures: list[str] = []
    for row in clean:
        sample = row.get("sample", "unknown")
        psnr = _finite_number(row.get("output_psnr"))
        ssim = _finite_number(row.get("output_ssim"))
        if psnr is not None and psnr < 40.0:
            failures.append(
                f"seed {seed} sample {sample}: clean PSNR {psnr:.3f} dB < 40.000 dB"
            )
        if ssim is not None and ssim < 0.99:
            failures.append(
                f"seed {seed} sample {sample}: clean SSIM {ssim:.6f} < 0.990000"
            )
    return failures


def _canonical_failures(
    seed: object,
    canonical: list[dict[str, object]],
) -> list[str]:
    failures: list[str] = []
    weak = [row for row in canonical if _strength_is(row, 0.01)]
    weak_psnr = _gains(weak, "psnr")
    weak_ssim = _gains(weak, "ssim")

    if weak_psnr:
        psnr_mean = mean(weak_psnr)
        if psnr_mean < 0.10:
            failures.append(
                f"seed {seed}: weak mean PSNR gain {psnr_mean:.3f} dB < 0.100 dB"
            )
        if min(weak_psnr) < -1.0:
            failures.append(
                f"seed {seed}: weak PSNR loss {min(weak_psnr):.3f} dB is worse than -1.000 dB"
            )
    if weak_ssim:
        ssim_mean = mean(weak_ssim)
        if ssim_mean < 0.001:
            failures.append(
                f"seed {seed}: weak mean SSIM gain {ssim_mean:.6f} < 0.001000"
            )

    paired_weak_gains = [
        pair for row in weak if (pair := _gain_pair(row)) is not None
    ]
    if paired_weak_gains:
        passing = sum(
            psnr_gain >= 0.05 and ssim_gain >= 0.0001
            for psnr_gain, ssim_gain in paired_weak_gains
        )
        coverage = passing / len(paired_weak_gains)
        if coverage < 0.75:
            failures.append(
                f"seed {seed}: weak coverage {coverage:.1%} < 75.0% "
                "for PSNR/SSIM gains"
            )

    for mode in SUPPORTED_MODES:
        direction_rows = [row for row in weak if _optional_int(row.get("mode")) == mode]
        direction_gains = _gains(direction_rows, "psnr")
        if direction_gains and mean(direction_gains) < 0.0:
            failures.append(
                f"seed {seed}: mode {mode} weak mean PSNR gain "
                f"{mean(direction_gains):.3f} dB is negative"
            )

    projections = [
        value
        for row in weak
        if (value := _finite_number(row.get("stripe_projection_left_pct")))
        is not None
    ]
    if projections:
        projection_mean = mean(projections)
        if projection_mean > 70.0:
            failures.append(
                f"seed {seed}: weak additive mean projection left "
                f"{projection_mean:.1f}% > 70.0%"
            )
        projection_coverage = sum(value <= 85.0 for value in projections) / len(
            projections
        )
        if projection_coverage < 0.75:
            failures.append(
                f"seed {seed}: weak additive projection coverage "
                f"{projection_coverage:.1%} < 75.0% at 85.0% left"
            )

    for strength, (label, baseline) in CANONICAL_BASELINES.items():
        strength_rows = [row for row in canonical if _strength_is(row, strength)]
        strength_gains = _gains(strength_rows, "psnr")
        if strength_gains:
            gain_mean = mean(strength_gains)
            minimum = baseline - 0.25
            if gain_mean < minimum:
                failures.append(
                    f"seed {seed}: {label} mean PSNR gain {gain_mean:.3f} dB "
                    f"< {minimum:.3f} dB (recorded {baseline:.3f} dB - 0.250 dB)"
                )
    return failures


def _robustness_failures(
    seed: object,
    robustness: list[dict[str, object]],
    groups: dict[GroupKey, list[dict[str, object]]],
) -> list[str]:
    failures: list[str] = []
    robustness_keys = {
        (
            seed,
            "synthetic",
            str(row.get("pattern", "")),
            _optional_float(row.get("strength")),
            _optional_int(row.get("mode")),
        )
        for row in robustness
    }
    for key in sorted(robustness_keys, key=str):
        group_gains = _gains(groups[key], "psnr")
        if not group_gains:
            continue
        label = f"{key[2]} strength {key[3]} mode {key[4]}"
        gain_mean = mean(group_gains)
        if gain_mean < 0.0:
            failures.append(
                f"seed {seed}: robustness mean PSNR gain {gain_mean:.3f} dB "
                f"is negative for {label}"
            )
        if min(group_gains) < -1.0:
            failures.append(
                f"seed {seed}: robustness PSNR loss {min(group_gains):.3f} dB "
                f"is worse than -1.000 dB for {label}"
            )
    return failures


def _nonfinite_failures(row: dict[str, object], key: GroupKey) -> list[str]:
    case_type = str(row.get("case_type", ""))
    if case_type == "synthetic":
        metrics = (
            "input_psnr",
            "output_psnr",
            "input_ssim",
            "output_ssim",
            "stripe_projection_left_pct",
        )
    elif case_type == "clean":
        metrics = (
            "output_psnr",
            "input_ssim",
            "output_ssim",
            "stripe_projection_left_pct",
        )
    else:
        return []

    failures = []
    for metric in metrics:
        if _finite_number(row.get(metric)) is None:
            failures.append(
                f"seed {key[0]} sample {row.get('sample', 'unknown')}: "
                f"non-finite {metric} for {key[2]} strength {key[3]} mode {key[4]}"
            )
    return failures


def _is_canonical_curtain(row: dict[str, object]) -> bool:
    mode = _optional_int(row.get("mode"))
    return (
        row.get("case_type") == "synthetic"
        and mode in SUPPORTED_MODES
        and row.get("pattern") == f"curtain_m{mode}"
        and row.get("carrier") == "additive"
        and _optional_int(row.get("profile_scale")) == 9
        and _is_close(_optional_float(row.get("angle_offset")), 0.0)
    )


def _is_robustness(row: dict[str, object]) -> bool:
    if row.get("case_type") != "synthetic":
        return False
    carrier = row.get("carrier")
    profile_scale = _optional_int(row.get("profile_scale"))
    angle_offset = _optional_float(row.get("angle_offset"))
    return (
        carrier == "multiplicative"
        or profile_scale in {3, 15}
        or (angle_offset is not None and not _is_close(angle_offset, 0.0))
    )


def _gains(rows: list[dict[str, object]], metric: str) -> list[float]:
    gains = []
    for row in rows:
        first = _finite_number(row.get(f"input_{metric}"))
        second = _finite_number(row.get(f"output_{metric}"))
        if first is not None and second is not None:
            gains.append(second - first)
    return gains


def _gain_pair(row: dict[str, object]) -> tuple[float, float] | None:
    psnr = _gains([row], "psnr")
    ssim = _gains([row], "ssim")
    if not psnr or not ssim:
        return None
    return psnr[0], ssim[0]


def _strength_is(row: dict[str, object], expected: float) -> bool:
    return _is_close(_optional_float(row.get("strength")), expected)


def _sample_stem(row: dict[str, object]) -> str:
    return Path(str(row.get("sample", ""))).stem


def _seed(row: dict[str, object]) -> object:
    value = row.get("seed")
    integer = _optional_int(value)
    return integer if integer is not None else str(value)


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_close(value: float | None, expected: float) -> bool:
    return value is not None and math.isclose(value, expected, abs_tol=1e-12)
