import math
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from statistics import mean


EXPECTED_SAMPLES = tuple(f"sample_{index:02d}.png" for index in range(2, 6))
EXPECTED_STRENGTHS = (0.01, 0.03, 0.06)
SUPPORTED_MODES = tuple(range(5))
CANONICAL_BASELINES = {
    0.03: ("medium", 0.919),
    0.06: ("strong", 3.783),
}


@dataclass(frozen=True)
class ExpectedPattern:
    mode: int
    carrier: str = "additive"
    profile_scale: int = 9
    angle_offset: float = 0.0


EXPECTED_PATTERNS = {
    **{
        f"curtain_m{mode}": ExpectedPattern(mode=mode)
        for mode in SUPPORTED_MODES
    },
    "sparse_m0": ExpectedPattern(mode=0),
    "nonstationary_m0": ExpectedPattern(mode=0),
    "curtain_narrow_m0": ExpectedPattern(mode=0, profile_scale=3),
    "curtain_broad_m0": ExpectedPattern(mode=0, profile_scale=15),
    **{
        f"curtain_multiplicative_m{mode}": ExpectedPattern(
            mode=mode,
            carrier="multiplicative",
        )
        for mode in SUPPORTED_MODES
    },
    **{
        f"curtain_offgrid_m{mode}": ExpectedPattern(
            mode=mode,
            angle_offset=7.5,
        )
        for mode in SUPPORTED_MODES
    },
}
CANONICAL_CURTAIN_PATTERNS = frozenset(
    f"curtain_m{mode}" for mode in SUPPORTED_MODES
)
ROBUSTNESS_PATTERNS = frozenset(
    {
        "curtain_narrow_m0",
        "curtain_broad_m0",
        *(f"curtain_multiplicative_m{mode}" for mode in SUPPORTED_MODES),
        *(f"curtain_offgrid_m{mode}" for mode in SUPPORTED_MODES),
    }
)

RowIdentity = tuple[
    object,
    str,
    str,
    float | None,
    int | None,
    str,
    int,
]


def evaluate_acceptance(rows: list[dict[str, object]]) -> list[str]:
    """Return deterministic benchmark gate failures; an empty list means pass."""
    failures: list[str] = []
    evaluated_rows = [row for row in rows if row.get("case_type") != "real"]
    seeds = {_seed(row) for row in evaluated_rows}

    if not seeds:
        return [
            "missing sample data: no benchmark rows",
            "missing strength data: no benchmark rows",
            "missing mode data: no benchmark rows",
        ]

    for seed in sorted(seeds, key=str):
        seed_rows = [row for row in evaluated_rows if _seed(row) == seed]
        validation_failures, unique_rows = _validate_seed_rows(seed, seed_rows)
        failures.extend(validation_failures)
        for identity, row in unique_rows:
            failures.extend(_nonfinite_failures(row, identity))

        clean = [row for _, row in unique_rows if row.get("case_type") == "clean"]
        canonical = [
            row
            for _, row in unique_rows
            if str(row.get("pattern")) in CANONICAL_CURTAIN_PATTERNS
        ]
        robustness = [
            row
            for _, row in unique_rows
            if str(row.get("pattern")) in ROBUSTNESS_PATTERNS
        ]
        failures.extend(_clean_failures(seed, clean))
        failures.extend(_canonical_failures(seed, canonical))
        failures.extend(_robustness_failures(seed, robustness))

    return sorted(failures)


def _validate_seed_rows(
    seed: object,
    rows: list[dict[str, object]],
) -> tuple[list[str], list[tuple[RowIdentity, dict[str, object]]]]:
    failures: list[str] = []
    levels = {
        level
        for row in rows
        if row.get("case_type") in {"clean", "synthetic"}
        and (level := _optional_int(row.get("level"))) is not None
    }
    if not levels:
        failures.append(f"seed {seed}: missing level data")

    candidates: dict[RowIdentity, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        case_type = str(row.get("case_type", ""))
        if case_type not in {"clean", "synthetic"}:
            failures.append(f"seed {seed}: unexpected case_type {case_type!r}")
            continue
        level = _optional_int(row.get("level"))
        if level is None:
            failures.append(
                f"seed {seed} sample {row.get('sample', 'unknown')}: "
                "missing or non-numeric level"
            )
            continue
        sample = str(row.get("sample", ""))
        if sample not in EXPECTED_SAMPLES:
            failures.append(f"seed {seed}: unexpected sample {sample!r}")
            continue

        if case_type == "clean":
            if not _matches_clean_metadata(row):
                failures.append(
                    f"seed {seed} sample {sample} level {level}: "
                    "metadata mismatch for clean row"
                )
                continue
            identity = (seed, "clean", "none", 0.0, None, sample, level)
        else:
            pattern = str(row.get("pattern", ""))
            expected = EXPECTED_PATTERNS.get(pattern)
            if expected is None:
                failures.append(f"seed {seed}: unexpected pattern {pattern}")
                continue
            if not _matches_pattern_metadata(row, expected):
                failures.append(
                    f"seed {seed} sample {sample} level {level}: metadata mismatch "
                    f"for {pattern}; expected mode={expected.mode}, "
                    f"carrier={expected.carrier}, "
                    f"profile_scale={expected.profile_scale}, "
                    f"angle_offset={expected.angle_offset}"
                )
                continue
            strength = _normalized_strength(row.get("strength"))
            if strength is None:
                failures.append(
                    f"seed {seed} sample {sample} level {level}: unexpected strength "
                    f"{row.get('strength')!r} for {pattern}"
                )
                continue
            identity = (
                seed,
                "synthetic",
                pattern,
                strength,
                expected.mode,
                sample,
                level,
            )
        candidates[identity].append(row)

    present_strengths = {
        identity[3]
        for identity in candidates
        if identity[1] == "synthetic"
    }
    for strength in EXPECTED_STRENGTHS:
        if strength not in present_strengths:
            failures.append(
                f"seed {seed}: missing strength {strength:.2f} from synthetic rows"
            )
    present_modes = {
        identity[4]
        for identity in candidates
        if identity[1] == "synthetic"
    }
    for mode in SUPPORTED_MODES:
        if mode not in present_modes:
            failures.append(f"seed {seed}: missing mode {mode} from synthetic rows")

    unique_rows: list[tuple[RowIdentity, dict[str, object]]] = []
    for identity in _expected_identities(seed, levels):
        matches = candidates.get(identity, [])
        if not matches:
            failures.append(_missing_identity_failure(identity))
        elif len(matches) > 1:
            failures.append(
                f"duplicate row identity ({len(matches)} rows): "
                f"{_identity_label(identity)}"
            )
        else:
            unique_rows.append((identity, matches[0]))
    return failures, unique_rows


def _expected_identities(seed: object, levels: set[int]) -> list[RowIdentity]:
    identities: list[RowIdentity] = []
    for level in sorted(levels):
        for sample in EXPECTED_SAMPLES:
            identities.append((seed, "clean", "none", 0.0, None, sample, level))
        for pattern, metadata in EXPECTED_PATTERNS.items():
            for strength in EXPECTED_STRENGTHS:
                for sample in EXPECTED_SAMPLES:
                    identities.append(
                        (
                            seed,
                            "synthetic",
                            pattern,
                            strength,
                            metadata.mode,
                            sample,
                            level,
                        )
                    )
    return identities


def _missing_identity_failure(identity: RowIdentity) -> str:
    return (
        f"seed {identity[0]}: missing row (missing sample {identity[5]} coverage) "
        f"for pattern {identity[2]}, strength {identity[3]}, mode {identity[4]}, "
        f"level {identity[6]}"
    )


def _identity_label(identity: RowIdentity) -> str:
    return (
        f"seed={identity[0]}, case_type={identity[1]}, pattern={identity[2]}, "
        f"strength={identity[3]}, mode={identity[4]}, sample={identity[5]}, "
        f"level={identity[6]}"
    )


def _matches_clean_metadata(row: dict[str, object]) -> bool:
    return (
        row.get("pattern") == "none"
        and _is_absent(row.get("mode"))
        and _is_close(_optional_float(row.get("strength")), 0.0)
        and row.get("carrier") == "additive"
        and _optional_int(row.get("profile_scale")) == 9
        and _is_close(_optional_float(row.get("angle_offset")), 0.0)
    )


def _matches_pattern_metadata(
    row: dict[str, object],
    expected: ExpectedPattern,
) -> bool:
    return (
        _optional_int(row.get("mode")) == expected.mode
        and row.get("carrier") == expected.carrier
        and _optional_int(row.get("profile_scale")) == expected.profile_scale
        and _is_close(
            _optional_float(row.get("angle_offset")),
            expected.angle_offset,
        )
    )


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
                f"seed {seed}: weak PSNR loss {min(weak_psnr):.3f} dB "
                "is worse than -1.000 dB"
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
        direction_rows = [
            row for row in weak if _optional_int(row.get("mode")) == mode
        ]
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
) -> list[str]:
    failures: list[str] = []
    gains = _gains(robustness, "psnr")
    if not gains:
        return failures
    gain_mean = mean(gains)
    if gain_mean < 0.0:
        failures.append(
            f"seed {seed}: robustness mean PSNR gain {gain_mean:.3f} dB is negative"
        )
    if min(gains) < -1.0:
        failures.append(
            f"seed {seed}: robustness PSNR loss {min(gains):.3f} dB "
            "is worse than -1.000 dB"
        )
    return failures


def _nonfinite_failures(
    row: dict[str, object],
    identity: RowIdentity,
) -> list[str]:
    case_type = str(row.get("case_type", ""))
    failures = []
    if case_type == "synthetic":
        metrics = (
            "input_psnr",
            "output_psnr",
            "input_ssim",
            "output_ssim",
            "stripe_projection_left_pct",
        )
    elif case_type == "clean":
        if not _valid_clean_input_psnr(row.get("input_psnr")):
            failures.append(
                f"seed {identity[0]} sample {identity[5]} level {identity[6]}: "
                "clean input_psnr must be finite or positive infinity"
            )
        metrics = (
            "output_psnr",
            "input_ssim",
            "output_ssim",
            "stripe_projection_left_pct",
        )
    else:
        return []

    for metric in metrics:
        if _finite_number(row.get(metric)) is None:
            failures.append(
                f"seed {identity[0]} sample {identity[5]} level {identity[6]}: "
                f"non-finite {metric} for {identity[2]} strength {identity[3]} "
                f"mode {identity[4]}"
            )
    return failures


def _valid_clean_input_psnr(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) or number == math.inf


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


def _normalized_strength(value: object) -> float | None:
    number = _optional_float(value)
    for expected in EXPECTED_STRENGTHS:
        if _is_close(number, expected):
            return expected
    return None


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
    if _is_absent(value) or isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        if not math.isfinite(value):
            return None
        integer = int(value)
        return integer if value == integer else None
    if isinstance(value, (str, Decimal)):
        try:
            number = value if isinstance(value, Decimal) else Decimal(value)
        except InvalidOperation:
            return None
        if not number.is_finite() or number != number.to_integral_value():
            return None
        return int(number)
    return None


def _is_absent(value: object) -> bool:
    return value is None or (isinstance(value, str) and value == "")


def _is_close(value: float | None, expected: float) -> bool:
    return value is not None and math.isclose(value, expected, abs_tol=1e-12)
