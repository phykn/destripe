import argparse
import statistics
import time
from pathlib import Path

from destripe import destripe
from benchmarks.synthetic import load_samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, default=Path("asset"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--process-size", type=int, default=256)
    args = parser.parse_args()
    images = [sample.image for sample in load_samples(args.asset_dir) if sample.has_ground_truth]
    for image in images:
        destripe(
            image,
            adaptive=2,
            iterations=args.iterations,
            process_size=args.process_size,
            device="cpu",
        )
    durations = []
    for _ in range(args.repeats):
        for image in images:
            started = time.perf_counter()
            destripe(
                image,
                adaptive=2,
                iterations=args.iterations,
                process_size=args.process_size,
                device="cpu",
            )
            durations.append(time.perf_counter() - started)
    print(f"median_seconds={statistics.median(durations):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
