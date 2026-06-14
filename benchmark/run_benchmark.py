# -*- coding: utf-8 -*-
"""
Benchmark helper for before/after AE latent metric pipelines.

Usage example:
python benchmark_s919_pipeline.py \
  --before-script src/before/ae_s919_original_variant_metrics.py \
  --after-script src/after/ae_s919_optimized_variant_metrics.py \
  --out-root results/benchmark \
  --sizes 10 30 50 \
  --repeat 3

Important:
- The original script has hard-coded LIMIT_IMAGES/OUT_DIR unless you modify it.
- For fair comparison, create a benchmark copy of the original script with the same LIMIT_IMAGES and OUT_DIR per run, or add CLI/env support to the original script.
- The optimized script already accepts --limit-images and --out-dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
from typing import Optional


def run_command(cmd: list[str], cwd: Optional[Path] = None) -> tuple[float, int]:
    start = perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    elapsed = perf_counter() - start
    return elapsed, proc.returncode


def read_after_stats(out_dir: Path) -> dict:
    stats_path = out_dir / "run_stats.json"
    if not stats_path.exists():
        return {}
    try:
        return json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def summarize_runs(times: list[float]) -> tuple[float, float]:
    if not times:
        return 0.0, 0.0
    if len(times) == 1:
        return times[0], 0.0
    return mean(times), stdev(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-script", type=Path, required=True)
    parser.add_argument("--after-script", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 30, 50])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-before", action="store_true")
    parser.add_argument("--skip-after", action="store_true")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    result_csv = args.out_root / "benchmark_results.csv"

    fieldnames = [
        "version",
        "limit_images",
        "repeat",
        "mean_time_sec",
        "std_time_sec",
        "total_pairs",
        "valid_pairs",
        "missing_pairs",
        "failed_pairs",
        "skipped_pairs",
        "metric_rows_written",
        "original_memory_hits",
        "original_disk_hits",
        "original_encoded",
        "variant_disk_hits",
        "variant_encoded",
        "notes",
    ]

    rows = []

    for size in args.sizes:
        if not args.skip_before:
            before_times = []
            for r in range(args.repeat):
                # The original script must be configured externally for this size and output path.
                print(f"[BENCH] before size={size} repeat={r+1}/{args.repeat}", flush=True)
                elapsed, code = run_command([args.python, str(args.before_script)])
                if code != 0:
                    print(f"[WARN] before failed with code {code}", flush=True)
                before_times.append(elapsed)
            m, s = summarize_runs(before_times)
            rows.append({
                "version": "before",
                "limit_images": size,
                "repeat": args.repeat,
                "mean_time_sec": m,
                "std_time_sec": s,
                "total_pairs": "",
                "valid_pairs": "",
                "missing_pairs": "",
                "failed_pairs": "",
                "skipped_pairs": "",
                "metric_rows_written": "",
                "original_memory_hits": "",
                "original_disk_hits": "",
                "original_encoded": "",
                "variant_disk_hits": "",
                "variant_encoded": "",
                "notes": "Original script needs same LIMIT_IMAGES/OUT_DIR configured manually.",
            })

        if not args.skip_after:
            after_times = []
            after_stats = {}
            for r in range(args.repeat):
                out_dir = args.out_root / f"after_size_{size}_repeat_{r+1}"
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                print(f"[BENCH] after size={size} repeat={r+1}/{args.repeat}", flush=True)
                cmd = [
                    args.python,
                    str(args.after_script),
                    "--limit-images",
                    str(size),
                    "--out-dir",
                    str(out_dir),
                    "--no-resume",
                ]
                elapsed, code = run_command(cmd)
                if code != 0:
                    print(f"[WARN] after failed with code {code}", flush=True)
                after_times.append(elapsed)
                after_stats = read_after_stats(out_dir)
            m, s = summarize_runs(after_times)
            rows.append({
                "version": "after",
                "limit_images": size,
                "repeat": args.repeat,
                "mean_time_sec": m,
                "std_time_sec": s,
                "total_pairs": after_stats.get("total_pairs", ""),
                "valid_pairs": after_stats.get("valid_pairs", ""),
                "missing_pairs": after_stats.get("missing_pairs", ""),
                "failed_pairs": after_stats.get("failed_pairs", ""),
                "skipped_pairs": after_stats.get("skipped_pairs", ""),
                "metric_rows_written": after_stats.get("metric_rows_written", ""),
                "original_memory_hits": after_stats.get("original_memory_hits", ""),
                "original_disk_hits": after_stats.get("original_disk_hits", ""),
                "original_encoded": after_stats.get("original_encoded", ""),
                "variant_disk_hits": after_stats.get("variant_disk_hits", ""),
                "variant_encoded": after_stats.get("variant_encoded", ""),
                "notes": "Cold output dir per repeat; memory cache works within one run.",
            })

    with open(result_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("[SAVED]", result_csv)


if __name__ == "__main__":
    main()
