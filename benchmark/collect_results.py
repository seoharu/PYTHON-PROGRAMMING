from pathlib import Path
import json
import csv
import re

ROOT = Path("/data1/seoy/Sphere/AE/ImageNetS919_AE_latent_sphere_white_mask/optimization_project")
LIMITS = [10, 30, 50, 100]

def parse_time_log(path: Path):
    if not path.exists():
        return None, None

    text = path.read_text(errors="ignore")

    elapsed = None
    max_rss_kb = None

    m = re.search(r"Elapsed \(wall clock\) time.*: (.+)", text)
    if m:
        elapsed = m.group(1).strip()

    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    if m:
        max_rss_kb = int(m.group(1))

    return elapsed, max_rss_kb

def elapsed_to_seconds(s):
    if s is None:
        return None

    parts = s.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except Exception:
        return None

    return None

def count_csv_rows(path: Path):
    if not path.exists():
        return None
    return max(0, sum(1 for _ in path.open(errors="ignore")) - 1)

def read_stats(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def append_row(rows, n, version, log_name, stats_path=None, before=False):
    log_path = ROOT / "logs" / log_name
    elapsed_str, rss_kb = parse_time_log(log_path)

    if before:
        metrics = ROOT / "results" / f"before_{n}" / "image_level_metrics.csv"
        valid_pairs = count_csv_rows(metrics)
        stats = {
            "valid_pairs": valid_pairs,
            "original_encoded": valid_pairs,
            "variant_encoded": valid_pairs,
            "original_memory_hits": 0,
            "original_disk_hits": 0,
            "variant_disk_hits": 0,
        }
    else:
        stats = read_stats(stats_path)

    rows.append({
        "limit_images": n,
        "version": version,
        "elapsed_raw": elapsed_str,
        "elapsed_sec": elapsed_to_seconds(elapsed_str),
        "max_rss_mb": round(rss_kb / 1024, 2) if rss_kb else None,
        "valid_pairs": stats.get("valid_pairs"),
        "original_encoded": stats.get("original_encoded"),
        "variant_encoded": stats.get("variant_encoded"),
        "original_memory_hits": stats.get("original_memory_hits"),
        "original_disk_hits": stats.get("original_disk_hits"),
        "variant_disk_hits": stats.get("variant_disk_hits"),
    })

rows = []

for n in LIMITS:
    append_row(
        rows,
        n,
        "before",
        f"before_{n}.log",
        before=True,
    )

    append_row(
        rows,
        n,
        "after_cold",
        f"after_{n}.log",
        stats_path=ROOT / "results" / f"after_{n}" / "run_stats_cold.json",
    )

    append_row(
        rows,
        n,
        "after_warm",
        f"after_{n}_warm.log",
        stats_path=ROOT / "results" / f"after_{n}" / "run_stats_warm.json",
    )

out = ROOT / "results" / "benchmark" / "benchmark_results.csv"
out.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "limit_images",
    "version",
    "elapsed_raw",
    "elapsed_sec",
    "max_rss_mb",
    "valid_pairs",
    "original_encoded",
    "variant_encoded",
    "original_memory_hits",
    "original_disk_hits",
    "variant_disk_hits",
]

with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("[SAVED]", out)
