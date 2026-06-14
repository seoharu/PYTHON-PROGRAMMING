from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/data1/seoy/Sphere/AE/ImageNetS919_AE_latent_sphere_white_mask/optimization_project")
csv_path = ROOT / "results" / "benchmark" / "benchmark_results.csv"
df = pd.read_csv(csv_path)

versions = ["before", "after_cold", "after_warm"]

# execution time
plt.figure()
for version in versions:
    sub = df[df["version"] == version].sort_values("limit_images")
    plt.plot(sub["limit_images"], sub["elapsed_sec"], marker="o", label=version)

plt.xlabel("Number of input images")
plt.ylabel("Execution time (sec)")
plt.title("Execution time: before vs after cold/warm")
plt.legend()
plt.tight_layout()
out = ROOT / "results" / "benchmark" / "execution_time_cold_warm.png"
plt.savefig(out, dpi=200)
print("[SAVED]", out)

# memory
plt.figure()
for version in versions:
    sub = df[df["version"] == version].sort_values("limit_images")
    plt.plot(sub["limit_images"], sub["max_rss_mb"], marker="o", label=version)

plt.xlabel("Number of input images")
plt.ylabel("Peak memory (MB)")
plt.title("Peak memory: before vs after cold/warm")
plt.legend()
plt.tight_layout()
out = ROOT / "results" / "benchmark" / "memory_cold_warm.png"
plt.savefig(out, dpi=200)
print("[SAVED]", out)

# original encoding
plt.figure()
for version in versions:
    sub = df[df["version"] == version].sort_values("limit_images")
    plt.plot(sub["limit_images"], sub["original_encoded"], marker="o", label=version)

plt.xlabel("Number of input images")
plt.ylabel("Original encoding count")
plt.title("Original encoding count: before vs after cold/warm")
plt.legend()
plt.tight_layout()
out = ROOT / "results" / "benchmark" / "original_encoding_cold_warm.png"
plt.savefig(out, dpi=200)
print("[SAVED]", out)

# variant encoding
plt.figure()
for version in versions:
    sub = df[df["version"] == version].sort_values("limit_images")
    plt.plot(sub["limit_images"], sub["variant_encoded"], marker="o", label=version)

plt.xlabel("Number of input images")
plt.ylabel("Variant encoding count")
plt.title("Variant encoding count: before vs after cold/warm")
plt.legend()
plt.tight_layout()
out = ROOT / "results" / "benchmark" / "variant_encoding_cold_warm.png"
plt.savefig(out, dpi=200)
print("[SAVED]", out)
