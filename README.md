# Sphere AE Latent Vector Optimization

This repository contains the before/after implementation, benchmark scripts, and benchmark results for optimizing an Autoencoder-based latent vector analysis pipeline used in the Sphere project.

The target pipeline compares original ImageNetS919 single-object images with multiple transformed variant images. Each image is passed through a pretrained ImageNet Autoencoder encoder, and the resulting latent vectors are compared using distance and angle-based metrics such as L2 distance, cosine similarity, cosine distance, angle in radians, norm difference, and relative L2.

---

## 1. Project Structure

```text
.
├─ README.md
│  └─ Project overview, setup instructions, data requirements, and benchmark guide
├─ requirements.txt
│  └─ Python package dependencies
├─ src/
│  ├─ before/
│  │  └─ ae_s919_original_variant_metrics.py
│  │     └─ Original AE latent metric pipeline before optimization
│  └─ after/
│     └─ ae_s919_optimized_variant_metrics.py
│        └─ Optimized pipeline with caching, generator iteration, config object, and run statistics
├─ benchmark/
│  ├─ run_benchmark.py
│  │  └─ Helper script for running before/after benchmark experiments
│  ├─ collect_results.py
│  │  └─ Collects execution time, peak memory, and encoding statistics into a CSV file
│  └─ plot_results.py
│     └─ Generates benchmark comparison figures from benchmark_results.csv
├─ results/
│  └─ benchmark/
│     ├─ benchmark_results.csv
│     │  └─ Final benchmark table comparing before, after cold, and after warm runs
│     ├─ execution_time_cold_warm.png
│     │  └─ Execution time comparison figure
│     ├─ memory_cold_warm.png
│     │  └─ Peak memory comparison figure
│     ├─ original_encoding_cold_warm.png
│     │  └─ Original image encoding count comparison figure
│     └─ variant_encoding_cold_warm.png
│        └─ Variant image encoding count comparison figure
└─ report/
   └─ report.pdf
      └─ Final written report for the assignment
```

---

## 2. Optimization Summary

The original code processes each original-variant pair independently. If one original image has 12 variants, the original image can be encoded 12 times even though its latent vector is identical across those comparisons.

The optimized code improves the pipeline by introducing:

* `dict`-based original latent memory cache
* disk cache reuse for saved `.npy` latent vectors
* generator-based pair iteration instead of full pair-list materialization
* `dataclass`-based experiment configuration
* key-based resume logic using `(class_id, image_id, variant)`
* timing and run statistics logging
* explicit metric computation using reused dot product and norm values

The main goal is not only to reduce one-time execution time, but also to make the research pipeline more reproducible, cache-aware, and easier to benchmark.

---

## 3. Environment Setup

### 3.1 Python Version

Recommended:

```bash
python >= 3.10
```

### 3.2 Create a Virtual Environment

Using conda:

```bash
conda create -n sphere-ae python=3.10
conda activate sphere-ae
```

Or using venv:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3.3 Install Dependencies

Install general dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains the basic Python packages used by the analysis and benchmark scripts.

Recommended contents:

```text
numpy>=1.24
pandas>=2.0
Pillow>=10.0
matplotlib>=3.7
tqdm>=4.66
torch
torchvision
```

For PyTorch, install the build that matches the server CUDA version. Check the server GPU environment with:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## 4. External Autoencoder Requirement

This project uses the pretrained ImageNet Autoencoder implementation from the following repository:

```text
https://github.com/Horizon2333/imagenet-autoencoder
```

In the experiment server, the repository was cloned locally and loaded by adding the local repository path to `sys.path`.

The checkpoint used in this experiment was:

```text
imagenet-vgg16.pth
```

In the original server experiment, the local Autoencoder repository and checkpoint were placed under a server-specific path such as:

```text
/data1/seoy/Sphere/AE/external/imagenet-autoencoder
/data1/seoy/Sphere/AE/external/imagenet-autoencoder/checkpoints/imagenet-vgg16.pth
```

The analysis scripts load the Autoencoder model from the external repository, restore the pretrained checkpoint, and use only the encoder part of the model to extract latent vectors.

Because the Autoencoder repository and checkpoint are external dependencies, they are not included in this repository. To reproduce the experiment, clone the Autoencoder repository separately and place the checkpoint at the path specified in the code, or pass the checkpoint path through command-line arguments when using the optimized script.

Conceptually, the Autoencoder loading process is:

```python
sys.path.insert(0, str(EXTERNAL_REPO_ROOT))

from models.builer import build_model

model = build_model(arch="vgg16")
checkpoint = torch.load(AE_CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

encoder = model.module.encoder
```

The exact implementation is included in:

```text
src/before/ae_s919_original_variant_metrics.py
src/after/ae_s919_optimized_variant_metrics.py
```

---

## 5. Dataset Requirement

This experiment uses ImageNet-S from the following repository:

```text
https://github.com/LUSSeg/ImageNet-S
```

Specifically, the experiment uses:

```text
ImageNetS919 train-full
```

The real ImageNetS919 images and generated variant images are not included in this repository because they are large dataset files. The code expects a metadata CSV file describing the ImageNetS919 images and the generated single-object variant images.

The metadata CSV must include at least the following columns:

| Column          | Description                                                             |
| --------------- | ----------------------------------------------------------------------- |
| `class_id`      | ImageNet class ID, for example `n01443537`                              |
| `image_id`      | Image identifier inside the class                                       |
| `original_path` | Absolute or relative path to the original ImageNetS919 image            |
| `n_fg_labels`   | Number of foreground labels; only rows with `n_fg_labels == 1` are used |

Only single-object images are analyzed:

```python
df = df[df["n_fg_labels"] == 1]
```

Example metadata:

```csv
class_id,image_id,original_path,n_fg_labels
n01443537,9812,/path/to/ImageNetS919/train-full/n01443537/9812.jpg,1
n01443537,16309,/path/to/ImageNetS919/train-full/n01443537/16309.jpg,1
```

### Expected Variant Directory Structure

The variant image root directory is expected to contain the generated image variants used in the Sphere project.

```text
VARIANT_ROOT/
├─ alpha/<class_id>/<image_id>.jpg
├─ bg_color/<class_id>/<image_id>.jpg
├─ bg_inpaint/<class_id>/<image_id>.jpg
├─ bg_transparent/<class_id>/<image_id>.png
├─ comp_black/<class_id>/<image_id>.jpg
├─ comp_white/<class_id>/<image_id>.jpg
├─ cutout_rgba/<class_id>/<image_id>.png
├─ gaussian/<class_id>/<image_id>.jpg
├─ masked_black/<class_id>/<image_id>.jpg
├─ masked_white/<class_id>/<image_id>.jpg
├─ salt_pepper/<class_id>/<image_id>.jpg
└─ mask_white/<class_id>/<image_id>.png
```

The code uses `.png` for:

```python
{"bg_transparent", "cutout_rgba", "mask_white"}
```

and `.jpg` for the other variants.

---

## 6. How to Run the Original Pipeline

The original code is located at:

```text
src/before/ae_s919_original_variant_metrics.py
```

The original script uses global constants for metadata path, variant root, output directory, Autoencoder repository path, checkpoint path, device, and image preprocessing options. Before running it on a new server, check and edit the constants near the top of the file.

Run:

```bash
python src/before/ae_s919_original_variant_metrics.py
```

For benchmark experiments, a temporary copy of the original script can be created and its `LIMIT_IMAGES` and `OUT_DIR` constants can be modified.

Example:

```bash
cp src/before/ae_s919_original_variant_metrics.py \
   src/before/ae_s919_original_variant_metrics_bench_50.py

python src/before/ae_s919_original_variant_metrics_bench_50.py
```

The original script is useful as a baseline because it represents the pair-centered pipeline before caching, generator iteration, and config separation were applied.

---

## 7. How to Run the Optimized Pipeline

The optimized code is located at:

```text
src/after/ae_s919_optimized_variant_metrics.py
```

Example run:

```bash
python src/after/ae_s919_optimized_variant_metrics.py \
  --meta-path /data1/seoy/Sphere/mask_transformation/ImageNetS919/annotation_hstack_colormap_metadata.csv \
  --variant-root /data1/seoy/Sphere/mask_transformation/ImageNetS919_single_object_variants_white_mask \
  --external-repo-root /data1/seoy/Sphere/AE/external/imagenet-autoencoder \
  --checkpoint-path /data1/seoy/Sphere/AE/external/imagenet-autoencoder/checkpoints/imagenet-vgg16.pth \
  --limit-images 50 \
  --out-dir results/after_50 \
  --no-resume
```

Common options:

| Option                 | Description                                                         |
| ---------------------- | ------------------------------------------------------------------- |
| `--meta-path`          | Path to the ImageNetS919 metadata CSV                               |
| `--variant-root`       | Root directory containing generated variant images                  |
| `--external-repo-root` | Local path to the cloned Autoencoder repository                     |
| `--checkpoint-path`    | Path to the pretrained Autoencoder checkpoint                       |
| `--limit-images`       | Limit the number of metadata rows for testing or benchmark          |
| `--out-dir`            | Output directory for metric CSV, summaries, stats, and latent files |
| `--no-resume`          | Disable key-based resume and recompute metric rows                  |
| `--no-disk-cache`      | Disable loading existing latent `.npy` files                        |
| `--variants`           | Run only selected variant types                                     |

The optimized run produces:

```text
image_level_metrics.csv
variant_summary.csv
overall_summary.csv
pairs_all.csv
missing_pairs.csv
failed_pairs.csv
run_stats.json
config.json
latents/
```

`run_stats.json` records statistics such as:

```text
total_pairs
valid_pairs
missing_pairs
failed_pairs
skipped_pairs
metric_rows_written
original_memory_hits
original_disk_hits
original_encoded
variant_disk_hits
variant_encoded
elapsed_sec
```

---

## 8. Cold Run vs Warm Run

The benchmark distinguishes between cold and warm execution.

| Condition    | Description                                                                            |
| ------------ | -------------------------------------------------------------------------------------- |
| `after_cold` | Optimized code is executed with no pre-existing latent cache                           |
| `after_warm` | Optimized code is executed again using saved latent `.npy` files from the previous run |

Cold run:

```bash
rm -rf results/after_100

/usr/bin/time -v python src/after/ae_s919_optimized_variant_metrics.py \
  --limit-images 100 \
  --out-dir results/after_100 \
  --no-resume \
  > logs/after_100.log 2>&1
```

Warm run:

```bash
/usr/bin/time -v python src/after/ae_s919_optimized_variant_metrics.py \
  --limit-images 100 \
  --out-dir results/after_100 \
  --no-resume \
  > logs/after_100_warm.log 2>&1
```

In the warm run, the optimized code can reuse saved original and variant latent vectors from disk cache. Therefore, `original_encoded` and `variant_encoded` can become 0 if all latent files already exist.

---

## 9. Benchmark

Benchmark scripts are located in:

```text
benchmark/
```

Run benchmark experiments:

```bash
python benchmark/run_benchmark.py
```

Collect execution time, peak memory, and encoding statistics:

```bash
python benchmark/collect_results.py
```

Generate plots:

```bash
python benchmark/plot_results.py
```

The benchmark result CSV is stored at:

```text
results/benchmark/benchmark_results.csv
```

Main figures:

```text
results/benchmark/execution_time_cold_warm.png
results/benchmark/memory_cold_warm.png
results/benchmark/original_encoding_cold_warm.png
results/benchmark/variant_encoding_cold_warm.png
```

---

## 10. Benchmark Result Summary

The benchmark compares three conditions:

| Condition    | Meaning                                                    |
| ------------ | ---------------------------------------------------------- |
| `before`     | Original implementation                                    |
| `after_cold` | Optimized implementation without pre-existing latent cache |
| `after_warm` | Optimized implementation with disk cache reuse             |

Observed result summary:

* `after_cold` removed repeated original encoding.
* `after_cold` was slower than `before` in wall-clock time because of additional bookkeeping overhead and remaining variant encoding.
* `after_warm` reused saved original and variant latent files from disk cache.
* In `after_warm`, encoder calls were eliminated for already cached latent vectors.
* For repeated analysis, metric recalculation, or summary regeneration, the optimized pipeline is substantially faster.

The final benchmark table is available at:

```text
results/benchmark/benchmark_results.csv
```

---

## 11. Synthetic Data / Minimal Input Example

The real experiment requires a pretrained ImageNet Autoencoder checkpoint, so the full AE latent extraction cannot be reproduced without the external Autoencoder repository and checkpoint.

However, the expected metadata and directory structure can be tested with synthetic images.

Example synthetic metadata:

```csv
class_id,image_id,original_path,n_fg_labels
n00000001,0001,/path/to/synthetic/original/n00000001/0001.jpg,1
n00000001,0002,/path/to/synthetic/original/n00000001/0002.jpg,1
```

Example synthetic directory layout:

```text
synthetic_data/
├─ metadata.csv
├─ original/
│  └─ n00000001/
│     ├─ 0001.jpg
│     └─ 0002.jpg
└─ variants/
   ├─ alpha/n00000001/0001.jpg
   ├─ alpha/n00000001/0002.jpg
   ├─ bg_transparent/n00000001/0001.png
   └─ ...
```

If a synthetic data generator is added, it can be placed under:

```text
benchmark/create_synthetic_data.py
```

Example command:

```bash
python benchmark/create_synthetic_data.py \
  --out-dir synthetic_data \
  --num-images 5
```

The synthetic data is intended to validate metadata and directory layout. Full AE execution still requires the external Autoencoder repository and checkpoint.

---

## 12. Files Not Included

The following files are intentionally excluded:

```text
*.npy
latents/
*.pth
*.pt
*.ckpt
logs/
results/before_*/
results/after_*/
real ImageNetS919 images
real variant images
Autoencoder checkpoint files
```

These files are large, server-specific, or reproducible from the provided scripts.

---

## 13. Report

The final report is stored in:

```text
report/report.pdf
```

If the report PDF is not included in the repository, it should be submitted separately according to the course submission instructions.

