
# Sphere AE Latent Vector Optimization

This directory contains the before/after implementation and benchmark results for optimizing an Autoencoder-based latent vector analysis pipeline used in the Sphere project.

## Overview

The target code compares original ImageNetS919 single-object images with multiple visual variants by extracting Autoencoder latent vectors and computing distance/angle-based metrics.



## Project Structure

```text
.
├─ README.md
│  └─ Project overview, optimization summary, benchmark explanation
├─ requirements.txt
│  └─ Python package dependencies required to run the scripts
├─ src/
│  ├─ before/
│  │  └─ ae_s919_original_variant_metrics.py
│  │     └─ Original AE latent metric pipeline before optimization
│  └─ after/
│     └─ ae_s919_optimized_variant_metrics.py
│        └─ Optimized pipeline with caching, generator iteration, config object, and run statistics
├─ benchmark/
│  ├─ run_benchmark.py
│  │  └─ Script for running before/after benchmark experiments
│  ├─ collect_results.py
│  │  └─ Script for collecting execution time, memory usage, and encoding statistics into CSV
│  └─ plot_results.py
│     └─ Script for generating benchmark comparison figures from the CSV results
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
├─ data/
│  └─ README.md
│     └─ Description of required metadata columns and expected image directory structure
└─ report/
   └─ report.pdf
      └─ Final written report for the assignment
