# -*- coding: utf-8 -*-
"""
Optimized AE latent metric pipeline for Sphere / ImageNetS919 variants.

Main changes from the original script:
1. dataclass-based config instead of scattered global constants
2. generator-based pair row production
3. original latent memory cache and disk cache
4. disk cache for variant latents
5. key-based resume using (class_id, image_id, variant)
6. separated pipeline functions following single-responsibility principle
7. timing decorator and runtime statistics for benchmarking
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Callable, Generator, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms


# ---------------------------------------------------------------------
# Dataset and external dependency information
# ---------------------------------------------------------------------
# Dataset source:
#   https://github.com/LUSSeg/ImageNet-S
#
# Dataset split used in this experiment:
#   ImageNetS919 train-full
#
# External Autoencoder source:
#   https://github.com/Horizon2333/imagenet-autoencoder
#
# Autoencoder checkpoint:
#   imagenet-vgg16.pth
# ---------------------------------------------------------------------

AE_REPO_URL = "https://github.com/Horizon2333/imagenet-autoencoder"


# ---------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------
# These are relative paths. The real dataset, generated variants,
# Autoencoder repository, and checkpoint are not included in this repo.
#
# Expected structure:
#   ImageNetS919/
#   ImageNetS919_single_object_variants_white_mask/
#   imagenet-autoencoder/
#   imagenet-autoencoder/checkpoints/imagenet-vgg16.pth
# ---------------------------------------------------------------------

DEFAULT_META_PATH = Path(
    "ImageNetS919/annotation_hstack_colormap_metadata.csv"
)

DEFAULT_VARIANT_ROOT = Path(
    "ImageNetS919_single_object_variants_white_mask"
)

DEFAULT_OUT_DIR = Path(
    "results/after_run"
)

DEFAULT_EXTERNAL_REPO_ROOT = Path(
    "imagenet-autoencoder"
)

DEFAULT_AE_CHECKPOINT_PATH = Path(
    "imagenet-autoencoder/checkpoints/imagenet-vgg16.pth"
)


DEFAULT_VARIANTS = (
    "alpha",
    "bg_color",
    "bg_inpaint",
    "bg_transparent",
    "comp_black",
    "comp_white",
    "cutout_rgba",
    "gaussian",
    "masked_black",
    "masked_white",
    "salt_pepper",
    "mask_white",
)

PNG_VARIANTS = {"bg_transparent", "cutout_rgba", "mask_white"}


@dataclass(frozen=True)
class LatentAnalysisConfig:
    meta_path: Path = DEFAULT_META_PATH
    variant_root: Path = DEFAULT_VARIANT_ROOT
    out_dir: Path = DEFAULT_OUT_DIR
    external_repo_root: Path = DEFAULT_EXTERNAL_REPO_ROOT
    checkpoint_path: Path = DEFAULT_AE_CHECKPOINT_PATH
    arch: str = "vgg16"
    device: str = "cuda"
    resize_size: int = 256
    crop_size: int = 224
    seed: int = 42
    variants: Tuple[str, ...] = DEFAULT_VARIANTS
    save_latents: bool = True
    use_disk_cache: bool = True
    use_memory_cache: bool = True
    resume: bool = True
    limit_images: Optional[int] = None
    progress_every: int = 100

    @property
    def latent_dir(self) -> Path:
        return self.out_dir / "latents"

    @property
    def metric_csv(self) -> Path:
        return self.out_dir / "image_level_metrics.csv"

    @property
    def pairs_all_csv(self) -> Path:
        return self.out_dir / "pairs_all.csv"

    @property
    def missing_pairs_csv(self) -> Path:
        return self.out_dir / "missing_pairs.csv"

    @property
    def failed_pairs_csv(self) -> Path:
        return self.out_dir / "failed_pairs.csv"

    @property
    def run_stats_json(self) -> Path:
        return self.out_dir / "run_stats.json"

    @property
    def config_json(self) -> Path:
        return self.out_dir / "config.json"


def timed(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = perf_counter()
            result = fn(*args, **kwargs)
            elapsed = perf_counter() - start
            print(f"[TIME] {name}: {elapsed:.4f}s", flush=True)
            return result

        return wrapper

    return decorator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> str:
    if device_name == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_eval_transform(resize_size: int, crop_size: int):
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ])


def load_pil_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


@timed("load_encoder")
def load_encoder_from_imagenet_ae(config: LatentAnalysisConfig, device: str):
    repo_root = os.path.abspath(config.external_repo_root)

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import utils  # type: ignore
    import models.builer as builder  # type: ignore

    args = SimpleNamespace()
    args.arch = config.arch
    args.parallel = 0
    args.batch_size = 1
    args.workers = 0

    model = builder.BuildAutoEncoder(args)
    utils.load_dict(config.checkpoint_path, model)

    model = model.to(device)
    model.eval()

    def encoder_fn(x: torch.Tensor) -> torch.Tensor:
        return model.module.encoder(x)

    return encoder_fn


@torch.no_grad()
def encode_image(encoder, transform, image: Image.Image, device: str) -> torch.Tensor:
    x = transform(image).unsqueeze(0).to(device, non_blocking=True)
    z = encoder(x)
    z = z.flatten(start_dim=1)
    return z[0].detach().cpu().float()


def variant_path_for(
    config: LatentAnalysisConfig,
    class_id: str,
    image_id: str,
    variant: str,
) -> Path:
    ext = ".png" if variant in PNG_VARIANTS else ".jpg"
    return config.variant_root / variant / class_id / f"{image_id}{ext}"


def latent_path_for(
    config: LatentAnalysisConfig,
    class_id: str,
    image_id: str,
    condition: str,
) -> Path:
    return config.latent_dir / class_id / f"{image_id}_{condition}_latent.npy"


def save_latent(path: Path, z: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, z.detach().cpu().numpy().astype(np.float32))


def load_latent(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.load(path)).float()


def iter_pair_rows(config: LatentAnalysisConfig) -> Generator[dict[str, Any], None, None]:
    df = pd.read_csv(config.meta_path)
    df = df[df["n_fg_labels"] == 1].copy()
    df = df.sort_values(["class_id", "image_id"]).reset_index(drop=True)

    if config.limit_images is not None:
        df = df.head(config.limit_images).copy()

    for _, r in df.iterrows():
        class_id = str(r["class_id"])
        image_id = str(r["image_id"])
        original_path = Path(str(r["original_path"]))

        for variant in config.variants:
            variant_path = variant_path_for(
                config,
                class_id,
                image_id,
                variant,
            )

            yield {
                "class_id": class_id,
                "image_id": image_id,
                "variant": variant,
                "original_path": str(original_path),
                "variant_path": str(variant_path),
                "original_exists": original_path.exists(),
                "variant_exists": variant_path.exists(),
            }


def pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["class_id"]),
        str(row["image_id"]),
        str(row["variant"]),
    )


def load_done_keys(metric_csv: Path) -> set[tuple[str, str, str]]:
    if not metric_csv.exists():
        return set()

    try:
        df = pd.read_csv(
            metric_csv,
            usecols=["class_id", "image_id", "variant"],
        )
    except Exception:
        return set()

    return {
        (str(r.class_id), str(r.image_id), str(r.variant))
        for r in df.itertuples(index=False)
    }


def compute_metrics(z_orig: torch.Tensor, z_var: torch.Tensor) -> dict[str, float]:
    z_orig = z_orig.float().flatten()
    z_var = z_var.float().flatten()

    diff = z_var - z_orig
    l2 = torch.linalg.norm(diff).item()

    norm_original = torch.linalg.norm(z_orig).item()
    norm_variant = torch.linalg.norm(z_var).item()

    dot = torch.dot(z_orig, z_var).item()
    denom = max(norm_original * norm_variant, 1e-12)

    cos = dot / denom
    cos = float(np.clip(cos, -1.0, 1.0))

    cosine_distance = 1.0 - cos
    angle_radian = float(np.arccos(cos))
    norm_difference = norm_variant - norm_original
    relative_l2 = l2 / max(norm_original, 1e-12)

    return {
        "l2": float(l2),
        "cosine_similarity": float(cos),
        "cosine_distance": float(cosine_distance),
        "angle_radian": float(angle_radian),
        "norm_original": float(norm_original),
        "norm_variant": float(norm_variant),
        "norm_difference": float(norm_difference),
        "relative_l2": float(relative_l2),
    }


class RuntimeStats:
    def __init__(self) -> None:
        self.total_pairs = 0
        self.valid_pairs = 0
        self.missing_pairs = 0
        self.failed_pairs = 0
        self.skipped_pairs = 0
        self.metric_rows_written = 0

        self.original_memory_hits = 0
        self.original_disk_hits = 0
        self.original_encoded = 0

        self.variant_disk_hits = 0
        self.variant_encoded = 0

        self.start_time = perf_counter()
        self.elapsed_sec = 0.0

    def finish(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.elapsed_sec = perf_counter() - self.start_time

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def get_original_latent(
    config: LatentAnalysisConfig,
    encoder,
    transform,
    row: dict[str, Any],
    device: str,
    original_cache: dict[tuple[str, str], torch.Tensor],
    stats: RuntimeStats,
) -> torch.Tensor:
    key = (str(row["class_id"]), str(row["image_id"]))

    if config.use_memory_cache and key in original_cache:
        stats.original_memory_hits += 1
        return original_cache[key]

    path = latent_path_for(
        config,
        row["class_id"],
        row["image_id"],
        "original",
    )

    if config.use_disk_cache and path.exists():
        z = load_latent(path)
        stats.original_disk_hits += 1
    else:
        image = load_pil_rgb(row["original_path"])
        z = encode_image(encoder, transform, image, device)
        stats.original_encoded += 1

        if config.save_latents:
            save_latent(path, z)

    if config.use_memory_cache:
        original_cache[key] = z

    return z


def get_variant_latent(
    config: LatentAnalysisConfig,
    encoder,
    transform,
    row: dict[str, Any],
    device: str,
    stats: RuntimeStats,
) -> torch.Tensor:
    path = latent_path_for(
        config,
        row["class_id"],
        row["image_id"],
        row["variant"],
    )

    if config.use_disk_cache and path.exists():
        stats.variant_disk_hits += 1
        return load_latent(path)

    image = load_pil_rgb(row["variant_path"])
    z = encode_image(encoder, transform, image, device)
    stats.variant_encoded += 1

    if config.save_latents:
        save_latent(path, z)

    return z


FIELDNAMES = [
    "class_id",
    "image_id",
    "variant",
    "original_path",
    "variant_path",
    "original_latent_path",
    "variant_latent_path",
    "latent_shape",
    "latent_dim",
    "l2",
    "cosine_similarity",
    "cosine_distance",
    "angle_radian",
    "norm_original",
    "norm_variant",
    "norm_difference",
    "relative_l2",
]


def append_csv_row(
    path: Path,
    fieldnames: list[str],
    row: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists() or path.stat().st_size == 0

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o: Any) -> str:
        if isinstance(o, Path):
            return str(o)
        return str(o)

    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=default),
        encoding="utf-8",
    )


@timed("summarize")
def summarize(config: LatentAnalysisConfig) -> None:
    if not config.metric_csv.exists() or config.metric_csv.stat().st_size == 0:
        print("[WARN] metric CSV does not exist; skip summarize.", flush=True)
        return

    df = pd.read_csv(config.metric_csv)

    if df.empty:
        print("[WARN] metric CSV is empty; skip summarize.", flush=True)
        return

    metric_cols = [
        "l2",
        "cosine_similarity",
        "cosine_distance",
        "angle_radian",
        "norm_original",
        "norm_variant",
        "norm_difference",
        "relative_l2",
    ]

    variant_summary = df.groupby("variant")[metric_cols].agg(
        ["mean", "std", "median", "min", "max"]
    )
    variant_summary.columns = [
        "_".join(c)
        for c in variant_summary.columns
    ]
    variant_summary = variant_summary.reset_index()

    overall_summary = df[metric_cols].agg(
        ["mean", "std", "median", "min", "max"]
    ).T.reset_index()
    overall_summary.columns = [
        "metric",
        "mean",
        "std",
        "median",
        "min",
        "max",
    ]

    variant_summary.to_csv(
        config.out_dir / "variant_summary.csv",
        index=False,
    )
    overall_summary.to_csv(
        config.out_dir / "overall_summary.csv",
        index=False,
    )

    print("[SAVED]", config.out_dir / "variant_summary.csv", flush=True)
    print("[SAVED]", config.out_dir / "overall_summary.csv", flush=True)


@timed("run_pipeline")
def run_pipeline(config: LatentAnalysisConfig) -> RuntimeStats:
    config.out_dir.mkdir(parents=True, exist_ok=True)

    if config.save_latents:
        config.latent_dir.mkdir(parents=True, exist_ok=True)

    write_json(config.config_json, asdict(config))

    set_seed(config.seed)

    device = resolve_device(config.device)
    print("[INFO] device:", device, flush=True)

    transform = build_eval_transform(
        config.resize_size,
        config.crop_size,
    )

    print("[INFO] loading AE encoder...", flush=True)
    encoder = load_encoder_from_imagenet_ae(config, device)
    print("[INFO] AE encoder loaded.", flush=True)

    stats = RuntimeStats()

    done_keys = load_done_keys(config.metric_csv) if config.resume else set()
    original_cache: dict[tuple[str, str], torch.Tensor] = {}

    if done_keys:
        print(
            f"[INFO] resume: {len(done_keys)} done pair keys loaded",
            flush=True,
        )

    for row in iter_pair_rows(config):
        stats.total_pairs += 1

        append_csv_row(
            config.pairs_all_csv,
            list(row.keys()),
            row,
        )

        if not (row["original_exists"] and row["variant_exists"]):
            stats.missing_pairs += 1
            append_csv_row(
                config.missing_pairs_csv,
                list(row.keys()),
                row,
            )
            continue

        stats.valid_pairs += 1

        key = pair_key(row)

        if key in done_keys:
            stats.skipped_pairs += 1
            continue

        try:
            z_orig = get_original_latent(
                config,
                encoder,
                transform,
                row,
                device,
                original_cache,
                stats,
            )

            z_var = get_variant_latent(
                config,
                encoder,
                transform,
                row,
                device,
                stats,
            )

            metric_dict = compute_metrics(z_orig, z_var)

            orig_latent_path = latent_path_for(
                config,
                row["class_id"],
                row["image_id"],
                "original",
            )
            var_latent_path = latent_path_for(
                config,
                row["class_id"],
                row["image_id"],
                row["variant"],
            )

            output_row = {
                "class_id": row["class_id"],
                "image_id": row["image_id"],
                "variant": row["variant"],
                "original_path": row["original_path"],
                "variant_path": row["variant_path"],
                "original_latent_path": str(orig_latent_path),
                "variant_latent_path": str(var_latent_path),
                "latent_shape": str(tuple(z_orig.shape)),
                "latent_dim": int(z_orig.numel()),
                **metric_dict,
            }

            append_csv_row(
                config.metric_csv,
                FIELDNAMES,
                output_row,
            )

            done_keys.add(key)
            stats.metric_rows_written += 1

        except Exception as e:
            stats.failed_pairs += 1

            failed_row = {
                **row,
                "error": repr(e),
            }

            append_csv_row(
                config.failed_pairs_csv,
                list(failed_row.keys()),
                failed_row,
            )

            print(f"[WARN] failed {key}: {e}", flush=True)

        if stats.valid_pairs > 0 and stats.valid_pairs % config.progress_every == 0:
            print(
                f"[PROGRESS] valid={stats.valid_pairs}, "
                f"written={stats.metric_rows_written}, "
                f"skipped={stats.skipped_pairs}, "
                f"missing={stats.missing_pairs}, "
                f"failed={stats.failed_pairs}",
                flush=True,
            )
            write_json(config.run_stats_json, stats.to_dict())

    stats.finish()

    write_json(config.run_stats_json, stats.to_dict())

    print("[DONE]", flush=True)
    print("[META]", config.metric_csv, flush=True)
    print("[STATS]", json.dumps(stats.to_dict(), indent=2), flush=True)

    summarize(config)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimized AE latent metric pipeline for ImageNetS919 variants."
    )

    parser.add_argument(
        "--meta-path",
        type=Path,
        default=DEFAULT_META_PATH,
    )
    parser.add_argument(
        "--variant-root",
        type=Path,
        default=DEFAULT_VARIANT_ROOT,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )
    parser.add_argument(
        "--external-repo-root",
        type=Path,
        default=DEFAULT_EXTERNAL_REPO_ROOT,
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_AE_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--arch",
        default="vgg16",
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--resize-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--limit-images",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--no-save-latents",
        action="store_true",
    )
    parser.add_argument(
        "--no-disk-cache",
        action="store_true",
    )
    parser.add_argument(
        "--no-memory-cache",
        action="store_true",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
    )

    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> LatentAnalysisConfig:
    variants = tuple(args.variants) if args.variants else DEFAULT_VARIANTS

    return LatentAnalysisConfig(
        meta_path=args.meta_path,
        variant_root=args.variant_root,
        out_dir=args.out_dir,
        external_repo_root=args.external_repo_root,
        checkpoint_path=args.checkpoint_path,
        arch=args.arch,
        device=args.device,
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        seed=args.seed,
        variants=variants,
        save_latents=not args.no_save_latents,
        use_disk_cache=not args.no_disk_cache,
        use_memory_cache=not args.no_memory_cache,
        resume=not args.no_resume,
        limit_images=args.limit_images,
        progress_every=args.progress_every,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    run_pipeline(config)


if __name__ == "__main__":
    main()
