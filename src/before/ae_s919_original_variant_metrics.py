# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
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


# ---------------------------------------------------------------------
# ImageNetS919 metadata and generated variant image paths
# ---------------------------------------------------------------------
# The real ImageNetS919 images are not included in this repository.
# Download ImageNet-S separately and place the metadata/image files
# according to the structure described in README.md.
META_PATH = Path(
    "ImageNetS919/annotation_hstack_colormap_metadata.csv"
)

VARIANT_ROOT = Path(
    "ImageNetS919_single_object_variants_white_mask"
)


# ---------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------
OUT_DIR = Path("results/before_run")
LATENT_DIR = OUT_DIR / "latents"


# ---------------------------------------------------------------------
# External ImageNet Autoencoder repository and checkpoint
# ---------------------------------------------------------------------
AE_REPO_URL = "https://github.com/Horizon2333/imagenet-autoencoder"

EXTERNAL_REPO_ROOT = Path("imagenet-autoencoder")

AE_CHECKPOINT_PATH = Path(
    "imagenet-autoencoder/checkpoints/imagenet-vgg16.pth"
)

ARCH = "vgg16"


# ---------------------------------------------------------------------
# Runtime options
# ---------------------------------------------------------------------
DEVICE = "cuda"
RESIZE_SIZE = 256
CROP_SIZE = 224
SEED = 42

VARIANTS = [
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
]

PNG_VARIANTS = {"bg_transparent", "cutout_rgba", "mask_white"}

LIMIT_IMAGES = None
SAVE_LATENTS = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


def build_eval_transform(resize_size, crop_size):
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ])


def load_pil_rgb(path):
    return Image.open(path).convert("RGB")


def load_encoder_from_imagenet_ae(repo_root, checkpoint_path, arch, device="cuda"):
    repo_root = os.path.abspath(repo_root)

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import utils
    import models.builer as builder

    args = SimpleNamespace()
    args.arch = arch
    args.parallel = 0
    args.batch_size = 1
    args.workers = 0

    model = builder.BuildAutoEncoder(args)
    utils.load_dict(checkpoint_path, model)

    model = model.to(device)
    model.eval()

    def encoder_fn(x):
        return model.module.encoder(x)

    return encoder_fn


@torch.no_grad()
def encode_image(encoder, transform, image, device):
    x = transform(image).unsqueeze(0).to(device, non_blocking=True)
    z = encoder(x)
    z = z.flatten(start_dim=1)
    return z[0].detach().cpu().float()


def compute_metrics(z_orig, z_var):
    z_orig = z_orig.float().flatten()
    z_var = z_var.float().flatten()

    diff = z_var - z_orig

    l2 = torch.linalg.norm(diff).item()
    norm_original = torch.linalg.norm(z_orig).item()
    norm_variant = torch.linalg.norm(z_var).item()

    cos = F.cosine_similarity(
        z_orig.unsqueeze(0),
        z_var.unsqueeze(0),
        dim=1,
    ).item()
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


def variant_path_for(class_id, image_id, variant):
    ext = ".png" if variant in PNG_VARIANTS else ".jpg"
    return VARIANT_ROOT / variant / class_id / f"{image_id}{ext}"


def latent_path_for(class_id, image_id, condition):
    return LATENT_DIR / class_id / f"{image_id}_{condition}_latent.npy"


def save_latent(path, z):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, z.numpy().astype(np.float32))


def build_pair_rows():
    df = pd.read_csv(META_PATH)
    df = df[df["n_fg_labels"] == 1].copy()
    df = df.sort_values(["class_id", "image_id"]).reset_index(drop=True)

    if LIMIT_IMAGES is not None:
        df = df.head(LIMIT_IMAGES).copy()

    rows = []

    for _, r in df.iterrows():
        class_id = str(r["class_id"])
        image_id = str(r["image_id"])
        original_path = Path(str(r["original_path"]))

        for variant in VARIANTS:
            variant_path = variant_path_for(class_id, image_id, variant)
            rows.append({
                "class_id": class_id,
                "image_id": image_id,
                "variant": variant,
                "original_path": str(original_path),
                "variant_path": str(variant_path),
                "original_exists": original_path.exists(),
                "variant_exists": variant_path.exists(),
            })

    return rows


def summarize(meta_csv):
    df = pd.read_csv(meta_csv)

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
    variant_summary.columns = ["_".join(c) for c in variant_summary.columns]
    variant_summary = variant_summary.reset_index()

    overall_summary = df[metric_cols].agg(
        ["mean", "std", "median", "min", "max"]
    ).T.reset_index()
    overall_summary.columns = ["metric", "mean", "std", "median", "min", "max"]

    variant_summary.to_csv(OUT_DIR / "variant_summary.csv", index=False)
    overall_summary.to_csv(OUT_DIR / "overall_summary.csv", index=False)

    print("[SAVED]", OUT_DIR / "variant_summary.csv", flush=True)
    print("[SAVED]", OUT_DIR / "overall_summary.csv", flush=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if SAVE_LATENTS:
        LATENT_DIR.mkdir(parents=True, exist_ok=True)

    device = DEVICE if torch.cuda.is_available() else "cpu"
    print("[INFO] device:", device, flush=True)

    rows = build_pair_rows()

    valid_rows = [
        r for r in rows
        if r["original_exists"] and r["variant_exists"]
    ]
    missing_rows = [
        r for r in rows
        if not (r["original_exists"] and r["variant_exists"])
    ]

    pd.DataFrame(rows).to_csv(OUT_DIR / "pairs_all.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(
        OUT_DIR / "missing_pairs.csv",
        index=False,
    )

    print("[INFO] total pairs:", len(rows), flush=True)
    print("[INFO] valid pairs:", len(valid_rows), flush=True)
    print("[INFO] missing pairs:", len(missing_rows), flush=True)

    transform = build_eval_transform(RESIZE_SIZE, CROP_SIZE)

    print("[INFO] loading AE encoder...", flush=True)
    encoder = load_encoder_from_imagenet_ae(
        repo_root=str(EXTERNAL_REPO_ROOT),
        checkpoint_path=str(AE_CHECKPOINT_PATH),
        arch=ARCH,
        device=device,
    )
    print("[INFO] AE encoder loaded.", flush=True)

    meta_csv = OUT_DIR / "image_level_metrics.csv"
    progress_json = OUT_DIR / "progress.json"

    fieldnames = [
        "row_index",
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

    start_idx = 0

    if progress_json.exists() and meta_csv.exists():
        try:
            progress = json.loads(progress_json.read_text())
            start_idx = int(progress.get("processed_pairs", 0))
            print("[INFO] resume from:", start_idx, flush=True)
        except Exception:
            start_idx = 0

    write_header = not meta_csv.exists() or start_idx == 0

    f = open(meta_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    if write_header:
        writer.writeheader()

    try:
        for idx, row in enumerate(valid_rows):
            if idx < start_idx:
                continue

            try:
                original_img = load_pil_rgb(row["original_path"])
                variant_img = load_pil_rgb(row["variant_path"])

                z_orig = encode_image(encoder, transform, original_img, device)
                z_var = encode_image(encoder, transform, variant_img, device)

                metric_dict = compute_metrics(z_orig, z_var)

                class_id = row["class_id"]
                image_id = row["image_id"]
                variant = row["variant"]

                orig_latent_path = latent_path_for(
                    class_id,
                    image_id,
                    "original",
                )
                var_latent_path = latent_path_for(
                    class_id,
                    image_id,
                    variant,
                )

                if SAVE_LATENTS:
                    if not orig_latent_path.exists():
                        save_latent(orig_latent_path, z_orig)
                    save_latent(var_latent_path, z_var)

                writer.writerow({
                    "row_index": idx,
                    "class_id": class_id,
                    "image_id": image_id,
                    "variant": variant,
                    "original_path": row["original_path"],
                    "variant_path": row["variant_path"],
                    "original_latent_path": str(orig_latent_path),
                    "variant_latent_path": str(var_latent_path),
                    "latent_shape": str(tuple(z_orig.shape)),
                    "latent_dim": int(z_orig.numel()),
                    **metric_dict,
                })

            except Exception as e:
                print(
                    f"[WARN] failed row {idx} "
                    f"{row['image_id']} {row['variant']}: {e}",
                    flush=True,
                )

            if (idx + 1) % 100 == 0:
                f.flush()
                progress_json.write_text(json.dumps({
                    "processed_pairs": idx + 1,
                    "total_pairs": len(valid_rows),
                }, indent=2))
                print(
                    f"[PROGRESS] {idx + 1}/{len(valid_rows)}",
                    flush=True,
                )

    finally:
        f.flush()
        f.close()

    progress_json.write_text(json.dumps({
        "processed_pairs": len(valid_rows),
        "total_pairs": len(valid_rows),
    }, indent=2))

    print("[DONE]", flush=True)
    print("[META]", meta_csv, flush=True)

    summarize(meta_csv)


if __name__ == "__main__":
    main()


