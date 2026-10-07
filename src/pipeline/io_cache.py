# -*- coding: utf-8 -*-
"""
Cache I/O for the MSc zero-shot counting project.

Reads (and later writes) the cache formats created by the B2, ABC and M1
notebooks, using EXACTLY the same file naming and array encodings, so the
rebuilt pipeline can consume the frozen dev100 caches and produce new
caches (e.g. held-out FSC-147 validation) in the same format.

Cache formats (as verified from the frozen dev100 caches)
---------------------------------------------------------
B0 masks (B2):
    {dataset_index:05d}_{stem}_B0_masks.npz
    packed_masks   (N, ceil(H*W/8)) uint8, np.packbits, bitorder="little"
    image_height, image_width, num_masks, bitorder
    areas, predicted_iou, stability_score, bbox_xywh, crop_box_xywh, point_xy

A3c masks (ABC):
    {dataset_index:05d}_{stem}_A3c.npz
    mask (H, W) uint8, threshold, percentile, height, width, ...

Marigold depth (M1):
    idx_{dataset_index:04d}__{stem}.npz
    depth (H, W) float32

M1 raw depth-split labels (M1 Step 7B):
    {dataset_index:05d}_{stem}_raw_depth_split_labels.npz
    proposal_labels (H, W) int32, original_num_components,
    raw_num_subcomponents, num_candidate_components, ...
"""

from pathlib import Path

import numpy as np


# ============================================================
# Default project locations
# ============================================================

PROJECT_ROOT = Path("/content/drive/MyDrive/MSc_Project_ZeroShot_Counting")
RESULTS_ROOT = PROJECT_ROOT / "results"

DEV100_CACHE_DIRS = {
    "b0_masks": (
        RESULTS_ROOT
        / "B1_experiments"
        / "B2_B0_Anchored_Semantic_Validation"
        / "cache"
        / "B0_masks"
    ),
    "a3c_masks": (
        RESULTS_ROOT / "B1_experiments" / "ABC_artifacts" / "A3c_masks"
    ),
    "depth": (
        RESULTS_ROOT / "M1_Marigold_Geometric_Diagnostics" / "depth_cache"
    ),
    "m1_split_labels": (
        RESULTS_ROOT
        / "M1_Marigold_Geometric_Diagnostics"
        / "step7_geometry_refinement"
        / "raw_split_label_maps"
    ),
}

# SAN score-map cache (ABC): cache/B1_SAN/FSC147_val_dev_100/{stem}.pt
SAN_DEV100_CACHE_DIR = PROJECT_ROOT / "cache" / "B1_SAN" / "FSC147_val_dev_100"


# ============================================================
# File naming (identical to the frozen caches)
# ============================================================

def image_stem(image_id):
    """'215.jpg' -> '215'"""
    return Path(str(image_id)).stem


def b0_masks_filename(dataset_index, image_id):
    return f"{int(dataset_index):05d}_{image_stem(image_id)}_B0_masks.npz"


def a3c_mask_filename(dataset_index, image_id):
    return f"{int(dataset_index):05d}_{image_stem(image_id)}_A3c.npz"


def depth_filename(dataset_index, image_id):
    return f"idx_{int(dataset_index):04d}__{image_stem(image_id)}.npz"


def m1_split_filename(dataset_index, image_id):
    return (
        f"{int(dataset_index):05d}_{image_stem(image_id)}"
        f"_raw_depth_split_labels.npz"
    )


def san_scores_filename(image_id):
    return f"{image_stem(image_id)}.pt"


def cache_paths(dataset_index, image_id, cache_dirs=None):
    """Return the four cache paths for one image."""
    d = DEV100_CACHE_DIRS if cache_dirs is None else cache_dirs
    return {
        "b0_masks": Path(d["b0_masks"])
        / b0_masks_filename(dataset_index, image_id),
        "a3c_masks": Path(d["a3c_masks"])
        / a3c_mask_filename(dataset_index, image_id),
        "depth": Path(d["depth"])
        / depth_filename(dataset_index, image_id),
        "m1_split_labels": Path(d["m1_split_labels"])
        / m1_split_filename(dataset_index, image_id),
    }


# ============================================================
# Binary mask packing (lossless, matches B2 / End-to-End caches)
# ============================================================

def pack_masks(mask_stack, bitorder="little"):
    """
    (N, H, W) bool -> (N, ceil(H*W/8)) uint8 using np.packbits.
    N may be zero.
    """
    mask_stack = np.asarray(mask_stack, dtype=bool)
    assert mask_stack.ndim == 3, f"Expected (N,H,W), got {mask_stack.shape}"
    n, h, w = mask_stack.shape
    if n == 0:
        return np.zeros((0, (h * w + 7) // 8), dtype=np.uint8)
    return np.packbits(
        mask_stack.reshape(n, h * w), axis=1, bitorder=bitorder
    )


def unpack_masks(packed, height, width, bitorder="little"):
    """(N, ceil(H*W/8)) uint8 -> (N, H, W) bool."""
    packed = np.asarray(packed, dtype=np.uint8)
    height, width = int(height), int(width)
    if packed.ndim != 2 or packed.shape[0] == 0:
        return np.zeros((0, height, width), dtype=bool)
    flat = np.unpackbits(
        packed, axis=1, count=height * width, bitorder=bitorder
    )
    return flat.reshape(packed.shape[0], height, width).astype(bool)


# ============================================================
# Loaders
# ============================================================

def load_b0_masks(path):
    """
    Load frozen B0 AMG masks.

    Returns
    -------
    masks : (N, H, W) bool
    meta  : dict with height, width, num_masks and per-mask AMG metadata
    """
    with np.load(path, allow_pickle=False) as data:
        height = int(data["image_height"])
        width = int(data["image_width"])
        num_masks = int(data["num_masks"])
        bitorder = str(data["bitorder"])
        masks = unpack_masks(data["packed_masks"], height, width, bitorder)
        meta = {
            "height": height,
            "width": width,
            "num_masks": num_masks,
            "areas": np.asarray(data["areas"]),
            "predicted_iou": np.asarray(data["predicted_iou"]),
            "stability_score": np.asarray(data["stability_score"]),
            "bbox_xywh": np.asarray(data["bbox_xywh"]),
            "point_xy": np.asarray(data["point_xy"]),
        }
    assert masks.shape[0] == num_masks, (
        f"{path}: unpacked {masks.shape[0]} masks, num_masks={num_masks}"
    )
    if num_masks > 0:
        assert np.array_equal(
            masks.reshape(num_masks, -1).sum(axis=1), meta["areas"]
        ), f"{path}: unpacked mask areas do not match stored areas"
    return masks, meta


def load_a3c_mask(path):
    """Load frozen A3c semantic mask -> (H, W) bool, meta."""
    with np.load(path, allow_pickle=False) as data:
        mask = np.asarray(data["mask"]).astype(bool)
        meta = {
            "threshold": float(data["threshold"]),
            "percentile": float(data["percentile"]),
            "height": int(data["height"]),
            "width": int(data["width"]),
        }
    assert mask.shape == (meta["height"], meta["width"])
    return mask, meta


def load_depth(path):
    """Load frozen Marigold relative depth -> (H, W) float32 (as in M1)."""
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["depth"]).astype(np.float32)


def load_m1_split_labels(path):
    """Load M1 Step-7B raw split label map + summary counts (for checks)."""
    with np.load(path, allow_pickle=False) as data:
        return {
            "proposal_labels": np.asarray(data["proposal_labels"]),
            "original_num_components": int(data["original_num_components"]),
            "raw_num_subcomponents": int(data["raw_num_subcomponents"]),
            "num_candidate_components": int(
                data["num_candidate_components"]
            ),
        }


def load_san_scores(path):
    """Load a frozen ABC SAN cache (.pt) -> dict with float32 numpy maps."""
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "image_id": obj["image_id"],
        "target_category": obj["target_category"],
        "vocabulary": list(obj["vocabulary"]),
        "height": int(obj["height"]),
        "width": int(obj["width"]),
        "target_score": obj["target_score"].numpy().astype(np.float32),
        "background_score": obj["background_score"].numpy().astype(np.float32),
    }


# ============================================================
# Writers (same formats as the frozen caches)
# ============================================================

def save_b0_masks(path, masks, meta):
    """Write B0 masks in the B2 cache format."""
    masks = np.asarray(masks, dtype=bool)
    n, h, w = masks.shape
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        packed_masks=pack_masks(masks, "little"),
        image_height=np.int32(h),
        image_width=np.int32(w),
        num_masks=np.int32(n),
        bitorder=np.array("little"),
        areas=np.asarray(meta["areas"], dtype=np.int64),
        predicted_iou=np.asarray(meta["predicted_iou"], dtype=np.float32),
        stability_score=np.asarray(meta["stability_score"], dtype=np.float32),
        bbox_xywh=np.asarray(meta["bbox_xywh"], dtype=np.float32).reshape(n, 4),
        crop_box_xywh=np.asarray(
            meta["crop_box_xywh"], dtype=np.float32
        ).reshape(n, 4),
        point_xy=np.asarray(meta["point_xy"], dtype=np.float32).reshape(n, 2),
    )


def save_a3c_mask(path, mask, threshold, percentile, dataset_index,
                  image_id, category):
    """Write an A3c mask in the ABC artefact format."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mask=mask.astype(np.uint8),
        dataset_index=np.int32(dataset_index),
        image_id=np.array(str(image_id)),
        category=np.array(str(category)),
        threshold=np.float32(threshold),
        percentile=np.float32(percentile),
        height=np.int32(h),
        width=np.int32(w),
    )


def save_depth(path, depth):
    """Write depth in the M1 cache format."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, depth=np.asarray(depth, dtype=np.float32))


def save_san_scores(path, image_id, class_name, target_score,
                    background_score):
    """Write SAN score maps in the ABC cache format (.pt)."""
    import torch

    target_score = np.asarray(target_score, dtype=np.float32)
    h, w = target_score.shape
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_id": str(image_id),
            "target_category": str(class_name),
            "vocabulary": [str(class_name), "background"],
            "height": int(h),
            "width": int(w),
            "target_score": torch.from_numpy(target_score),
            "background_score": torch.from_numpy(
                np.asarray(background_score, dtype=np.float32)
            ),
        },
        path,
    )


# ============================================================
# End-to-End checkpoint guided masks (B1 checkpoint cache)
# ============================================================

CHECKPOINT_GUIDED_CACHE = (
    RESULTS_ROOT / "B1_experiments" / "End_to_End_Checkpoint" / "guided_mask_cache"
)


def guided_cache_path(system, dataset_index, image_id, root=None):
    root = CHECKPOINT_GUIDED_CACHE if root is None else Path(root)
    return root / system / f"{int(dataset_index):04d}_{image_stem(image_id)}.npz"


def load_guided_masks(path):
    """Checkpoint guided masks: packed little-endian, one mask per prompt."""
    with np.load(path, allow_pickle=False) as d:
        h, w = int(d["height"]), int(d["width"])
        return {
            "masks": unpack_masks(d["packed_masks"], h, w, bitorder="little"),
            "prompts_xy": np.asarray(d["prompt_points_xy"], dtype=np.float64).reshape(-1, 2),
            "scores": np.asarray(d["sam2_scores"], dtype=np.float64),
            "areas": np.asarray(d["mask_area_px"], dtype=np.int64),
        }
