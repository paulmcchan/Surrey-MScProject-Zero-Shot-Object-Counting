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
