# -*- coding: utf-8 -*-
"""
Frozen candidates for the second, pre-declared held-out comparison (B4c).

Five systems chosen at the end of the improvement phase (B3S-B3AC, developed
on FSC-147 train_dev400). Every counting function below is copied from the
notebook that produced the train-pool counts and must reproduce them exactly:

    B3PN_Part_and_Normals_Merge  -> GPV_R4+P[MixA], GPV_K-n+P[MixA],
                                    GPV_K-n+P[LM0], H1b-K3n+P[LM0]
    B3AC_Area_Cap_and_P_on_S     -> H1b+P[MixA] (and the four above again)

Candidates (keys equal the per-image CSV column names):

    1  GPV_R4+P[MixA]   R4 mask count after P               (semantic-only arm)
    2  GPV_K-n+P[MixA]  GPV-K with normal-edge watershed K3n, after P
    3  GPV_K-n+P[LM0]   as 2, on the unzoomed LM0 mask set
    4  H1b+P[MixA]      H1b_v1 counting (depth G2), after P
    5  H1b-K3n+P[LM0]   H1b with K3n on rule-A merged masks, after P

Inputs are cached artefacts only (no models are run here):

    tiled_argmax  san_vocab_tiled/{stem}.npz       expanded-vocabulary SAN,
                                                   2x2 tiles (60%), class 0 = target
    normals       normals/{stem}.npz               Marigold normals v1-1
    LM0           sam2_large_masked_M0/{b0,guided} SAM2.1-L on the masked image
    Z-LM0         zoomed/LM0/b0                    as LM0 automatic, crop_n_layers 1
    SAN target    san/{stem}.pt                    original SAN score (for D1 peaks)
    A3c, depth    a3c_masks/, depth/               frozen caches (H1b family)

The "> 50% outside the semantic region" mask discard was applied when the
LM0 / Z-LM0 caches were written (B4b keep_inside: kept if inside > 0.5 x area),
so it is not repeated here.

Empty mask set: every candidate counts 0. (B3AC writes 0 explicitly; B3PN's
functions also return 0 because H1b/SG with an empty anchor and an empty
recovery region recovers nothing.)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import gpv
from . import hybrid
from . import io_cache
from . import multiplicity as K

VERSION = "candidates_v1"

# ============================================================
# Frozen constants
# ============================================================

NMS_IOU = 0.5                 # guided NMS and R4 pooling
PART_FRAC = 0.25              # P: part if area < 0.25 x object size ...
NEST_FRAC = 0.80              # ... AND > 80% of it lies inside another kept mask
MIN_VERIFIED = 3              # object size: verified singles needed before fallback
K3N_H_FRAC = 0.10             # K3n marker dynamic: 0.1 x (p95 - p5) of edge in region
K3N_MIN_SEG_FRAC = 0.25       # K3n segments counted if area >= 0.25 x a_ref
REGION_MIN_FRAC = 0.50        # merged-mask region qualifies if area >= 0.5 x a_ref
NORMAL_EDGE_SIGMA = 1.0

CANDIDATES = {
    "GPV_R4+P[MixA]":  {"base": "MixA", "counter": "GPV_R4"},
    "GPV_K-n+P[MixA]": {"base": "MixA", "counter": "GPV_K-n"},
    "GPV_K-n+P[LM0]":  {"base": "LM0",  "counter": "GPV_K-n"},
    "H1b+P[MixA]":     {"base": "MixA", "counter": "H1b"},
    "H1b-K3n+P[LM0]":  {"base": "LM0",  "counter": "H1b-K3n"},
}
CANDIDATE_NAMES = list(CANDIDATES)

# Mask-set bases: (automatic-mask root, guided-mask root), relative to a cache root
BASES = {
    "MixA": ("zoomed/LM0", "sam2_large_masked_M0"),            # Z-LM0 auto + LM0 guided
    "LM0":  ("sam2_large_masked_M0", "sam2_large_masked_M0"),  # LM0 auto + LM0 guided
}


# ============================================================
# Normal edges and K3n (B3MN / B3PN Step 1.3)
# ============================================================

def normal_edge_map(nrm):
    """(H, W, 3) normals -> (H, W) float32 edge map.

    Per pixel: max over the 4 neighbours of (1 - cos) between unit normals
    (np.roll; border rows/cols copied from their inner neighbours), then
    Gaussian sigma 1.
    """
    n = nrm.astype(np.float32)
    n = n / np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-6)
    edge = np.zeros(n.shape[:2], np.float32)
    for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        edge = np.maximum(edge, 1.0 - (n * np.roll(n, shift=(dy, dx), axis=(0, 1))).sum(axis=2))
    edge[0, :] = edge[1, :]; edge[-1, :] = edge[-2, :]; edge[:, 0] = edge[:, 1]; edge[:, -1] = edge[:, -2]
    return ndimage.gaussian_filter(edge, NORMAL_EDGE_SIGMA)


def k3n_segments(region, edge, min_seg, h_frac=K3N_H_FRAC):
    """Marker watershed on the normal-edge map inside `region`;
    markers = h-minima of edge (h = h_frac x (p95 - p5)); h <= 0 -> 1."""
    e = edge[region]
    h = h_frac * float(np.percentile(e, 95) - np.percentile(e, 5))
    return 1 if h <= 0 else K._marker_watershed(edge, region, -edge, h, min_seg)


# ============================================================
# Mask sets (R4 from cached LM0 / Z-LM0 masks)
# ============================================================

def r4_from_files(b0_npz, guided_npz):
    """Guided NMS 0.5, then R4 = guided kept U automatic masks, NMS 0.5.
    Returns (masks (N, H, W) bool, scores (N,) float)."""
    bm, bmeta = io_cache.load_b0_masks(b0_npz)
    g = io_cache.load_guided_masks(guided_npz)
    kg = gpv.nms_masks(g["masks"], g["scores"], NMS_IOU) if g["masks"].shape[0] else np.zeros(0, int)
    m, s, _ = gpv.build_r4(g["masks"][kg], np.asarray(g["scores"])[kg], bm, bmeta["predicted_iou"], NMS_IOU)
    return m, np.asarray(s, float)


# ============================================================
# Object size and part-mask filter P (B3PN / B3AC Step 2.1)
# ============================================================

def object_size(masks, scores, amax, d1):
    """Median area of verified single masks:
    target-dominant (most frequent tiled_argmax class inside the mask is 0)
    AND not rule-A merged AND SAM2 score >= image median.
    Fallbacks: target-dominant masks (if < 3 verified), then all masks."""
    n = masks.shape[0]
    areas = masks.reshape(n, -1).sum(1).astype(float)
    n_cls = int(amax.max()) + 1
    dom_t = np.array([np.bincount(amax[m], minlength=n_cls).argmax() == 0 if m.any() else False for m in masks])
    merged = gpv.merged_mask_flags(masks, gpv.points_per_mask(masks, d1), "A")
    verified = dom_t & ~merged & (np.asarray(scores) >= np.median(scores))
    if verified.sum() >= MIN_VERIFIED:
        return float(np.median(areas[verified]))
    if dom_t.any():
        return float(np.median(areas[dom_t]))
    return float(np.median(areas))


def part_filter(masks, scores, amax, d1):
    """P: drop (simultaneously) every mask with area < 0.25 s AND
    max intersection with any other mask of the set > 0.80 x own area.
    s = object_size of the UNFILTERED set. Sets with < 2 masks are unchanged.
    Returns (masks, scores, n_dropped, s) — s is NaN when not computed."""
    n = masks.shape[0]
    if n < 2:
        return masks, np.asarray(scores, float), 0, float("nan")
    s = object_size(masks, scores, amax, d1)
    flat = masks.reshape(n, -1).astype(np.float32)
    areas = flat.sum(1)
    inter = flat @ flat.T
    np.fill_diagonal(inter, 0)
    nested = (inter.max(axis=1) > NEST_FRAC * np.maximum(areas, 1))
    drop = (areas < PART_FRAC * s) & nested
    keep = ~drop
    return masks[keep], np.asarray(scores, float)[keep], int(drop.sum()), s


# ============================================================
# Counters
# ============================================================

def gpv_r4_count(masks, scores=None, **_):
    return int(np.asarray(masks).shape[0])


def gpv_k_n_count(masks, scores, d1, edge, **_):
    """GPV-K with K3n: rule-A flags; a_ref = median non-flagged area;
    regions by descending score (K.build_regions), area >= 0.5 a_ref;
    Gate 1: D1 peaks in regions > #non-flagged -> #non-flagged + sum
    max(1, K3n(region, 0.25 a_ref)); else #masks."""
    masks = np.asarray(masks, bool)
    if masks.shape[0] == 0:
        return 0
    flagged = gpv.merged_mask_flags(masks, gpv.points_per_mask(masks, d1), "A")
    a_ref = K.reference_area(masks, flagged)
    n_nonflag = int((~flagged).sum())
    regions = [rg for _, rg in K.build_regions(masks, scores, flagged)
               if rg.sum() > 0 and rg.sum() >= REGION_MIN_FRAC * a_ref]
    if not (regions and sum(K.k1_peaks(rg, d1) for rg in regions) > n_nonflag):
        return int(masks.shape[0])
    return n_nonflag + sum(max(1, k3n_segments(rg, edge, K3N_MIN_SEG_FRAC * a_ref)) for rg in regions)


def h1b_count(masks, d1, a3c, depth, sg_cfg, **_):
    """H1b_v1 counting rule on the given mask set."""
    masks = np.asarray(masks, bool)
    if masks.shape[0] == 0:
        return 0.0
    return float(hybrid.h1b_count(masks, d1, a3c, depth, sg_cfg)[0])


def h1b_k3n_count(masks, d1, a3c, depth, edge, sg_cfg, **_):
    """H1b with K3n on rule-A merged masks, applied only when H1b's Stage 2 is ON
    and merged masks exist; otherwise the H1b count.
    anchor = non-merged masks; a_ref = median anchor area (fallback: median
    merged area); merged masks visited in INDEX order (not by score), each
    region = mask minus anchor union minus earlier regions; qualifying
    regions (>= 0.5 a_ref) add max(1, K3n(region, 0.25 a_ref))."""
    r4_m = np.asarray(masks, bool)
    if r4_m.shape[0] == 0:
        return 0.0
    base, info = hybrid.h1b_count(r4_m, d1, a3c, depth, sg_cfg)
    if not info["stage2_on"]:
        return float(base)
    merged = gpv.merged_mask_flags(r4_m, gpv.points_per_mask(r4_m, d1), "A")
    if not merged.any():
        return float(base)
    anchor = r4_m[~merged]
    areas = anchor.reshape(anchor.shape[0], -1).sum(1) if anchor.shape[0] else np.zeros(0)
    a_ref = (float(np.median(areas)) if areas.size
             else float(np.median(r4_m[merged].reshape(merged.sum(), -1).sum(1))))
    taken = anchor.any(axis=0) if anchor.shape[0] else np.zeros(r4_m.shape[1:], bool)
    total = int(anchor.shape[0])
    for i in np.flatnonzero(merged):
        rg = r4_m[i] & ~taken
        taken |= rg
        if rg.sum() > 0 and rg.sum() >= REGION_MIN_FRAC * a_ref:
            total += max(1, k3n_segments(rg, edge, K3N_MIN_SEG_FRAC * a_ref))
    return float(total)


COUNTERS = {"GPV_R4": gpv_r4_count, "GPV_K-n": gpv_k_n_count,
            "H1b": h1b_count, "H1b-K3n": h1b_k3n_count}


# ============================================================
# Per-image inputs and the five counts
# ============================================================

@dataclass
class CandidatePaths:
    """Cache directories for one subset. `root` holds the four candidate caches
    (san_vocab_tiled, normals, sam2_large_masked_M0, zoomed/LM0); the other
    directories may live elsewhere (e.g. dev100 uses the B3R fresh caches)."""
    root: Path
    san_dir: Path
    a3c_dir: Path
    depth_dir: Path

    @classmethod
    def standard(cls, root):
        """Layout of train_dev400 and held-out: everything under one root."""
        root = Path(root)
        return cls(root, root / "san", root / "a3c_masks", root / "depth")


def load_image_inputs(paths, dataset_index, image_id):
    """Load all cached inputs for one image. Returns a dict."""
    di, stem = int(dataset_index), io_cache.image_stem(image_id)
    dirs = {"b0_masks": paths.root, "a3c_masks": paths.a3c_dir,
            "depth": paths.depth_dir, "m1_split_labels": paths.root}
    p = io_cache.cache_paths(di, image_id, dirs)
    a3c, _ = io_cache.load_a3c_mask(p["a3c_masks"])
    depth = io_cache.load_depth(p["depth"])
    san = io_cache.load_san_scores(Path(paths.san_dir) / io_cache.san_scores_filename(image_id))
    with np.load(paths.root / "normals" / f"{stem}.npz") as v:
        edge = normal_edge_map(v["normals"])
    with np.load(paths.root / "san_vocab_tiled" / f"{stem}.npz") as v:
        amax = v["tiled_argmax"]
    sets = {}
    for b, (auto_rel, guided_rel) in BASES.items():
        sets[b] = r4_from_files(paths.root / auto_rel / "b0" / f"{di:05d}_{stem}.npz",
                                paths.root / guided_rel / "guided" / f"{di:04d}_{stem}.npz")
    return {"a3c": a3c, "depth": depth, "d1": gpv.d1_peaks(san["target_score"]),
            "edge": edge, "amax": amax, "sets": sets}


def compute_candidates(inp, sg_cfg, names=CANDIDATE_NAMES):
    """Five candidate counts (+ diagnostics) from `load_image_inputs` output."""
    out, filtered = {}, {}
    for b, (m, s) in inp["sets"].items():
        if m.shape[0] == 0:
            filtered[b] = (m, s)
            out[f"n_r4[{b}]"], out[f"parts_dropped[{b}]"] = 0, 0
            out[f"object_size[{b}]"] = float("nan")
            continue
        mp, sp, n_drop, size = part_filter(m, s, inp["amax"], inp["d1"])
        filtered[b] = (mp, sp)
        out[f"n_r4[{b}]"], out[f"parts_dropped[{b}]"], out[f"object_size[{b}]"] = int(m.shape[0]), n_drop, size
    for name in names:
        spec = CANDIDATES[name]
        mp, sp = filtered[spec["base"]]
        out[name] = COUNTERS[spec["counter"]](mp, scores=sp, d1=inp["d1"], a3c=inp["a3c"],
                                              depth=inp["depth"], edge=inp["edge"], sg_cfg=sg_cfg)
    return out


# ============================================================
# Frozen configuration record (written to Drive by the evaluation notebook)
# ============================================================

def frozen_config(name, sg_config_path=None):
    spec = CANDIDATES[name]
    base_desc = {"MixA": "automatic = Z-LM0 (SAM2.1-L AMG, pps 32, crop_n_layers 1, masked input); "
                         "guided = LM0 (SAM2.1-L, dense_recovery prompts, 1 positive point, multimask off)",
                 "LM0": "automatic = LM0 (SAM2.1-L AMG, pps 32, no crops, masked input); guided = LM0"}
    return {
        "candidate": name, "module_version": VERSION, "counter": spec["counter"], "base": spec["base"],
        "base_description": base_desc[spec["base"]],
        "semantic_region": "SAN expanded vocabulary ([class, background] + STUFF_BACKGROUND_VOCAB), "
                           "2x2 corner tiles of 60%, scores averaged; region = tiled_argmax == 0",
        "mask_discard": "kept if > 50% of pixels inside the region (applied at caching, B4b)",
        "mask_set": f"guided NMS {NMS_IOU} -> R4 (guided U automatic, NMS {NMS_IOU})",
        "P": {"part_frac": PART_FRAC, "nest_frac": NEST_FRAC, "object_size_min_verified": MIN_VERIFIED,
              "object_size": "median area of verified singles (target-dominant, not rule-A, score >= median) on "
                             "the unfiltered set; fallbacks target-dominant, all"},
        "K3n": {"h_frac": K3N_H_FRAC, "min_seg_frac": K3N_MIN_SEG_FRAC, "region_min_frac": REGION_MIN_FRAC,
                "normals": "prs-eth/marigold-normals-v1-1, fp16, 4 steps, ensemble 1, 768 px, seed 0",
                "edge": "max 4-neighbour (1 - cos), borders copied, Gaussian sigma 1"}
               if spec["counter"] in ("GPV_K-n", "H1b-K3n") else None,
        "D1": "gpv.d1_peaks on original SAN target score",
        "sg_config": str(sg_config_path) if spec["counter"].startswith("H1b") else None,
        "empty_mask_set": "count 0",
    }
