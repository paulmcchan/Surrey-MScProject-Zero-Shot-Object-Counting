# -*- coding: utf-8 -*-
"""
Multiplicity estimators (K options) for merged masks — B3GK Section 4.

Common protocol (pre-declared):
    flagged  = merged-mask rule A (area > 4 x median kept area AND >= 2 D1 peaks)
    A_ref    = median area of NON-flagged kept masks (fallback: all kept masks)
    r        = max(2, round(sqrt(A_ref / pi)))          typical object radius
    regions  = flagged masks processed by descending score; each region is
               the flagged mask minus pixels of non-flagged kept masks and
               minus regions already assigned to earlier flagged masks
    qualify  = region area >= 0.5 * A_ref  (otherwise multiplicity 0:
               the flagged mask only duplicates objects already counted)
    count    = (# non-flagged kept masks) + sum of region multiplicities
    every qualifying region contributes at least 1

Estimators (multiplicity of one qualifying region):
    K0  : 1                                  region rule only (reference)
    K1b : number of D1 peaks in the region
    K1  : number of D5-768 peaks in the region   (Section 4b)
    K2  : marker watershed on the distance transform (shape);
          markers = h-maxima, h = 1 px
    K3  : marker watershed on the depth-gradient magnitude (depth edges);
          markers = h-minima, h = 0.1 x (p95 - p5) of region gradient
    K2/K3 count segments with area >= 0.25 * A_ref
    K4  : OmniCount depth carving (tolerance 0.2, +/-5 px window,
          opening radius 5), number of connected components
    K5  : round(region area / A_ref)
"""

import numpy as np
from scipy import ndimage

STRUCT4 = ndimage.generate_binary_structure(2, 1)


# ============================================================
# Regions
# ============================================================

def reference_area(masks, flagged):
    masks = np.asarray(masks, dtype=bool)
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.float64)
    pool = areas[~flagged] if (~flagged).sum() > 0 else areas
    return float(np.median(pool)) if pool.size else 0.0


def object_radius_from_area(a_ref):
    return int(max(2, round(np.sqrt(max(a_ref, 0.0) / np.pi))))


def build_regions(masks, scores, flagged):
    """
    Returns list of (mask_index, region_bool) for flagged masks, in
    descending-score order, with non-flagged pixels and earlier regions removed.
    """
    masks = np.asarray(masks, dtype=bool)
    h, w = masks.shape[1:]
    nonflag = masks[~flagged]
    taken = nonflag.any(axis=0) if nonflag.shape[0] else np.zeros((h, w), bool)
    order = [i for i in np.argsort(-np.asarray(scores), kind="stable") if flagged[i]]
    regions = []
    for i in order:
        region = masks[i] & ~taken
        regions.append((int(i), region))
        taken |= region
    return regions


# ============================================================
# Estimators
# ============================================================

def _count_segments(labels, min_area):
    if labels.max() == 0:
        return 0
    sizes = np.bincount(labels.ravel())[1:]
    return int((sizes >= min_area).sum())


def _marker_watershed(elevation, region, marker_map, h, min_area):
    """
    Marker-controlled watershed. Markers = regional maxima of marker_map
    with dynamic >= h (h-maxima), restricted to the region; plateaus give
    one marker. Count = segments with area >= min_area.
    """
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed

    work = np.where(region, marker_map, marker_map[region].min() - 2 * h - 1.0)
    peaks = h_maxima(work, h).astype(bool) & region
    markers, n = ndimage.label(peaks, structure=np.ones((3, 3), bool))
    if n == 0:
        return 1
    labels = watershed(elevation, markers, mask=region)
    return _count_segments(labels, min_area)


def k1_peaks(region, peaks_xy):
    pts = np.asarray(peaks_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return 0
    h, w = region.shape
    px = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
    py = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
    return int(region[py, px].sum())


def k2_watershed_shape(region, min_area, h=1.0):
    """Shape: markers = h-maxima (h = 1 px) of the region's distance transform."""
    dt = ndimage.distance_transform_edt(region)
    return _marker_watershed(-dt, region, dt, h, min_area)


def k3_watershed_depth_gradient(region, depth, min_area, sigma=1.0, h_frac=0.1):
    """
    Depth edges: Sobel gradient magnitude of Gaussian-smoothed (sigma 1)
    depth is the elevation; markers = h-minima of the gradient inside the
    region, h = h_frac x (95th - 5th percentile of gradient in the region).
    """
    d = ndimage.gaussian_filter(np.asarray(depth, dtype=np.float64), sigma)
    grad = np.hypot(ndimage.sobel(d, axis=0), ndimage.sobel(d, axis=1))
    g = grad[region]
    h = h_frac * float(np.percentile(g, 95) - np.percentile(g, 5))
    if h <= 0:
        return 1
    return _marker_watershed(grad, region, -grad, h, min_area)


def k4_omnicount_carving(region, depth, tolerance=0.2, win=5, opening_radius=5):
    """
    OmniCount preprocessing/extract_bin_mask.py, single-class case:
    every pixel within the +/-win window (range(x-win, x+win)) of a region
    pixel joins the refined mask if |depth - mean depth of region| < tolerance;
    then binary opening with a radius-5 cross; count 4-connected components.
    """
    depth = np.asarray(depth, dtype=np.float64)
    mean_depth = float(depth[region].mean())
    # window offsets -win .. win-1 (as range(x-win, x+win))
    footprint = np.zeros((2 * win, 2 * win), dtype=bool)
    footprint[:, :] = True
    grown = ndimage.binary_dilation(region, structure=footprint,
                                    origin=(0, 0)) if region.any() else region
    refined = grown & (np.abs(depth - mean_depth) < tolerance)
    strel = ndimage.iterate_structure(STRUCT4, opening_radius)
    opened = ndimage.binary_opening(refined, structure=strel)
    _, n = ndimage.label(opened, structure=STRUCT4)
    return int(n)


def k5_area_ratio(region, a_ref):
    return int(np.rint(region.sum() / a_ref)) if a_ref > 0 else 1


# ============================================================
# Image-level count
# ============================================================

def count_with_estimator(masks, scores, flagged, estimator, **ctx):
    """
    estimator(region, ctx) -> int. Returns (count, details list).
    ctx must include a_ref; estimators read what they need from it.
    """
    masks = np.asarray(masks, dtype=bool)
    n_nonflag = int((~flagged).sum())
    a_ref = ctx["a_ref"]
    details, total = [], n_nonflag
    for i, region in build_regions(masks, scores, flagged):
        area = int(region.sum())
        if area < 0.5 * a_ref or area == 0:
            m = 0
        else:
            m = max(1, int(estimator(region, ctx)))
        total += m
        details.append({"mask_index": i, "region_area": area, "multiplicity": m})
    return total, details


# ============================================================
# Frozen GPV-K count (B3GK freeze, GPV_K_v1)
# ============================================================

def gpv_k_count(r4_masks, r4_scores, d1_peaks_xy, depth):
    """
    GPV-K = R4 + Gate 1 + K3 (frozen as GPV_K_v1).

    1. Rule A flags: area > 4 x median kept area AND >= 2 D1 peaks.
    2. A_ref = median area of non-flagged masks; regions by descending score;
       qualifying regions have area >= 0.5 x A_ref.
    3. Gate 1: sum of D1 peaks inside qualifying regions > number of
       non-flagged masks (and at least one qualifying region).
    4. Gate ON : count = non-flagged masks + sum over qualifying regions of
                 max(1, K3 watershed segments >= 0.25 x A_ref).
       Gate OFF: count = number of R4 masks.

    Returns (count, gate_on).
    """
    from .gpv import points_per_mask, merged_mask_flags

    masks = np.asarray(r4_masks, dtype=bool)
    if masks.shape[0] == 0:
        return 0, False
    peaks = points_per_mask(masks, d1_peaks_xy)
    flagged = merged_mask_flags(masks, peaks, "A")
    a_ref = reference_area(masks, flagged)
    n_nonflag = int((~flagged).sum())
    regions = [r for _, r in build_regions(masks, r4_scores, flagged)
               if r.sum() > 0 and r.sum() >= 0.5 * a_ref]
    peaks_in = sum(k1_peaks(r, d1_peaks_xy) for r in regions)
    if not (regions and peaks_in > n_nonflag):
        return int(masks.shape[0]), False
    min_seg = 0.25 * a_ref
    total = n_nonflag + sum(max(1, k3_watershed_depth_gradient(r, depth, min_seg))
                            for r in regions)
    return int(total), True


# ============================================================
# SG with K replacing G2 (B3RK)
# ============================================================

def sg_k_count(b0_masks, a3c_mask, depth, sg_cfg, estimator):
    """
    SG_v1.1 with its G2 multiplicity replaced by a K estimator.

    Unchanged from SG_v1.1: B0 anchor, C1 recovery, split candidates,
    Stage-1 / Stage-2 gate (computed on SG_v1 G2 extras), refinable set.

    For each refinable recovery component when Stage 2 is ON:
        region = component pixels NOT covered by any B0 mask
                 (same intent as the v1.1 double-counting fix)
        m      = K segments in region (>= 1 if region non-empty, else 0)
        extra  = max(m - 1, 0)          (the component's +1 is already in S)
    A_ref (single-object size) = median B0 mask area; fallback = median
    C1 component area if the image has no B0 masks. K2/K3 keep only
    segments >= 0.25 x A_ref.

    estimator: "K2" (shape watershed) or "K3" (depth-gradient watershed).
    Returns dict with S_count, SG_count, stage2_on, extra_units, n_refinable.
    """
    from . import sg as sg_mod

    summary, comps = sg_mod.compute_sg_image(b0_masks, a3c_mask, depth, sg_cfg)
    s_count = summary["S_count"]
    out = {"S_count": s_count, "stage2_on": summary["stage2_on"],
           "n_refinable": summary["refinable_components"],
           "G2_SG_count": summary["SG_count"]}
    if not summary["stage2_on"]:
        out.update({"SG_count": float(s_count), "extra_units": 0, "a_ref": np.nan})
        return out

    b0_masks = np.asarray(b0_masks, dtype=bool)
    h, w = np.asarray(a3c_mask).shape
    union = b0_masks.any(axis=0) if b0_masks.shape[0] else np.zeros((h, w), bool)
    labels, areas, _, _ = sg_mod.c1_components(a3c_mask)
    if b0_masks.shape[0]:
        a_ref = float(np.median(b0_masks.reshape(b0_masks.shape[0], -1).sum(axis=1)))
    else:
        a_ref = float(np.median(areas)) if areas.size else 1.0
    min_seg = 0.25 * a_ref

    extra = 0
    for c in comps:
        if not c["refinable"]:
            continue
        region = (labels == c["component_id"]) & ~union
        if not region.any():
            continue
        if estimator == "K3":
            m = k3_watershed_depth_gradient(region, depth, min_seg)
        elif estimator == "K2":
            m = k2_watershed_shape(region, min_seg)
        else:
            raise ValueError(estimator)
        extra += max(max(1, int(m)) - 1, 0)

    out.update({"SG_count": float(s_count + sg_cfg.correction_weight * extra),
                "extra_units": int(extra), "a_ref": a_ref})
    return out
