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
