# -*- coding: utf-8 -*-
"""
SG counting logic — SG_v1 (frozen) and SG_v1.1 (double-counting fix).

SG = B0 + C1 recovery + adaptive G2 geometry

Faithfully re-implements the frozen behaviour developed in:
    ABC  (A3c -> C1 components + centroids)
    B2   (C1 recovery = centroid outside every B0 mask)
    M1   (Step 7A candidates, Step 7B k-means depth split,
          Step 7C/7D G2 anchor + satellite retention)
    B3   (refinable components, two-stage gate, SG count)

IMPORTANT — Stage-1 gate (see B3 erratum, 5 Oct 2026)
-----------------------------------------------------
Stage 1 (candidate_area_fraction >= 0.85) is part of SG_v1. It acts through
the refinable-component definition:

    refinable = C1 recovery AND split candidate AND stage-1 ON

so Stage 2 can only activate in images that pass Stage 1.

Variants
--------
"v1"   : frozen SG_v1 behaviour (reproduces B3 Step 3F SG_count).
"v1.1" : identical gates; a refinable component's extra units count only
         retained G2 fragments whose rounded centroid lies OUTSIDE every
         B0 mask:  extra = max(n_uncovered_retained - 1, 0).
         The gate decision is computed on the SG_v1 extra units, so the fix
         changes only the counted units, never the gate.
"""

import json
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata
from sklearn.cluster import KMeans


# 4-connectivity, as used in ABC (ndimage.label default) and M1 Step 7B
STRUCTURE_4 = ndimage.generate_binary_structure(rank=2, connectivity=1)

VALID_VARIANTS = ("v1", "v1.1")


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class SGConfig:
    candidate_quantile: float = 0.75
    satellite_parent_fraction: float = 0.001
    stage1_threshold: float = 0.85
    stage2_threshold: float = 5.0
    correction_weight: float = 1.0
    kmeans_k: int = 2
    kmeans_random_state: int = 0
    kmeans_n_init: int = 10
    variant: str = "v1"

    def __post_init__(self):
        assert self.variant in VALID_VARIANTS, self.variant

    @classmethod
    def from_frozen_json(cls, path, variant="v1"):
        """Build from the (erratum-corrected) SG_v1_frozen_config.json."""
        with open(path, "r") as f:
            cfg = json.load(f)

        assert cfg["architecture_id"] == "SG_v1"
        g = cfg["geometry"]

        assert g["variant"] == "G2"
        assert g["correction_cap"] is None, "Cap not supported (none frozen)."
        assert g["stage1_gate"] is not None, (
            "stage1_gate is null — apply the B3 erratum first."
        )

        return cls(
            candidate_quantile=float(g["candidate_component_quantile"]),
            satellite_parent_fraction=float(
                g["satellite_parent_area_fraction_threshold"]
            ),
            stage1_threshold=float(g["stage1_gate"]["threshold"]),
            stage2_threshold=float(g["gate_threshold"]),
            correction_weight=float(g["correction_weight"]),
            variant=variant,
        )

    def to_dict(self):
        return asdict(self)


# ============================================================
# C1 components (ABC)
# ============================================================

def c1_components(a3c_mask):
    """
    A3c binary mask -> 4-connected components with ABC-identical centroids.

    Component IDs follow scipy.ndimage.label order (1..n), so
    component_id = ABC point_index + 1 (B3 Step 2A identity).

    Returns
    -------
    labels    : (H, W) int32 label map (0 = background)
    areas     : (n,) int64
    centroids : (n, 2) float64 [x, y]  (mean pixel coordinate, as ABC)
    slices    : list of bounding-box slices (len n)
    """
    mask = np.asarray(a3c_mask).astype(bool)
    labels, n = ndimage.label(mask, structure=STRUCTURE_4)
    slices = ndimage.find_objects(labels)

    areas = np.zeros(n, dtype=np.int64)
    centroids = np.zeros((n, 2), dtype=np.float64)

    for i, sl in enumerate(slices):
        comp_id = i + 1
        ys, xs = np.nonzero(labels[sl] == comp_id)
        # integer offsets keep coordinates exact before averaging
        xs = xs + sl[1].start
        ys = ys + sl[0].start
        areas[i] = xs.size
        centroids[i] = (float(xs.mean()), float(ys.mean()))

    return labels, areas, centroids, slices


# ============================================================
# B0 coverage (B2 rule: np.rint + clip -> mask pixel)
# ============================================================

def points_covered(points_xy, coverage_mask):
    """True where the rounded, clipped point lies on coverage_mask."""
    points_xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if points_xy.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    h, w = coverage_mask.shape
    px = np.clip(np.rint(points_xy[:, 0]).astype(np.int64), 0, w - 1)
    py = np.clip(np.rint(points_xy[:, 1]).astype(np.int64), 0, h - 1)
    return coverage_mask[py, px].astype(bool)


def b0_union(b0_masks, height, width):
    """Union of all B0 masks; a point is 'outside every B0 mask' iff not in it."""
    b0_masks = np.asarray(b0_masks, dtype=bool)
    if b0_masks.shape[0] == 0:
        return np.zeros((height, width), dtype=bool)
    return b0_masks.any(axis=0)


# ============================================================
# Split candidates (M1 Step 7A)
# ============================================================

def split_candidates(areas, quantile):
    """
    Within-image area percentile rank (pandas rank(method='average',
    pct=True) equivalent) >= quantile.
    """
    areas = np.asarray(areas)
    if areas.size == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    pct = rankdata(areas, method="average") / areas.size
    return pct >= quantile, pct


# ============================================================
# G2 depth split for one candidate component (M1 Step 7B-7D)
# ============================================================

def g2_split_component(comp_mask_crop, depth_crop, offset_yx, cfg):
    """
    Split one candidate component by 1-D k-means on depth, then
    4-connected components within each depth cluster; retain G2 fragments.

    Operates on the component's bounding-box crop (labels and pixel order
    are identical to full-image processing).

    Returns dict with
        degenerate         : bool (fewer than k unique depth values)
        raw_fragment_count : int
        g2_count           : int   (retained fragments; 1 if degenerate)
        retained_centroids : (m, 2) float64 [x, y] in full-image coords
    """
    comp_area = int(comp_mask_crop.sum())
    values = depth_crop[comp_mask_crop].reshape(-1, 1)

    # Degenerate geometry: preserve unchanged (M1 Step 7B rule)
    if np.unique(values).size < cfg.kmeans_k:
        ys, xs = np.nonzero(comp_mask_crop)
        c = np.array(
            [[xs.mean() + offset_yx[1], ys.mean() + offset_yx[0]]],
            dtype=np.float64,
        )
        return {
            "degenerate": True,
            "raw_fragment_count": 1,
            "g2_count": 1,
            "retained_centroids": c,
        }

    kmeans = KMeans(
        n_clusters=cfg.kmeans_k,
        random_state=cfg.kmeans_random_state,
        n_init=cfg.kmeans_n_init,
    )
    assignment = kmeans.fit_predict(values)

    cluster_map = np.full(comp_mask_crop.shape, -1, dtype=np.int8)
    cluster_map[comp_mask_crop] = assignment.astype(np.int8)

    # Fragments in proposal-ID order: cluster 0 first, then raster order
    fragments = []  # (proposal_id, cluster, area, centroid_xy)
    proposal_id = 0
    for cluster_id in range(cfg.kmeans_k):
        cluster_mask = comp_mask_crop & (cluster_map == cluster_id)
        spatial, n_spatial = ndimage.label(cluster_mask, structure=STRUCTURE_4)
        for s in range(1, n_spatial + 1):
            ys, xs = np.nonzero(spatial == s)
            fragments.append((
                proposal_id,
                cluster_id,
                int(xs.size),
                (
                    float((xs + offset_yx[1]).mean()),
                    float((ys + offset_yx[0]).mean()),
                ),
            ))
            proposal_id += 1

    # Anchor = largest fragment per depth cluster (tie -> lowest proposal ID)
    retained = []
    for cluster_id in range(cfg.kmeans_k):
        frags = [f for f in fragments if f[1] == cluster_id]
        if not frags:
            continue
        frags.sort(key=lambda f: (-f[2], f[0]))
        retained.append(frags[0])  # anchor
        for f in frags[1:]:        # satellites
            if f[2] / comp_area >= cfg.satellite_parent_fraction:
                retained.append(f)

    retained.sort(key=lambda f: f[0])
    return {
        "degenerate": False,
        "raw_fragment_count": len(fragments),
        "g2_count": len(retained),
        "retained_centroids": np.array(
            [f[3] for f in retained], dtype=np.float64
        ).reshape(-1, 2),
    }


# ============================================================
# Full SG count for one image
# ============================================================

def compute_sg_image(b0_masks, a3c_mask, depth, cfg):
    """
    Compute S and SG (variant per cfg.variant) for one image.

    Parameters
    ----------
    b0_masks : (N, H, W) bool   frozen B0 AMG masks
    a3c_mask : (H, W) bool      A3c semantic mask
    depth    : (H, W) float32   Marigold relative depth
    cfg      : SGConfig

    Returns
    -------
    summary    : dict of image-level counts, features and gate states
    components : list of per-C1-component dicts (for diagnostics)
    """
    a3c_mask = np.asarray(a3c_mask).astype(bool)
    h, w = a3c_mask.shape
    assert depth.shape == (h, w), "depth / A3c shape mismatch"

    b0_masks = np.asarray(b0_masks, dtype=bool)
    assert b0_masks.shape[1:] == (h, w) or b0_masks.shape[0] == 0
    b0_count = int(b0_masks.shape[0])
    union = b0_union(b0_masks, h, w)

    # ---- C1 components + recovery -------------------------
    labels, areas, centroids, slices = c1_components(a3c_mask)
    n_comp = int(areas.size)

    covered = points_covered(centroids, union)
    recovery = ~covered

    # ---- Split candidates + Stage-1 feature ---------------
    candidate, area_pct = split_candidates(areas, cfg.candidate_quantile)
    total_area = int(areas.sum())
    candidate_area_fraction = (
        float(areas[candidate].sum()) / total_area if total_area > 0 else 0.0
    )
    stage1_on = candidate_area_fraction >= cfg.stage1_threshold

    # ---- G2 for every candidate (diagnostic completeness) --
    g2_count = np.ones(n_comp, dtype=np.int64)
    raw_frag = np.ones(n_comp, dtype=np.int64)
    uncovered_retained = np.ones(n_comp, dtype=np.int64)
    degenerate = np.zeros(n_comp, dtype=bool)

    for i in np.flatnonzero(candidate):
        sl = slices[i]
        comp_crop = labels[sl] == (i + 1)
        res = g2_split_component(
            comp_crop,
            depth[sl],
            (sl[0].start, sl[1].start),
            cfg,
        )
        g2_count[i] = res["g2_count"]
        raw_frag[i] = res["raw_fragment_count"]
        degenerate[i] = res["degenerate"]
        uncovered_retained[i] = int(
            (~points_covered(res["retained_centroids"], union)).sum()
        )

    extra_v1 = g2_count - 1
    extra_v11 = np.maximum(uncovered_retained - 1, 0)

    # ---- Refinable + Stage-2 gate (always on SG_v1 extras) --
    refinable = recovery & candidate & stage1_on
    n_refinable = int(refinable.sum())
    extra_refinable_v1 = int(extra_v1[refinable].sum())
    stage2_feature = extra_refinable_v1 / max(n_refinable, 1)
    stage2_on = stage2_feature >= cfg.stage2_threshold

    counted_extra = extra_v1 if cfg.variant == "v1" else extra_v11
    g_effective = (
        cfg.correction_weight * float(counted_extra[refinable].sum())
        if stage2_on
        else 0.0
    )

    n_recovery = int(recovery.sum())
    s_count = b0_count + n_recovery
    sg_count = s_count + g_effective

    summary = {
        "B0_count": b0_count,
        "C1_components": n_comp,
        "C1_recovery": n_recovery,
        "S_count": s_count,
        "split_candidates": int(candidate.sum()),
        "degenerate_candidates": int(degenerate[candidate].sum()),
        "raw_subcomponents": int(raw_frag.sum()),
        "candidate_area_fraction": candidate_area_fraction,
        "stage1_on": bool(stage1_on),
        "refinable_components": n_refinable,
        "G2_extra_refinable_v1": extra_refinable_v1,
        "G2_extra_refinable_v11": int(extra_v11[refinable].sum()),
        "stage2_feature": float(stage2_feature),
        "stage2_on": bool(stage2_on),
        "G_effective": float(g_effective),
        "SG_count": float(sg_count),
        "variant": cfg.variant,
    }

    components = [
        {
            "component_id": i + 1,
            "area": int(areas[i]),
            "centroid_x": float(centroids[i, 0]),
            "centroid_y": float(centroids[i, 1]),
            "area_percentile": float(area_pct[i]),
            "covered_by_B0": bool(covered[i]),
            "recovery": bool(recovery[i]),
            "split_candidate": bool(candidate[i]),
            "degenerate": bool(degenerate[i]),
            "raw_fragments": int(raw_frag[i]),
            "G2_count": int(g2_count[i]),
            "uncovered_retained": int(uncovered_retained[i]),
            "extra_v1": int(extra_v1[i]),
            "extra_v11": int(extra_v11[i]),
            "refinable": bool(refinable[i]),
        }
        for i in range(n_comp)
    ]

    return summary, components
