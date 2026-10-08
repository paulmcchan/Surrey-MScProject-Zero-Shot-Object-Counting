# -*- coding: utf-8 -*-
"""
Hybrid SG / GPV architectures (B3H).

H1b_v1 (frozen B4 candidate) — GPV-anchored SG, local recovery:
    anchor  = GPV_R4_v1 kept masks minus rule-A merged masks
    SG_v1.1 unchanged with the anchor in place of B0, except that C1
    recovery (and hence G2) applies only to components whose centroid lies
    inside the union of the removed merged masks.

H2_v1 (pre-declared B4 DIAGNOSTIC, not eligible for selection) — routing:
    SG_v1.1 count if SG_v1.1's Stage-2 gate is ON and GPV_K_v1's Gate 1 is
    OFF; otherwise GPV_K_v1 count.
"""

import numpy as np

from . import gpv
from . import sg as sg_mod


def h1b_count(r4_masks, d1_peaks_xy, a3c_mask, depth, sg_cfg):
    """
    Returns (count, info) for H1b_v1.

    r4_masks    : (N, H, W) bool  GPV_R4_v1 kept masks
    d1_peaks_xy : (K, 2)          frozen D1 peaks (for rule A)
    a3c_mask    : (H, W) bool     A3c semantic mask
    depth       : (H, W) float32  Marigold depth
    sg_cfg      : SGConfig        SG_v1.1 frozen config (variant v1.1)
    """
    masks = np.asarray(r4_masks, dtype=bool)
    a3c_mask = np.asarray(a3c_mask, dtype=bool)
    if masks.shape[0]:
        peaks = gpv.points_per_mask(masks, d1_peaks_xy)
        merged = gpv.merged_mask_flags(masks, peaks, "A")
    else:
        merged = np.zeros(0, dtype=bool)
    anchor = masks[~merged] if masks.shape[0] else np.zeros((0,) + a3c_mask.shape, bool)
    merged_union = (masks[merged].any(axis=0) if merged.any()
                    else np.zeros(a3c_mask.shape, dtype=bool))
    summary, _ = sg_mod.compute_sg_image(anchor, a3c_mask, depth, sg_cfg,
                                         recovery_region=merged_union)
    info = {"n_r4": int(masks.shape[0]), "n_merged": int(merged.sum()),
            "n_anchor": int(anchor.shape[0]), "recovery": summary["C1_recovery"],
            "extra_units": summary["G_effective"], "stage2_on": summary["stage2_on"]}
    return float(summary["SG_count"]), info


def h2_route(sg_count, sg_stage2_on, gpv_k_count, gpv_gate1_on):
    """H2_v1 routing (diagnostic): SG where SG gate ON and Gate 1 OFF, else GPV_K."""
    return float(sg_count) if (bool(sg_stage2_on) and not bool(gpv_gate1_on)) else float(gpv_k_count)
