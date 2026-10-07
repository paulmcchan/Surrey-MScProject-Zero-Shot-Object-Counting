# -*- coding: utf-8 -*-
"""
GPV — Guided Propose-Verify counting with SAM2.

Building blocks for the guided pipeline ablation ladder (B3G):

    Rung 1  retain-all guided masks (End-to-End checkpoint protocol)
    Rung 2  mask deduplication                      <- this module (v1)
    Rung 3  multimask selection                     (later)
    Rung 4  + B0 AMG proposal pool                  (later)
    Rung 5  + semantic verification                 (later)
    Rung 6  + geometric group-mask handling         (later)

All parameters are fixed by convention BEFORE running on dev100
(no tuning): IoU threshold 0.5; group mask = area > 4x median kept area.

Masks are (N, H, W) bool stacks; scores are SAM2 predicted IoU scores.
"""

import numpy as np


# ============================================================
# Bounding boxes (used to skip non-overlapping pairs)
# ============================================================

def mask_bboxes(masks):
    """(N, H, W) bool -> (N, 4) int [y0, x0, y1, x1] (exclusive end).
    Empty masks get an empty box (0, 0, 0, 0)."""
    masks = np.asarray(masks, dtype=bool)
    n = masks.shape[0]
    boxes = np.zeros((n, 4), dtype=np.int64)
    rows = masks.any(axis=2)
    cols = masks.any(axis=1)
    for i in range(n):
        ys = np.flatnonzero(rows[i])
        xs = np.flatnonzero(cols[i])
        if ys.size:
            boxes[i] = (ys[0], xs[0], ys[-1] + 1, xs[-1] + 1)
    return boxes


def _boxes_overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def mask_iou(m1, m2, b1, b2):
    """Exact IoU of two masks, computed on the union of their boxes."""
    if not _boxes_overlap(b1, b2):
        return 0.0
    y0, x0 = min(b1[0], b2[0]), min(b1[1], b2[1])
    y1, x1 = max(b1[2], b2[2]), max(b1[3], b2[3])
    a = m1[y0:y1, x0:x1]
    b = m2[y0:y1, x0:x1]
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter / union)


# ============================================================
# Rung 2 — deduplication
# ============================================================

def score_order(scores):
    """Indices sorted by descending score (ties: lower index first)."""
    scores = np.asarray(scores, dtype=np.float64)
    return np.lexsort((np.arange(scores.size), -scores))


def nms_masks(masks, scores, iou_threshold=0.5):
    """
    Greedy mask NMS: visit masks by descending score; keep a mask unless
    its IoU with an already-kept mask exceeds iou_threshold.

    Returns kept indices (in visiting order).
    """
    masks = np.asarray(masks, dtype=bool)
    if masks.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    boxes = mask_bboxes(masks)
    kept = []
    for i in score_order(scores):
        if not masks[i].any():
            continue
        duplicate = False
        for k in kept:
            if mask_iou(masks[i], masks[k], boxes[i], boxes[k]) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(int(i))
    return np.asarray(kept, dtype=np.int64)


def prompt_suppression(masks, scores, prompts_xy, candidate_idx=None):
    """
    Visit masks by descending score; drop a mask if its OWN prompt point
    lies inside a mask already kept (the object was already segmented).

    candidate_idx: restrict to these indices (e.g. after NMS).
    Returns kept indices.
    """
    masks = np.asarray(masks, dtype=bool)
    n, h, w = masks.shape if masks.ndim == 3 else (0, 0, 0)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    pts = np.asarray(prompts_xy, dtype=np.float64).reshape(-1, 2)
    px = np.clip(np.rint(pts[:, 0]).astype(np.int64), 0, w - 1)
    py = np.clip(np.rint(pts[:, 1]).astype(np.int64), 0, h - 1)

    pool = np.arange(n) if candidate_idx is None else np.asarray(candidate_idx)
    order = pool[score_order(np.asarray(scores)[pool])]

    covered = np.zeros((h, w), dtype=bool)
    kept = []
    for i in order:
        if not masks[i].any():
            continue
        if covered[py[i], px[i]]:
            continue
        kept.append(int(i))
        covered |= masks[i]
    return np.asarray(kept, dtype=np.int64)


# ============================================================
# Diagnostics (GT used for EVALUATION only)
# ============================================================

def gt_mask_incidence(masks, gt_points_xy):
    """
    Returns
    -------
    gt_per_mask   : (N,) number of GT points inside each mask
    masks_per_gt  : (G,) number of masks containing each GT point
    """
    masks = np.asarray(masks, dtype=bool)
    gt = np.asarray(gt_points_xy, dtype=np.float64).reshape(-1, 2)
    if masks.shape[0] == 0 or gt.shape[0] == 0:
        return (np.zeros(masks.shape[0], dtype=np.int64),
                np.zeros(gt.shape[0], dtype=np.int64))
    h, w = masks.shape[1:]
    px = np.clip(np.rint(gt[:, 0]).astype(np.int64), 0, w - 1)
    py = np.clip(np.rint(gt[:, 1]).astype(np.int64), 0, h - 1)
    hits = masks[:, py, px]                       # (N, G)
    return hits.sum(axis=1), hits.sum(axis=0)


def group_mask_flags(masks, factor=4.0):
    """Flag masks with area > factor x median area of the given set."""
    masks = np.asarray(masks, dtype=bool)
    if masks.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    med = np.median(areas)
    return areas > factor * med


# ============================================================
# Rung 4 — proposal pool (guided masks + B0 AMG masks)
# ============================================================

def pool_masks(mask_sets, score_sets, source_names):
    """
    Concatenate several (N_i, H, W) mask stacks into one proposal pool.

    Returns masks (N, H, W), scores (N,), source (N,) array of names.
    """
    stacks, scores, sources = [], [], []
    shape = None
    for m, s, name in zip(mask_sets, score_sets, source_names):
        m = np.asarray(m, dtype=bool)
        if m.shape[0] == 0:
            continue
        shape = m.shape[1:] if shape is None else shape
        assert m.shape[1:] == shape, "mask shapes differ within one image"
        stacks.append(m)
        scores.append(np.asarray(s, dtype=np.float64))
        sources.append(np.full(m.shape[0], name, dtype=object))
    if not stacks:
        return (np.zeros((0, 1, 1), dtype=bool), np.zeros(0),
                np.zeros(0, dtype=object))
    return np.concatenate(stacks), np.concatenate(scores), np.concatenate(sources)


# ============================================================
# Rung 5 — semantic verification (SAN target vs background)
# ============================================================

def mask_semantic_evidence(masks, target_score, background_score):
    """
    Per-mask SAN evidence.

    Returns dict of (N,) arrays:
        mean_target, mean_background : mean scores inside the mask
        target_pixel_fraction        : share of mask pixels where
                                       target > background
    """
    masks = np.asarray(masks, dtype=bool)
    n = masks.shape[0]
    if n == 0:
        z = np.zeros(0)
        return {"mean_target": z, "mean_background": z, "target_pixel_fraction": z}
    flat = masks.reshape(n, -1).astype(np.float32)
    area = np.maximum(flat.sum(axis=1), 1.0)
    t = np.asarray(target_score, dtype=np.float32).ravel()
    b = np.asarray(background_score, dtype=np.float32).ravel()
    win = (t > b).astype(np.float32)
    return {
        "mean_target": (flat @ t) / area,
        "mean_background": (flat @ b) / area,
        "target_pixel_fraction": (flat @ win) / area,
    }


def verify_v1(evidence):
    """V1: mask-level SAN decision — mean target > mean background."""
    return evidence["mean_target"] > evidence["mean_background"]


def verify_v2(evidence):
    """V2: pixel majority — at least half the mask pixels prefer target."""
    return evidence["target_pixel_fraction"] >= 0.5


# ============================================================
# Rung 6a — group-mask multiplicity by area ratio (no depth)
# ============================================================

def area_ratio_multiplicity(masks, factor=4.0):
    """
    Masks with area > factor x median area (of this set) are group masks;
    each contributes max(1, round(area / median_area)). Others contribute 1.

    Returns multiplicity (N,) int, group flags (N,) bool, median area.
    """
    masks = np.asarray(masks, dtype=bool)
    n = masks.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool), 0.0
    areas = masks.reshape(n, -1).sum(axis=1).astype(np.float64)
    med = float(np.median(areas))
    group = areas > factor * med
    mult = np.ones(n, dtype=np.int64)
    if med > 0:
        mult[group] = np.maximum(1, np.rint(areas[group] / med)).astype(np.int64)
    return mult, group, med


# ============================================================
# Rung 3 — multimask selection
# ============================================================

def select_multimask(masks3, scores3, point_xy, score_threshold=0.8):
    """
    Choose one of SAM2's multimask outputs for a single positive point.

    Rule (pre-declared): among masks with predicted IoU >= score_threshold
    (SAM2 AMG's default pred_iou_thresh) that contain the prompt point,
    take the SMALLEST; if none qualifies, take the highest-scoring mask.

    Returns (mask (H, W) bool, score float, rule_used str).
    """
    masks3 = np.asarray(masks3, dtype=bool)
    scores3 = np.asarray(scores3, dtype=np.float64)
    h, w = masks3.shape[1:]
    px = int(np.clip(np.rint(point_xy[0]), 0, w - 1))
    py = int(np.clip(np.rint(point_xy[1]), 0, h - 1))
    areas = masks3.reshape(masks3.shape[0], -1).sum(axis=1)
    ok = (scores3 >= score_threshold) & masks3[:, py, px] & (areas > 0)
    if ok.any():
        cand = np.flatnonzero(ok)
        j = cand[np.argmin(areas[cand])]
        return masks3[j], float(scores3[j]), "smallest_confident"
    j = int(np.argmax(scores3))
    return masks3[j], float(scores3[j]), "fallback_top_score"


# ============================================================
# Rung 6b — seeds inside group masks, re-prompting, resolution
# ============================================================

def object_radius(kept_masks):
    """Typical object radius from the median kept-mask area: sqrt(A/pi), >= 2."""
    kept_masks = np.asarray(kept_masks, dtype=bool)
    if kept_masks.shape[0] == 0:
        return 2
    med = float(np.median(kept_masks.reshape(kept_masks.shape[0], -1).sum(axis=1)))
    return int(max(2, round(np.sqrt(med / np.pi))))


def snap_points_to_mask(points_xy, mask):
    """Move points lying outside `mask` to the nearest mask pixel."""
    from scipy import ndimage

    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0 or not mask.any():
        return np.zeros((0, 2))
    h, w = mask.shape
    _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
    out = []
    for x, y in pts:
        cx = int(np.clip(np.rint(x), 0, w - 1))
        cy = int(np.clip(np.rint(y), 0, h - 1))
        if mask[cy, cx]:
            out.append((float(cx), float(cy)))
        else:
            out.append((float(ix[cy, cx]), float(iy[cy, cx])))
    return np.asarray(out, dtype=np.float64)


def _peaks(score_map, region, radius, cap):
    from skimage.feature import peak_local_max

    if not region.any():
        return np.zeros((0, 2))
    work = np.where(region, score_map, score_map[region].min() - 1.0)
    coords = peak_local_max(
        work, min_distance=radius, labels=region.astype(np.int32),
        exclude_border=False, num_peaks=cap,
    )
    if coords.size == 0:  # flat region: use its centroid
        ys, xs = np.nonzero(region)
        return snap_points_to_mask([[xs.mean(), ys.mean()]], region)
    return coords[:, ::-1].astype(np.float64)  # (row, col) -> (x, y)


def seeds_distance_transform(region, radius, cap=200):
    """Non-depth control: peaks of the region's Euclidean distance transform."""
    from scipy import ndimage
    return _peaks(ndimage.distance_transform_edt(region), region, radius, cap)


def seeds_semantic(region, target_score, radius, cap=200):
    """Semantic-only control: SAN target-score peaks inside the region."""
    return _peaks(np.asarray(target_score, dtype=np.float64), region, radius, cap)


def seeds_depth(region, depth, sg_cfg, cap=200):
    """
    Geometric prior: SG's G2 depth split of the region (k-means k=2 on
    depth + 4-connected fragments + G2 retention); seeds = retained
    fragment centroids, snapped into the region.
    """
    from scipy import ndimage
    from .sg import g2_split_component

    if not region.any():
        return np.zeros((0, 2))
    sl = ndimage.find_objects(region.astype(np.int32))[0]
    res = g2_split_component(region[sl], np.asarray(depth)[sl],
                             (sl[0].start, sl[1].start), sg_cfg)
    pts = res["retained_centroids"][:cap]
    return snap_points_to_mask(pts, region)


def resolve_groups(base_masks, base_scores, candidate_idx, new_masks, new_scores,
                   new_group, iou_threshold=0.5):
    """
    Replace candidate group masks by their re-prompted sub-masks.

    Pool = non-candidate base masks + all new masks; IoU-NMS by score.
    A candidate group mask is REPLACED by its surviving new masks if at
    least 2 survive; otherwise the group mask itself counts once (and its
    single survivor, if any, is discarded to avoid double counting).

    Returns dict(count, replaced_groups, kept_groups, added_units).
    """
    base_masks = np.asarray(base_masks, dtype=bool)
    cand = set(int(i) for i in candidate_idx)
    non_cand = [i for i in range(base_masks.shape[0]) if i not in cand]

    stacks = [base_masks[non_cand]] if non_cand else []
    scores = [np.asarray(base_scores, dtype=np.float64)[non_cand]] if non_cand else []
    tags = [np.full(len(non_cand), -1)] if non_cand else []
    if len(new_masks):
        stacks.append(np.asarray(new_masks, dtype=bool))
        scores.append(np.asarray(new_scores, dtype=np.float64))
        tags.append(np.asarray(new_group, dtype=np.int64))
    if not stacks:
        return {"count": len(cand), "replaced_groups": 0,
                "kept_groups": len(cand), "added_units": 0}

    pool = np.concatenate(stacks)
    pool_scores = np.concatenate(scores)
    pool_tags = np.concatenate(tags)
    keep = nms_masks(pool, pool_scores, iou_threshold)
    kept_tags = pool_tags[keep]

    count = int((kept_tags == -1).sum())
    replaced = kept = added = 0
    for gidx in cand:
        survivors = int((kept_tags == gidx).sum())
        if survivors >= 2:
            count += survivors
            replaced += 1
            added += survivors - 1
        else:
            count += 1
            kept += 1
    return {"count": count, "replaced_groups": replaced,
            "kept_groups": kept, "added_units": added}


# ============================================================
# Shared builders (reused by B3GK)
# ============================================================

def build_r4(guided_masks, guided_scores, b0_masks, b0_scores, iou_threshold=0.5):
    """
    R4 (B3G Section 4): guided D1-kept masks U B0 AMG masks,
    IoU-NMS by predicted IoU. Returns (kept_masks, kept_scores, kept_source).
    """
    pool, scores, source = pool_masks([guided_masks, b0_masks],
                                      [guided_scores, b0_scores],
                                      ["guided", "b0"])
    if pool.shape[0] == 0 or pool.shape[1:] == (1, 1):
        return pool[:0], scores[:0], source[:0]
    keep = nms_masks(pool, scores, iou_threshold)
    return pool[keep], scores[keep], source[keep]


def d1_peaks(target_score, min_distance_frac=0.020, score_threshold=0.30):
    """
    Frozen D1 reference points (B1 D notebook, Step 11D.8):
    per-image min-max normalised SAN target score; peak_local_max with
    min_distance = round(0.020 x short side) px, threshold_abs = 0.30,
    exclude_border=False. Returns (K, 2) float [x, y].
    """
    from skimage.feature import peak_local_max

    s = np.asarray(target_score, dtype=np.float32)
    lo, hi = float(s.min()), float(s.max())
    norm = np.zeros_like(s) if hi - lo < 1e-12 else (s - lo) / (hi - lo)
    h, w = norm.shape
    md = max(1, int(round(min_distance_frac * min(h, w))))
    yx = peak_local_max(norm, min_distance=md, threshold_abs=score_threshold,
                        exclude_border=False)
    return yx[:, ::-1].astype(np.float64) if len(yx) else np.zeros((0, 2))


def points_per_mask(masks, points_xy):
    """Number of points inside each mask (np.rint + clip rule)."""
    masks = np.asarray(masks, dtype=bool)
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if masks.shape[0] == 0 or pts.shape[0] == 0:
        return np.zeros(masks.shape[0], dtype=np.int64)
    h, w = masks.shape[1:]
    px = np.clip(np.rint(pts[:, 0]).astype(np.int64), 0, w - 1)
    py = np.clip(np.rint(pts[:, 1]).astype(np.int64), 0, h - 1)
    return masks[:, py, px].sum(axis=1)


def merged_mask_flags(masks, peaks_per_mask, rule, factor=4.0, min_single_ref=3):
    """
    GT-free merged-mask rules (B3GK Step 3):
        A : area > factor x median(all kept areas)        AND peaks >= 2
        B : peaks >= 2
        C : peaks >= 3
        D : area > factor x median(areas of 1-peak masks) AND peaks >= 2
            (falls back to the all-mask median if fewer than
             min_single_ref one-peak masks exist)
    """
    masks = np.asarray(masks, dtype=bool)
    n = masks.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    peaks = np.asarray(peaks_per_mask)
    areas = masks.reshape(n, -1).sum(axis=1).astype(np.float64)
    if rule == "B":
        return peaks >= 2
    if rule == "C":
        return peaks >= 3
    if rule == "A":
        return (areas > factor * np.median(areas)) & (peaks >= 2)
    if rule == "D":
        single = areas[peaks == 1]
        ref = np.median(single) if single.size >= min_single_ref else np.median(areas)
        return (areas > factor * ref) & (peaks >= 2)
    raise ValueError(rule)
