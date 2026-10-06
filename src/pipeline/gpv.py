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
