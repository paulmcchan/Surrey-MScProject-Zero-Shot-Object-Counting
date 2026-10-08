# -*- coding: utf-8 -*-
"""
GPV guided prompts — the End-to-End checkpoint system `dense_recovery`.

dense_recovery = F1 fusion of
    C1          : A3c 4-connected components, centroid (mean pixel) per region
    D6B9-strict : SAN intermediate-feature peaks (B1 D notebook, Step 12D.9)
with association tolerance tau = 0.050 (B1 E notebook, Step 13D.2b).

Ported faithfully from the B1 D and E notebooks:
  D6B9-strict
    query maps        T_q = P(t|q) * sigmoid(M_q)           (SAN mask logits)
    query admission   P(t|q) >= 0.10, greedy redundancy filter
                      (cosine similarity < 0.90 to every kept query,
                       queries ranked by P(t|q))
    per query         normalise by max; Gaussian modulation sigma 0.75
                      (mode nearest, rescaled to original max); renormalise;
                      peak_local_max(min_distance 1, threshold_abs 0.70,
                      exclude_border False); absolute floor: raw T_q >= 0.20;
                      DARK-style Taylor refinement on the working map
                      (log, central differences, condition <= 1e6,
                       |offset| <= 1 cell, else keep discrete peak)
    to image coords   x = (fx + 0.5) * W / feature_w,  y likewise
    score             P(t|q) * working_map[fy, fx]
    consolidation     greedy by descending score; drop points within
                      merge radius = diagonal of one feature cell
  F1 fusion (tau = 0.050)
    a D point is associated with the C1 region containing it (rint + clip);
    otherwise with the nearest region if its Euclidean distance to that
    region / short side <= tau.
    points = all D points + C1 centroids of regions with no associated D point.
"""

import numpy as np

D6B9_QUERY_PROB_THRESHOLD = 0.10
D6B9_SIMILARITY_THRESHOLD = 0.90
D6B9_ABSOLUTE_THRESHOLD = 0.20
D6B9_GAUSSIAN_SIGMA = 0.75
D6B9_RELATIVE_PEAK_THRESHOLD = 0.70
D6B9_FEATURE_MIN_DISTANCE = 1
F1_TAU = 0.050


# ============================================================
# D6B9-strict helpers (verbatim logic from the D notebook)
# ============================================================

def gaussian_modulate(activation, sigma):
    from scipy.ndimage import gaussian_filter

    activation = np.asarray(activation, dtype=np.float64)
    original_max = float(np.max(activation))
    smoothed = gaussian_filter(activation, sigma=float(sigma), mode="nearest")
    smoothed_max = float(np.max(smoothed))
    if original_max > 0 and smoothed_max > 0:
        smoothed = smoothed * original_max / smoothed_max
    return smoothed


def taylor_refine_peak(activation, peak_yx, eps=1e-10, max_offset=1.0,
                       condition_limit=1e6):
    activation = np.asarray(activation, dtype=np.float64)
    y, x = int(peak_yx[0]), int(peak_yx[1])
    H, W = activation.shape
    result = {"refined_y": float(y), "refined_x": float(x), "valid": False}
    if y <= 0 or y >= H - 1 or x <= 0 or x >= W - 1:
        return result
    local = np.maximum(activation[y - 1:y + 2, x - 1:x + 2], eps)
    L = np.log(local)
    dx = 0.5 * (L[1, 2] - L[1, 0])
    dy = 0.5 * (L[2, 1] - L[0, 1])
    gradient = np.array([dx, dy], dtype=np.float64)
    dxx = L[1, 2] - 2.0 * L[1, 1] + L[1, 0]
    dyy = L[2, 1] - 2.0 * L[1, 1] + L[0, 1]
    dxy = 0.25 * (L[2, 2] - L[2, 0] - L[0, 2] + L[0, 0])
    hessian = np.array([[dxx, dxy], [dxy, dyy]], dtype=np.float64)
    if not (np.all(np.isfinite(gradient)) and np.all(np.isfinite(hessian))):
        return result
    try:
        cond = np.linalg.cond(hessian)
    except np.linalg.LinAlgError:
        return result
    if not np.isfinite(cond) or cond > condition_limit:
        return result
    try:
        delta = -np.linalg.solve(hessian, gradient)
    except np.linalg.LinAlgError:
        return result
    delta_x, delta_y = float(delta[0]), float(delta[1])
    if not (np.isfinite(delta_x) and np.isfinite(delta_y)):
        return result
    if abs(delta_x) > max_offset or abs(delta_y) > max_offset:
        return result
    result.update({"refined_x": float(x) + delta_x, "refined_y": float(y) + delta_y,
                   "valid": True})
    return result


def select_diverse_queries(query_maps, target_query_prob, prob_threshold,
                           similarity_threshold):
    import torch

    candidate_indices = torch.where(target_query_prob >= prob_threshold)[0]
    if len(candidate_indices) == 0:
        return [], 0
    candidate_indices = candidate_indices[
        torch.argsort(target_query_prob[candidate_indices], descending=True)]
    maps = query_maps[candidate_indices].reshape(len(candidate_indices), -1).double()
    norms = torch.linalg.vector_norm(maps, dim=1, keepdim=True).clamp_min(1e-12)
    maps_norm = maps / norms
    kept = []
    for pos in range(len(candidate_indices)):
        if not kept:
            kept.append(pos)
            continue
        sims = maps_norm[kept] @ maps_norm[pos]
        if torch.all(sims < similarity_threshold):
            kept.append(pos)
    selected = [int(candidate_indices[p]) for p in kept]
    return selected, len(candidate_indices) - len(selected)


def consolidate_points_greedy(points_xy, scores, merge_radius_px):
    points_xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    scores = np.asarray(scores, dtype=np.float64)
    if len(points_xy) == 0:
        return np.empty((0, 2)), np.empty((0,))
    order = np.argsort(scores)[::-1]
    kept_p, kept_s = [], []
    for idx in order:
        p = points_xy[idx]
        if not kept_p or bool(np.all(np.linalg.norm(np.asarray(kept_p) - p, axis=1)
                                     > merge_radius_px)):
            kept_p.append(p)
            kept_s.append(scores[idx])
    return np.asarray(kept_p, dtype=np.float64), np.asarray(kept_s, dtype=np.float64)


def d6b9_strict_points(mask_preds, target_query_prob, image_h, image_w):
    """
    mask_preds        : SAN pre-sigmoid query mask logits, (Q, h, w) or (1, Q, h, w)
    target_query_prob : SAN target-class query probabilities, (Q,) or (1, Q)
    Returns (K, 2) float64 [x, y] in image coordinates.
    """
    import torch
    from skimage.feature import peak_local_max

    mask_preds = torch.as_tensor(mask_preds).float()
    target_query_prob = torch.as_tensor(target_query_prob).float()
    if mask_preds.ndim == 4:
        mask_preds = mask_preds[0]
    if target_query_prob.ndim == 2:
        target_query_prob = target_query_prob[0]
    query_maps = target_query_prob[:, None, None] * torch.sigmoid(mask_preds)
    feature_h, feature_w = int(query_maps.shape[-2]), int(query_maps.shape[-1])
    merge_radius_px = float(np.sqrt((image_w / feature_w) ** 2 + (image_h / feature_h) ** 2))

    selected, _ = select_diverse_queries(query_maps, target_query_prob,
                                         D6B9_QUERY_PROB_THRESHOLD,
                                         D6B9_SIMILARITY_THRESHOLD)
    pts, scs = [], []
    for q in selected:
        query_prob = float(target_query_prob[q])
        qmap = query_maps[q].cpu().numpy().astype(np.float64)
        qmax = float(qmap.max())
        if qmax <= 0:
            continue
        working = gaussian_modulate(qmap / qmax, sigma=D6B9_GAUSSIAN_SIGMA)
        wmax = float(working.max())
        if wmax > 0:
            working = working / wmax
        peaks_rc = peak_local_max(working, min_distance=D6B9_FEATURE_MIN_DISTANCE,
                                  threshold_abs=D6B9_RELATIVE_PEAK_THRESHOLD,
                                  exclude_border=False)
        for fy, fx in peaks_rc:
            if float(qmap[fy, fx]) < D6B9_ABSOLUTE_THRESHOLD:
                continue
            rx, ry = float(fx), float(fy)
            r = taylor_refine_peak(working, [int(fy), int(fx)])
            if r["valid"]:
                rx, ry = r["refined_x"], r["refined_y"]
            pts.append([(rx + 0.5) * image_w / feature_w, (ry + 0.5) * image_h / feature_h])
            scs.append(query_prob * float(working[int(fy), int(fx)]))
    points, _ = consolidate_points_greedy(pts, scs, merge_radius_px)
    return points


# ============================================================
# F1 fusion (E notebook, Step 13D.2b)
# ============================================================

def f1_fusion(c1_labels, c1_centroids_xy, d_points_xy, tau=F1_TAU):
    """
    c1_labels       : (H, W) int, 0 = background, regions 1..n
    c1_centroids_xy : (n, 2) [x, y], row i = region i + 1
    d_points_xy     : (K, 2) [x, y]
    Returns (points (M, 2), info dict).
    """
    from scipy.ndimage import distance_transform_edt

    labels = np.asarray(c1_labels)
    h, w = labels.shape
    short_side = float(min(h, w))
    n_regions = int(labels.max())
    d_pts = np.asarray(d_points_xy, dtype=np.float64).reshape(-1, 2)
    cents = np.asarray(c1_centroids_xy, dtype=np.float64).reshape(-1, 2)
    assert cents.shape[0] == n_regions

    associated_regions = set()
    if n_regions and len(d_pts):
        px = np.clip(np.rint(d_pts[:, 0]).astype(int), 0, w - 1)
        py = np.clip(np.rint(d_pts[:, 1]).astype(int), 0, h - 1)
        containing = labels[py, px]
        outside = np.flatnonzero(containing == 0)
        for k in np.flatnonzero(containing > 0):
            associated_regions.add(int(containing[k]))
        if len(outside):
            dist = np.stack([distance_transform_edt(labels != rid)[py[outside], px[outside]]
                             for rid in range(1, n_regions + 1)])          # (n, K_out)
            nearest = dist.argmin(axis=0)
            d_near = dist[nearest, np.arange(len(outside))]
            for j in np.flatnonzero(d_near / short_side <= tau):
                associated_regions.add(int(nearest[j]) + 1)

    c_only = [rid for rid in range(1, n_regions + 1) if rid not in associated_regions]
    c_pts = cents[[rid - 1 for rid in c_only]] if c_only else np.zeros((0, 2))
    points = np.concatenate([d_pts, c_pts]) if len(d_pts) or len(c_pts) else np.zeros((0, 2))
    return points, {"n_d": int(len(d_pts)), "n_c_added": int(len(c_pts)),
                    "n_regions": n_regions}


def dense_recovery_points(a3c_mask, mask_preds, target_query_prob):
    """Full dense_recovery prompt set for one image."""
    from .sg import c1_components

    a3c_mask = np.asarray(a3c_mask, dtype=bool)
    h, w = a3c_mask.shape
    labels, _, centroids, _ = c1_components(a3c_mask)
    d_pts = d6b9_strict_points(mask_preds, target_query_prob, h, w)
    return f1_fusion(labels, centroids, d_pts)
