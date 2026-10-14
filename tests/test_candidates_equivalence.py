"""Equivalence test: notebook (B3PN/B3AC) definitions vs src/pipeline/candidates.py
on synthetic images. Low stage2_threshold forces H1b Stage 2 ON in some images."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import ndimage
import src.pipeline.gpv as gpv, src.pipeline.hybrid as hybrid, src.pipeline.multiplicity as K
import src.pipeline.sg as sg, src.pipeline.candidates as C

SG_CFG = sg.SGConfig(variant="v1.1", stage1_threshold=0.0, stage2_threshold=0.5)

# ---------- verbatim notebook code (B3PN Step 1.3 / 2.1, B3AC Step 2.1) ----------
PART_FRAC, NEST_FRAC = 0.25, 0.80
def normal_edge_map(nrm):
    n = nrm.astype(np.float32)
    n = n / np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-6)
    edge = np.zeros(n.shape[:2], np.float32)
    for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        edge = np.maximum(edge, 1.0 - (n * np.roll(n, shift=(dy, dx), axis=(0, 1))).sum(axis=2))
    edge[0, :] = edge[1, :]; edge[-1, :] = edge[-2, :]; edge[:, 0] = edge[:, 1]; edge[:, -1] = edge[:, -2]
    return ndimage.gaussian_filter(edge, 1.0)
def k3n_segments(region, edge, min_seg, h_frac=0.1):
    e = edge[region]
    h = h_frac * float(np.percentile(e, 95) - np.percentile(e, 5))
    return 1 if h <= 0 else K._marker_watershed(edge, region, -edge, h, min_seg)
def gpv_k_n_count(r4_m, r4_s, d1, edge):
    masks = np.asarray(r4_m, bool)
    if masks.shape[0] == 0:
        return 0
    flagged = gpv.merged_mask_flags(masks, gpv.points_per_mask(masks, d1), "A")
    a_ref = K.reference_area(masks, flagged)
    n_nonflag = int((~flagged).sum())
    regions = [rg for _, rg in K.build_regions(masks, r4_s, flagged) if rg.sum() > 0 and rg.sum() >= 0.5 * a_ref]
    if not (regions and sum(K.k1_peaks(rg, d1) for rg in regions) > n_nonflag):
        return int(masks.shape[0])
    return n_nonflag + sum(max(1, k3n_segments(rg, edge, 0.25 * a_ref)) for rg in regions)
def h1b_k3n_count(r4_m, d1, a3c, depth, edge):
    base, info = hybrid.h1b_count(r4_m, d1, a3c, depth, SG_CFG)
    if not info["stage2_on"] or r4_m.shape[0] == 0:
        return base
    merged = gpv.merged_mask_flags(r4_m, gpv.points_per_mask(r4_m, d1), "A")
    if not merged.any():
        return base
    anchor = r4_m[~merged]
    areas = anchor.reshape(anchor.shape[0], -1).sum(1) if anchor.shape[0] else np.zeros(0)
    a_ref = float(np.median(areas)) if areas.size else float(np.median(r4_m[merged].reshape(merged.sum(), -1).sum(1)))
    taken = anchor.any(axis=0) if anchor.shape[0] else np.zeros(r4_m.shape[1:], bool)
    total = int(anchor.shape[0])
    for i in np.flatnonzero(merged):
        rg = r4_m[i] & ~taken
        taken |= rg
        if rg.sum() > 0 and rg.sum() >= 0.5 * a_ref:
            total += max(1, k3n_segments(rg, edge, 0.25 * a_ref))
    return float(total)
def object_size(masks, scores, amax, d1):
    n = masks.shape[0]
    areas = masks.reshape(n, -1).sum(1).astype(float)
    n_cls = int(amax.max()) + 1
    dom_t = np.array([np.bincount(amax[m], minlength=n_cls).argmax() == 0 if m.any() else False for m in masks])
    merged = gpv.merged_mask_flags(masks, gpv.points_per_mask(masks, d1), "A")
    verified = dom_t & ~merged & (np.asarray(scores) >= np.median(scores))
    if verified.sum() >= 3:
        return float(np.median(areas[verified]))
    if dom_t.any():
        return float(np.median(areas[dom_t]))
    return float(np.median(areas))
def part_filter_pn(masks, scores, amax, d1):
    n = masks.shape[0]
    if n < 2:
        return masks, np.asarray(scores, float), 0
    s = object_size(masks, scores, amax, d1)
    flat = masks.reshape(n, -1).astype(np.float32)
    areas = flat.sum(1); inter = flat @ flat.T; np.fill_diagonal(inter, 0)
    drop = (areas < PART_FRAC * s) & (inter.max(axis=1) > NEST_FRAC * np.maximum(areas, 1))
    return masks[~drop], np.asarray(scores, float)[~drop], int(drop.sum())
def part_filter_ac(masks, scores, s):
    n = masks.shape[0]
    if n < 2:
        return masks, np.asarray(scores, float)
    flat = masks.reshape(n, -1).astype(np.float32)
    areas = flat.sum(1); inter = flat @ flat.T; np.fill_diagonal(inter, 0)
    drop = (areas < PART_FRAC * s) & (inter.max(axis=1) > NEST_FRAC * np.maximum(areas, 1))
    return masks[~drop], np.asarray(scores, float)[~drop]

def notebook_counts(sets, d1, amax, a3c, dep, edge):
    row = {}
    # B3PN loop
    for b in ["MixA", "LM0"]:
        m, s = sets[b]
        mp, sp, _ = part_filter_pn(m, s, amax, d1)
        row[f"GPV_R4+P[{b}]"] = int(mp.shape[0])
        row[f"GPV_K-n+P[{b}]"] = gpv_k_n_count(mp, sp, d1, edge)
        row[f"H1b-K3n+P[{b}]"] = h1b_k3n_count(mp, d1, a3c, dep, edge)
    # B3AC loop (H1b+P)
    m, s = sets["MixA"]
    if m.shape[0] == 0:
        row["H1b+P[MixA]_ac"] = 0
    else:
        mp, sp = part_filter_ac(m, s, object_size(m, s, amax, d1))
        row["H1b+P[MixA]_ac"] = float(hybrid.h1b_count(mp, d1, a3c, dep, SG_CFG)[0])
    return row

# ---------- synthetic scenes ----------
def scene(rng, H=96, W=128):
    n_obj = rng.integers(0, 25)
    yy, xx = np.mgrid[:H, :W]
    objs, cents = [], []
    for _ in range(n_obj):
        cy, cx, r = rng.integers(5, H-5), rng.integers(5, W-5), rng.integers(3, 8)
        objs.append((yy-cy)**2 + (xx-cx)**2 <= r*r); cents.append((cx, cy))
    masks = list(objs)
    for _ in range(rng.integers(0, 3)):                    # merged masks
        if len(objs) >= 3:
            idx = rng.choice(len(objs), 3, replace=False)
            masks.append(ndimage.binary_dilation(np.any([objs[i] for i in idx], 0), iterations=2))
    for o in objs[: rng.integers(0, 4)]:                   # part masks
        ys, xs = np.nonzero(o)
        if len(ys):
            part = np.zeros_like(o); part[ys[0]:ys[0]+2, xs[:3].min():xs[:3].min()+3] = True
            masks.append(part & o)
    masks = np.array([m for m in masks if m.any()], bool).reshape(-1, H, W)
    scores = rng.uniform(0.6, 1.0, masks.shape[0])
    target = ndimage.gaussian_filter(np.any(objs, 0).astype(float) if objs else np.zeros((H, W)), 2) + rng.uniform(0, .05, (H, W))
    d1 = gpv.d1_peaks(target)
    amax = np.where(target > 0.2, 0, rng.integers(1, 4, (H, W))).astype(np.int64)
    a3c = target > 0.25
    dep = (ndimage.gaussian_filter(rng.uniform(0, 1, (H, W)), 4) + 0.3*a3c).astype(np.float32)
    nrm = np.stack([ndimage.gaussian_filter(rng.normal(size=(H, W)), 3) for _ in range(3)], -1)
    nrm[..., 2] += 1.0 + a3c
    return masks, scores, d1, amax, a3c, dep, nrm.astype(np.float16)

rng = np.random.default_rng(0)
n_img, mism, stage2 = 300, 0, 0
for k in range(n_img):
    m, s, d1, amax, a3c, dep, nrm = scene(rng)
    m2, s2 = (m[::-1], s[::-1]) if k % 2 else (m, s)       # two bases with different orders
    if k % 25 == 0: m2, s2 = m2[:0], s2[:0]                 # empty LM0 base
    sets = {"MixA": (m, s), "LM0": (m2, s2)}
    edge_nb, edge_mod = normal_edge_map(nrm), C.normal_edge_map(nrm)
    assert np.array_equal(edge_nb, edge_mod)
    nb = notebook_counts(sets, d1, amax, a3c, dep, edge_nb)
    out = C.compute_candidates({"a3c": a3c, "depth": dep, "d1": d1, "edge": edge_mod, "amax": amax, "sets": sets}, SG_CFG)
    pairs = [("GPV_R4+P[MixA]", "GPV_R4+P[MixA]"), ("GPV_K-n+P[MixA]", "GPV_K-n+P[MixA]"),
             ("GPV_K-n+P[LM0]", "GPV_K-n+P[LM0]"), ("H1b+P[MixA]_ac", "H1b+P[MixA]"),
             ("H1b-K3n+P[LM0]", "H1b-K3n+P[LM0]")]
    for a, b in pairs:
        if not np.isclose(nb[a], out[b]):
            mism += 1; print("MISMATCH", k, b, nb[a], out[b])
    if m.shape[0] and hybrid.h1b_count(m, d1, a3c, dep, SG_CFG)[1]["stage2_on"]: stage2 += 1
print(f"images {n_img} | mismatches {mism} | H1b stage2 ON in {stage2} | example row {out}")

rng = np.random.default_rng(0); g1 = hk = 0
for k in range(n_img):
    m, s, d1, amax, a3c, dep, nrm = scene(rng)
    out = C.compute_candidates({"a3c": a3c, "depth": dep, "d1": d1, "edge": C.normal_edge_map(nrm),
                                "amax": amax, "sets": {"MixA": (m, s), "LM0": (m, s)}}, SG_CFG)
    g1 += out["GPV_K-n+P[MixA]"] != out["GPV_R4+P[MixA]"]
    mp, _, _, _ = C.part_filter(m, s, amax, d1) if m.shape[0] else (m, s, 0, 0)
    hk += m.shape[0] > 0 and out["H1b-K3n+P[LM0]"] != C.h1b_count(mp, d1, a3c, dep, SG_CFG)
print("branch coverage: Gate-1 changes count in", g1, "| K3n changes H1b count in", hk)
