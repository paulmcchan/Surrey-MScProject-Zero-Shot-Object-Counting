# -*- coding: utf-8 -*-
"""
candidates_v2 — shared core of the two final pipelines (B5.1).

Systems (selected in B4c; roles fixed):
    "A"  GPV_R4+P[MixA]   (primary)   Z-LM0 automatic + LM0 guided -> R4 -> P -> number of masks
    "B"  GPV_K-n+P[LM0]   (secondary) LM0 automatic + LM0 guided   -> R4 -> P -> GPV_K-n (normal edges)

Modes:
    "single"  one class per image (FSC-147, CARPK): exactly candidates_v1 (B4c), plus a fallback
              ladder when the system's mask set is empty: L1 = two-class SAN region, L2 = whole image.
    "multi"   several classes per image (OmniCount-191): U_both (B5.0 Part 4 decision) —
              per-class tiled SAN regions, score-based resolution, ONE SAM2 run on the union of the
              raw regions, object-level assignment (guided masks -> their prompt class; automatic
              masks -> class whose resolved region covers > 50%), P per class, count. No ladder.

Every function below is taken from code validated in B5.0 (Part 2 Step 2.3: single-pass SAN;
Part 3 Step 3.1: prompts / D1 / A3c; Part 4 Step 4.2: generation, R4 with origin, assignment,
counting). With one class, "multi" reproduces "single" (ladder off) and candidates_v1 exactly.

Heavy dependencies (torch, SAN/detectron2, SAM2, diffusers) are imported lazily inside the
functions that need them, so this module imports on a CPU-only machine.
"""

import json
import time
from pathlib import Path

import numpy as np

from . import b0, gpv, prompts, semantic
from . import candidates as C

VERSION = "candidates_v2"

SYSTEMS = {
    "A": {"name": "GPV_R4+P[MixA]", "generator": "MixA", "counter": "R4"},
    "B": {"name": "GPV_K-n+P[LM0]", "generator": "LM0", "counter": "Kn"},
}
TILE_FRAC = 0.60
INSIDE_FRAC = 0.5             # > 50%-inside rule (discard at generation; assignment coverage)
NORMALS = {"model": "prs-eth/marigold-normals-v1-1", "dtype": "float16", "num_inference_steps": 4,
           "ensemble_size": 1, "processing_resolution": 768, "seed": 0}
SAM2 = {"checkpoint": "sam2.1_hiera_large.pt", "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "points_per_side": 32, "zoom_crop_n_layers": 1}


def frozen_config():
    return {
        "version": VERSION, "systems": SYSTEMS, "modes": ["single", "multi"],
        "san": {"vocabulary": "class + 'background' + STUFF_BACKGROUND_VOCAB (v1 substring rule)",
                "tiles": f"2x2 corner tiles of {TILE_FRAC:.0%}, scores averaged", "short_side": semantic.SAN_SHORT_SIDE,
                "single_pass": "union vocabulary, per-class subset softmax (exact)"},
        "sam2": SAM2, "normals": NORMALS, "inside_frac": INSIDE_FRAC,
        "counting": {"nms_iou": C.NMS_IOU, "part_frac": C.PART_FRAC, "nest_frac": C.NEST_FRAC,
                     "k3n_h_frac": C.K3N_H_FRAC, "k3n_min_seg_frac": C.K3N_MIN_SEG_FRAC,
                     "region_min_frac": C.REGION_MIN_FRAC},
        "single_mode": "v1 + ladder (L1 two-class region, L2 whole image) when the mask set is empty",
        "multi_mode": "U_both: union generation, prompt-consistent guided masks, resolved-region coverage; no ladder",
        "candidates_v1": C.VERSION,
    }


# ============================================================
# Models
# ============================================================

class Models:
    """Holds the frozen models. Build with load_models(); only what the requested systems need is loaded."""
    def __init__(self):
        self.san = self.predictor = None
        self.amg = {}
        self.normals = None
        self.device = "cuda"


def load_models(san_root, san_ckpt, sam2_ckpt, systems=("A", "B"), device="cuda", sam2_cfg=SAM2["config"]):
    import torch  # noqa: F401
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    M = Models(); M.device = device
    M.san = semantic.build_san(Path(san_root), Path(san_ckpt), device=device)
    assert M.san.sem_seg_postprocess_before_inference, "capture assumes post-processing before inference"
    M.predictor = SAM2ImagePredictor(build_sam2(sam2_cfg, str(sam2_ckpt), device=device).eval())
    gens = {SYSTEMS[s]["generator"] for s in systems}
    if "MixA" in gens:
        mz = build_sam2(sam2_cfg, str(sam2_ckpt), device=device, apply_postprocessing=False).eval()
        M.amg["MixA"] = SAM2AutomaticMaskGenerator(mz, points_per_side=SAM2["points_per_side"],
                                                   crop_n_layers=SAM2["zoom_crop_n_layers"])
    if "LM0" in gens:
        M.amg["LM0"] = b0.build_amg(sam2_ckpt, device=device, model_cfg=sam2_cfg, verify_sha256=False)
    if any(SYSTEMS[s]["counter"] == "Kn" for s in systems):
        import torch
        from diffusers import MarigoldNormalsPipeline
        M.normals = MarigoldNormalsPipeline.from_pretrained(NORMALS["model"], dtype=torch.float16).to(device)
        M.normals.set_progress_bar_config(disable=True)
    return M


# ============================================================
# Semantic evidence (B5.0 Part 2 Step 2.3; Part 3 Step 3.1)
# ============================================================

def union_vocabulary(prompt_list):
    return list(dict.fromkeys(list(prompt_list) + ["background"] + list(semantic.STUFF_BACKGROUND_VOCAB)))


def san_capture(M, img, vocab, low_res=False):
    """One SAN forward: (mask_cls logits (Q, V+1), sigmoid(mask_pred) (Q, H, W)[, low-res mask logits (Q, h, w)])."""
    import torch
    tensor, h, w = semantic.prepare_san_input(img)
    cap, low = {}, {}

    def grab(mask_cls, mask_pred):
        cap["cls"] = mask_cls.float(); cap["sig"] = mask_pred.float().sigmoid()
        return torch.zeros((1,) + tuple(mask_pred.shape[1:]), device=mask_pred.device)
    hook = (M.san.side_adapter_network.register_forward_hook(
        lambda mod, inp, out: low.__setitem__("mp", out[0][-1].detach().float())) if low_res else None)
    M.san.semantic_inference = grab
    try:
        with torch.no_grad():
            M.san([{"image": tensor.to(M.device), "height": h, "width": w, "vocabulary": list(vocab)}])
    finally:
        del M.san.semantic_inference
        if hook is not None:
            hook.remove()
    return (cap["cls"], cap["sig"], low["mp"][0]) if low_res else (cap["cls"], cap["sig"])


def subset_maps(cls, sig, idx):
    import torch
    import torch.nn.functional as F
    cols = torch.tensor(list(idx) + [cls.shape[-1] - 1], device=cls.device)
    return torch.einsum("qc,qhw->chw", F.softmax(cls[:, cols], dim=-1)[:, :-1], sig)


def tiled_capture(M, img, vocab):
    w, h = img.size
    th, tw = int(np.ceil(TILE_FRAC * h)), int(np.ceil(TILE_FRAC * w))
    caps = [(y0, x0) + san_capture(M, img.crop((x0, y0, x0 + tw, y0 + th)), vocab)
            for y0 in (0, h - th) for x0 in (0, w - tw)]
    return caps, (h, w, th, tw)


def tiled_subset(M, caps, shape, idx):
    import torch
    h, w, th, tw = shape
    acc = torch.zeros((len(idx), h, w), device=M.device); cnt = torch.zeros((h, w), device=M.device)
    for y0, x0, cls, sig in caps:
        acc[:, y0:y0 + th, x0:x0 + tw] += subset_maps(cls, sig, idx); cnt[y0:y0 + th, x0:x0 + tw] += 1.0
    return acc / cnt[None]


def class_evidence(cls_u, sig_u, mp_low, U, p):
    """Two-class target score, S2 region, A3c, D1 peaks and dense_recovery prompts for prompt p."""
    import torch
    import torch.nn.functional as F
    m2 = subset_maps(cls_u, sig_u, [U.index(p), U.index("background")])
    target = m2[0].cpu().numpy(); s2 = (m2[0] > m2[1]).cpu().numpy()
    cols = torch.tensor([U.index(p), U.index("background"), cls_u.shape[-1] - 1], device=cls_u.device)
    tqp = F.softmax(cls_u[:, cols], dim=-1)[:, 0]
    a3c, _ = semantic.a3c_mask(target)
    pts, _ = prompts.dense_recovery_points(a3c, mp_low.cpu(), tqp.cpu())
    return target, s2, a3c, gpv.d1_peaks(target), np.asarray(pts, float).reshape(-1, 2)


def compute_evidence(M, pil, prompt_of):
    """Per class: R0 (tiled expanded argmax == 0), T0, AM (argmax map), S2, D1, PTS. prompt_of: {class: prompt}."""
    import torch
    U = union_vocabulary(prompt_of.values())
    caps, shape = tiled_capture(M, pil, U)
    cls_u, sig_u, mp_low = san_capture(M, pil, U, low_res=True)
    ev = {k: {} for k in ["R0", "T0", "AM", "S2", "D1", "PTS"]}
    for c, p in prompt_of.items():
        Tm = tiled_subset(M, caps, shape, [U.index(x) for x in semantic.expanded_vocabulary(p)])
        am = Tm.argmax(0)
        ev["R0"][c] = (am == 0).cpu().numpy(); ev["T0"][c] = Tm[0].cpu().numpy()
        ev["AM"][c] = am.cpu().numpy().astype(np.int32)
        _, ev["S2"][c], _, ev["D1"][c], ev["PTS"][c] = class_evidence(cls_u, sig_u, mp_low, U, p)
        del Tm, am
    del caps, cls_u, sig_u, mp_low
    if M.device == "cuda":
        torch.cuda.empty_cache()
    return ev


def normals_edge(M, pil):
    import torch
    g = torch.Generator(device=M.device).manual_seed(NORMALS["seed"])
    with torch.inference_mode():
        out = M.normals(pil, num_inference_steps=NORMALS["num_inference_steps"], ensemble_size=NORMALS["ensemble_size"],
                        processing_resolution=NORMALS["processing_resolution"], generator=g)
    nrm = np.squeeze(np.asarray(out.prediction)).astype(np.float32)
    assert nrm.shape[:2] == (pil.size[1], pil.size[0]) and nrm.shape[2] == 3, nrm.shape
    return C.normal_edge_map(nrm.astype(np.float16))          # as cached (float16) in B3MN / B4b


# ============================================================
# Mask generation and R4 with origin (B5.0 Part 4 Step 4.2)
# ============================================================

def keep_inside(masks, region):
    if masks.shape[0] == 0:
        return np.zeros(0, bool)
    inside = (masks & region[None]).reshape(masks.shape[0], -1).sum(1)
    return inside > INSIDE_FRAC * np.maximum(masks.reshape(masks.shape[0], -1).sum(1), 1)


def safe_amg(amg, image):
    try:
        return b0.run_amg(amg, image), False
    except IndexError:                                   # SAM2 empty-result case with crops
        h, w = image.shape[:2]
        return (np.zeros((0, h, w), bool), {"predicted_iou": np.zeros(0, np.float32)}), True


def generate(M, img_np, region, pts_by_class, generator):
    """Automatic (MixA = Z-LM0, LM0) + LM0 guided on image x region; > 50%-outside discard; guided origins kept."""
    import torch
    h, w = img_np.shape[:2]
    inp = (img_np * region[..., None]).astype(np.uint8)
    (bm, bmeta), _ = safe_amg(M.amg[generator], inp)
    biou = np.asarray(bmeta["predicted_iou"])
    gm, gs, go = [], [], []
    with torch.inference_mode():
        if any(len(p) for _, p in pts_by_class):
            M.predictor.set_image(inp)
        for c, pts in pts_by_class:
            for p in pts:
                m, sc, _ = M.predictor.predict(point_coords=np.asarray([p], np.float32),
                                               point_labels=np.asarray([1], np.int32), multimask_output=False)
                gm.append(np.asarray(m[0], bool)); gs.append(float(np.asarray(sc)[0])); go.append(c)
    gm = np.asarray(gm) if gm else np.zeros((0, h, w), bool); gs = np.asarray(gs, float); go = np.asarray(go, object)
    kb, kg = keep_inside(bm, region), keep_inside(gm, region)
    return bm[kb], biou[kb], gm[kg], gs[kg], go[kg]


def r4_tracked(bm, biou, gm, gs, go):
    """Guided NMS, then R4 pool (guided first, automatic second), NMS; returns masks, scores, origin.
    Asserts equality with gpv.build_r4."""
    kg = gpv.nms_masks(gm, gs, C.NMS_IOU) if gm.shape[0] else np.zeros(0, int)
    g2, s2, o2 = gm[kg], gs[kg], go[kg]
    m_ref, _, _ = gpv.build_r4(g2, s2, bm, biou, C.NMS_IOU)
    if g2.shape[0] + bm.shape[0] == 0:
        return np.zeros((0,) + gm.shape[1:], bool), np.zeros(0), np.zeros(0, object)
    pool = np.concatenate([x for x in (g2, bm) if x.shape[0]])
    sc = np.concatenate([np.asarray(s2, float), np.asarray(biou, float)])
    org = np.concatenate([o2, np.full(bm.shape[0], None, object)])
    keep = gpv.nms_masks(pool, sc, C.NMS_IOU)
    assert np.array_equal(pool[keep], np.asarray(m_ref, bool)), "origin tracking diverged from build_r4"
    return pool[keep], sc[keep], org[keep]


# ============================================================
# Resolution, assignment, counting (B5.0 Part 4 Step 4.2)
# ============================================================

def resolve_np(R, T, members):
    out = dict(R)
    if len(members) >= 2:
        stack = np.stack([np.where(R[c], T[c], -1.0) for c in members]); win = stack.argmax(0)
        for i, c in enumerate(members):
            out[c] = R[c] & (win == i)
    return out


def pooled(masks, maps):
    n = masks.shape[0]; flat = masks.reshape(n, -1)
    area = np.maximum(flat.sum(1), 1).astype(np.float32)
    out = np.zeros((n, len(maps)), np.float32)
    for k, mp in enumerate(maps):
        v = mp.reshape(-1).astype(np.float32)
        for i0 in range(0, n, 64):
            out[i0:i0 + 64, k] = flat[i0:i0 + 64].astype(np.float32) @ v
    return out / area[:, None]


def assign(masks, origin, classes, REG, T0):
    """Object-level assignment (U_both): coverage > 50% of REG[c]; ties -> highest mean T0;
    guided masks may go only to their prompt class. Returns class index per mask (-1 = discard)."""
    n = masks.shape[0]
    if n == 0 or not classes:
        return np.full(n, -1)
    cov = pooled(masks, [REG[c].astype(np.float32) for c in classes])
    sc = pooled(masks, [T0[c] for c in classes]); sc[cov <= INSIDE_FRAC] = -np.inf
    for i, o in enumerate(origin):
        if o is not None:
            sc[i, np.array([c != o for c in classes])] = -np.inf
    lab = sc.argmax(1); lab[~np.isfinite(sc.max(1))] = -1
    return lab


def count_system(counter, m, s, amax, d1, edge):
    """P, then R4 mask count or GPV_K-n (candidates_v1 functions)."""
    if m.shape[0] == 0:
        return 0
    mp, sp, _, _ = C.part_filter(m, np.asarray(s, float), amax, d1)
    if counter == "R4":
        return int(mp.shape[0])
    return int(C.gpv_k_n_count(mp, sp, d1=d1, edge=edge))


# ============================================================
# Modes
# ============================================================

def count_single(M, img_np, ev, c, system, edge, ladder=True):
    """Single-class mode: v1 on R0; if the mask set is empty, L1 (two-class region) then L2 (whole image)."""
    spec = SYSTEMS[system]
    levels = [(0, ev["R0"][c], ev["AM"][c])]
    if ladder:
        s2 = ev["S2"][c]
        levels += [(1, s2, (~s2).astype(np.int32)), (2, np.ones_like(s2), np.zeros(s2.shape, np.int32))]
    for lvl, reg, amax in levels:
        if not reg.any():
            continue
        m, s, _ = r4_tracked(*generate(M, img_np, reg, [(c, ev["PTS"][c])], spec["generator"]))
        if m.shape[0] > 0:
            return count_system(spec["counter"], m, s, amax, ev["D1"][c], edge), lvl
    return 0, -1


def count_multi(M, img_np, ev, classes, system, edge):
    """Multi-label mode U_both."""
    spec = SYSTEMS[system]
    out = {c: 0 for c in classes}
    ne = [c for c in classes if ev["R0"][c].any()]
    if not ne:
        return out
    R0res = resolve_np(ev["R0"], ev["T0"], ne)
    union = np.any([ev["R0"][c] for c in ne], axis=0)
    m, s, org = r4_tracked(*generate(M, img_np, union, [(c, ev["PTS"][c]) for c in ne], spec["generator"]))
    lab = assign(m, org, ne, R0res, ev["T0"])
    for k, c in enumerate(ne):
        am = ev["AM"][c].copy(); am[ev["R0"][c] & ~R0res[c]] = 1
        out[c] = count_system(spec["counter"], m[lab == k], s[lab == k], am, ev["D1"][c], edge)
    return out


def count_image(M, image, class_names, mode, systems=("A", "B"), prompt_of=None, ladder=True):
    """Count every class in class_names. image: path or PIL image. prompt_of: optional {class: text prompt}
    (default: the class name). Returns {"counts": {system: {class: n}}, "diag": {...}}."""
    from PIL import Image
    assert mode in ("single", "multi"), mode
    t0 = time.time()
    path = image if isinstance(image, (str, Path)) else None
    pil = Image.open(path).convert("RGB") if path else image.convert("RGB")
    img_np = b0.load_image_rgb_cv2(path) if path else np.asarray(pil, np.uint8).copy()
    assert img_np.shape[:2] == (pil.size[1], pil.size[0])
    classes = list(class_names)
    if mode == "single":
        assert len(classes) == 1, "single mode expects exactly one class"
    prompt_of = {c: (prompt_of or {}).get(c, c) for c in classes}
    ev = compute_evidence(M, pil, prompt_of)
    diag = {"t_evidence_s": time.time() - t0, "nonempty_classes": int(sum(ev["R0"][c].any() for c in classes))}
    edge = None
    if any(SYSTEMS[s]["counter"] == "Kn" for s in systems):
        ts = time.time(); edge = normals_edge(M, pil); diag["t_normals_s"] = time.time() - ts
    counts = {}
    for s in systems:
        ts = time.time()
        if mode == "single":
            n, lvl = count_single(M, img_np, ev, classes[0], s, edge, ladder)
            counts[s] = {classes[0]: n}; diag[f"{s}_ladder_level"] = lvl
        else:
            counts[s] = count_multi(M, img_np, ev, classes, s, edge)
        diag[f"t_{s}_s"] = time.time() - ts
    diag["t_total_s"] = time.time() - t0
    return {"counts": counts, "diag": diag}


# ============================================================
# Resumable batch runner (B6)
# ============================================================

def run_batch(M, items, out_dir, mode, systems=("A", "B"), ladder=True, progress=True):
    """items: iterable of dicts {"key", "image_path", "classes", optional "prompts" {class: prompt}, optional "meta"}.
    Writes one JSON per item to out_dir (skips existing). Returns the number processed in this call."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir.parent / f"{out_dir.name}_config.json"
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps({**frozen_config(), "mode": mode, "systems_run": list(systems),
                                        "ladder": ladder}, indent=2))
    todo = [it for it in items if not (out_dir / f"{it['key']}.json").exists()]
    it_ = todo
    if progress:
        from tqdm.auto import tqdm
        it_ = tqdm(todo, desc=f"{VERSION} {mode}")
    for it in it_:
        r = count_image(M, it["image_path"], it["classes"], mode, systems, it.get("prompts"), ladder)
        rec = {"key": it["key"], "image_path": str(it["image_path"]), "classes": list(it["classes"]),
               "meta": it.get("meta", {}), "version": VERSION, **r}
        tmp = out_dir / f"{it['key']}.json.tmp"
        tmp.write_text(json.dumps(rec)); tmp.replace(out_dir / f"{it['key']}.json")    # atomic write
    return len(todo)
