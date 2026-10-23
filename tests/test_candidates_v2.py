"""CPU unit tests for src/pipeline/candidates_v2.py and src/counting (synthetic data, fake SAM2).

Checks:
 1. Core assignment / resolution / pooling / counting equal the B5.0 Part 4 notebook implementations.
 2. One class: multi mode == single mode with the ladder off (both systems).
 3. Ladder order: L0 if R0 gives masks; else L1 (two-class region); else L2 (whole image).
 4. Standalone pipelines (src.counting) == joint runner.
 5. run_batch is resumable and writes one JSON per item.
"""
import json, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.pipeline.candidates_v2 as core
import src.pipeline.candidates as C
import src.pipeline.gpv as gpv
import src.pipeline.b0 as b0
from src.counting import pipeline_a, pipeline_b

# ---------------- fake SAM2 (deterministic in the masked input) ----------------
def fake_run_amg(amg, image):
    h, w = image.shape[:2]; rng = np.random.default_rng(int(image.sum()) % 9973)
    lit = image.any(2); masks = []
    for _ in range(rng.integers(0, 12)):
        y, x = rng.integers(0, h - 8), rng.integers(0, w - 8); s = rng.integers(4, 9)
        m = np.zeros((h, w), bool); m[y:y + s, x:x + s] = True
        if (m & lit).sum() > 10:
            masks.append(m)
    masks = np.array(masks, bool).reshape(-1, h, w)
    return masks, {"predicted_iou": rng.uniform(.6, 1, len(masks)).astype(np.float32)}
b0.run_amg = fake_run_amg

class FakePredictor:
    def set_image(self, im): self.im = im
    def predict(self, point_coords, point_labels, multimask_output):
        h, w = self.im.shape[:2]; x, y = point_coords[0].astype(int); m = np.zeros((1, h, w), bool)
        m[0, max(0, y - 4):y + 4, max(0, x - 4):x + 4] = True
        return m, np.array([0.5 + 0.4 * ((x * 7 + y) % 10) / 10]), None

def fake_models():
    M = core.Models(); M.device = "cpu"; M.predictor = FakePredictor(); M.amg = {"MixA": object(), "LM0": object()}
    return M

def scene(rng, classes, H=60, W=80):
    img = rng.integers(0, 255, (H, W, 3)).astype(np.uint8)
    ev = {k: {} for k in ["R0", "T0", "AM", "S2", "D1", "PTS"]}
    for c in classes:
        R = np.zeros((H, W), bool)
        for _ in range(rng.integers(0, 4)):
            y, x = rng.integers(0, H - 15), rng.integers(0, W - 15); R[y:y + rng.integers(8, 15), x:x + rng.integers(8, 15)] = True
        T = rng.uniform(0, 1, (H, W)).astype(np.float32); T[R] += 0.5
        am = np.where(R, 0, rng.integers(1, 5, (H, W))).astype(np.int32)
        s2 = R | (rng.uniform(0, 1, (H, W)) > 0.7)
        ys, xs = np.nonzero(R)
        pts = np.stack([xs[::17], ys[::17]], 1).astype(float) if len(ys) else np.zeros((0, 2))
        ev["R0"][c], ev["T0"][c], ev["AM"][c], ev["S2"][c] = R, T, am, s2
        ev["D1"][c] = gpv.d1_peaks(T); ev["PTS"][c] = pts
    edge = rng.uniform(0, 1, (H, W)).astype(np.float32)
    return img, ev, edge

# ---------------- B5.0 Part 4 reference implementations (verbatim) ----------------
def ref_assign_rule(masks, origin, classes, REG, T0, prompt_consistent=True):
    n = masks.shape[0]
    if n == 0 or not classes:
        return np.full(n, -1)
    cov = core.pooled(masks, [REG[c].astype(np.float32) for c in classes])
    sc = core.pooled(masks, [T0[c] for c in classes]); sc[cov <= 0.5] = -np.inf
    if prompt_consistent:
        for i, o in enumerate(origin):
            if o is not None:
                sc[i, np.array([c != o for c in classes])] = -np.inf
    lab = sc.argmax(1); lab[~np.isfinite(sc.max(1))] = -1
    return lab

def ref_count_system(system, m, s, amax, d1, edge):
    if m.shape[0] == 0:
        return 0
    mp, sp, _, _ = C.part_filter(m, np.asarray(s, float), amax, d1)
    return int(mp.shape[0]) if system == "R4" else int(C.gpv_k_n_count(mp, sp, d1=d1, edge=edge))

def test_core_equals_part4(n=60):
    rng = np.random.default_rng(0); M = fake_models()
    for _ in range(n):
        classes = ["a", "b", "c"]; img, ev, edge = scene(rng, classes)
        ne = [c for c in classes if ev["R0"][c].any()]
        if not ne:
            continue
        R0res = core.resolve_np(ev["R0"], ev["T0"], ne)
        union = np.any([ev["R0"][c] for c in ne], axis=0)
        for gen, ctr in [("MixA", "R4"), ("LM0", "Kn")]:
            m, s, org = core.r4_tracked(*core.generate(M, img, union, [(c, ev["PTS"][c]) for c in ne], gen))
            lab = core.assign(m, org, ne, R0res, ev["T0"])
            assert np.array_equal(lab, ref_assign_rule(m, org, ne, R0res, ev["T0"], True))
            for k, c in enumerate(ne):
                am = ev["AM"][c].copy(); am[ev["R0"][c] & ~R0res[c]] = 1
                assert core.count_system(ctr, m[lab == k], s[lab == k], am, ev["D1"][c], edge) == \
                       ref_count_system(ctr, m[lab == k], s[lab == k], am, ev["D1"][c], edge)

def test_one_class_multi_equals_single(n=80):
    rng = np.random.default_rng(1); M = fake_models(); checked = 0
    for _ in range(n):
        img, ev, edge = scene(rng, ["x"])
        for sysk in ["A", "B"]:
            single, lvl = core.count_single(M, img, ev, "x", sysk, edge, ladder=False)
            multi = core.count_multi(M, img, ev, ["x"], sysk, edge)["x"]
            assert single == multi, (sysk, single, multi)
            checked += single > 0
    assert checked > 10, "too few non-trivial cases"

def test_ladder_order():
    rng = np.random.default_rng(2); M = fake_models()
    img, ev, edge = scene(rng, ["x"])
    ev["R0"]["x"] = np.zeros_like(ev["R0"]["x"])                                     # empty R0 -> L1
    n, lvl = core.count_single(M, img, ev, "x", "A", edge, ladder=True)
    assert lvl in (1, 2)
    ev["S2"]["x"] = np.zeros_like(ev["S2"]["x"])                                     # empty S2 too -> L2
    n2, lvl2 = core.count_single(M, img, ev, "x", "A", edge, ladder=True)
    assert lvl2 == 2 or (lvl2 == -1 and n2 == 0)
    n3, lvl3 = core.count_single(M, img, ev, "x", "A", edge, ladder=False)            # ladder off -> 0
    assert (n3, lvl3) == (0, -1)

def test_standalone_equals_joint(n=20):
    rng = np.random.default_rng(3); M = fake_models()
    from PIL import Image
    for _ in range(n):
        classes = ["a", "b"]; img, ev, edge = scene(rng, classes)
        core.compute_evidence = lambda M_, pil, prompt_of, ev=ev: ev
        core.normals_edge = lambda M_, pil, edge=edge: edge
        pil = Image.fromarray(img)
        joint = core.count_image(M, pil, classes, "multi", systems=("A", "B"))["counts"]
        assert pipeline_a.count_image(M, pil, classes, "multi")["counts"] == joint["A"]
        assert pipeline_b.count_image(M, pil, classes, "multi")["counts"] == joint["B"]

def test_run_batch_resumable():
    rng = np.random.default_rng(4); M = fake_models()
    from PIL import Image
    img, ev, edge = scene(rng, ["a", "b"])
    core.compute_evidence = lambda M_, pil, prompt_of: ev
    core.normals_edge = lambda M_, pil: edge
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i in range(3):
            p = Path(d) / f"im{i}.png"; Image.fromarray(img).save(p); paths.append(p)
        b0.load_image_rgb_cv2 = lambda p: np.asarray(Image.open(p).convert("RGB"))
        items = [{"key": f"k{i}", "image_path": p, "classes": ["a", "b"]} for i, p in enumerate(paths)]
        out = Path(d) / "rows"
        assert core.run_batch(M, items, out, "multi", progress=False) == 3
        assert core.run_batch(M, items, out, "multi", progress=False) == 0
        rec = json.loads((out / "k0.json").read_text())
        assert set(rec["counts"]) == {"A", "B"} and (Path(d) / "rows_config.json").exists()

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
