"""Pipeline A: GPV_R4+P[MixA] (primary).

Z-LM0 automatic + LM0 guided -> R4 -> P -> number of masks.
Modes: "single" (one class; v1 + fallback ladder) and "multi" (several classes; U_both).

Usage:
    from src.counting import pipeline_a
    models = pipeline_a.load(san_root, san_ckpt, sam2_ckpt)
    out = pipeline_a.count_image(models, "image.jpg", ["apples", "banana"], mode="multi")
    out["counts"]   # {"apples": 3, "banana": 2}
"""
from src.pipeline import candidates_v2 as core

SYSTEM = "A"
NAME = core.SYSTEMS[SYSTEM]["name"]


def load(san_root, san_ckpt, sam2_ckpt, device="cuda"):
    """Load only the models this pipeline needs."""
    return core.load_models(san_root, san_ckpt, sam2_ckpt, systems=(SYSTEM,), device=device)


def count_image(models, image, class_names, mode="multi", prompt_of=None, ladder=True):
    """Counts for each class in class_names. Returns {"counts": {class: n}, "diag": {...}}."""
    r = core.count_image(models, image, class_names, mode, systems=(SYSTEM,), prompt_of=prompt_of, ladder=ladder)
    return {"counts": r["counts"][SYSTEM], "diag": r["diag"]}


def run_batch(models, items, out_dir, mode, ladder=True, progress=True):
    """Resumable batch run (one JSON per item in out_dir)."""
    return core.run_batch(models, items, out_dir, mode, systems=(SYSTEM,), ladder=ladder, progress=progress)
