# -*- coding: utf-8 -*-
"""
B0 — unguided SAM2 automatic mask generation (anchor counter).

Reproduces the frozen B2 B0 mask cache configuration:

    SAM2.1 Hiera-Small
    repo commit      2b90b9f5ceec907a1c18123530e92e794ad901a4
    checkpoint SHA   6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38
    model config     configs/sam2.1/sam2.1_hiera_s.yaml
    build_sam2(..., apply_postprocessing=False)
    SAM2AutomaticMaskGenerator(model)   # all default parameters
    image read with cv2 (BGR -> RGB), as in B2
    B0 count = number of returned masks; no extra filtering

Installation (SAM2 repo at the frozen commit) is done in the notebook.
"""

import hashlib
from pathlib import Path

import numpy as np

FROZEN_SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
FROZEN_CHECKPOINT_SHA256 = (
    "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
)
FROZEN_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"


def sha256_file(path, block_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def load_image_rgb_cv2(path):
    """Read an image exactly as B2 did: cv2.imread -> BGR2RGB, uint8 HxWx3."""
    import cv2

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert image_bgr is not None, f"Failed to read image: {path}"
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def build_amg(checkpoint_path, device="cuda", model_cfg=FROZEN_MODEL_CFG,
              verify_sha256=True):
    """Build SAM2 + default SAM2AutomaticMaskGenerator (frozen B0 setup)."""
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists(), f"Missing checkpoint: {checkpoint_path}"

    if verify_sha256:
        observed = sha256_file(checkpoint_path)
        assert observed == FROZEN_CHECKPOINT_SHA256, (
            f"Checkpoint SHA256 mismatch:\n  expected {FROZEN_CHECKPOINT_SHA256}"
            f"\n  observed {observed}"
        )

    model = build_sam2(
        model_cfg,
        str(checkpoint_path),
        device=device,
        apply_postprocessing=False,
    )
    return SAM2AutomaticMaskGenerator(model)


def run_amg(mask_generator, image_rgb):
    """
    Run AMG on one RGB uint8 image.

    Returns
    -------
    masks : (N, H, W) bool, in AMG output order
    meta  : dict of per-mask arrays (same fields as the B2 cache)
    """
    import torch

    h, w = image_rgb.shape[:2]
    with torch.inference_mode():
        outputs = mask_generator.generate(image_rgb)

    n = len(outputs)
    if n == 0:
        masks = np.zeros((0, h, w), dtype=bool)
    else:
        masks = np.stack(
            [np.asarray(o["segmentation"], dtype=bool) for o in outputs]
        )

    meta = {
        "areas": np.array([int(o["area"]) for o in outputs], dtype=np.int64),
        "predicted_iou": np.array(
            [float(o["predicted_iou"]) for o in outputs], dtype=np.float32
        ),
        "stability_score": np.array(
            [float(o["stability_score"]) for o in outputs], dtype=np.float32
        ),
        "bbox_xywh": np.array(
            [o["bbox"] for o in outputs], dtype=np.float32
        ).reshape(n, 4),
        "crop_box_xywh": np.array(
            [o["crop_box"] for o in outputs], dtype=np.float32
        ).reshape(n, 4),
        "point_xy": np.array(
            [o["point_coords"][0] for o in outputs], dtype=np.float32
        ).reshape(n, 2),
    }
    return masks, meta


def compare_mask_sets(masks_a, masks_b):
    """
    Order-insensitive exact comparison of two (N, H, W) bool mask stacks.

    Returns dict: count_a, count_b, exact_multiset_match, n_shared_exact.
    """
    from collections import Counter

    def keys(m):
        m = np.asarray(m, dtype=bool)
        if m.shape[0] == 0:
            return Counter()
        packed = np.packbits(m.reshape(m.shape[0], -1), axis=1)
        return Counter(row.tobytes() for row in packed)

    ka, kb = keys(masks_a), keys(masks_b)
    shared = sum((ka & kb).values())
    return {
        "count_a": int(sum(ka.values())),
        "count_b": int(sum(kb.values())),
        "exact_multiset_match": ka == kb,
        "n_shared_exact": int(shared),
    }
