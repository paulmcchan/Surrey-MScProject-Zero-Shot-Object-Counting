# -*- coding: utf-8 -*-
"""
Semantic evidence — SAN target score map and A3c semantic mask.

Reproduces the frozen ABC setup:

    SAN (MendelXu/SAN), config san_clip_vit_res4_coco.yaml,
    checkpoint san_vit_b_16.pth (huggingface Mendel192/san)
    shortest side resized to 640 (PIL BILINEAR, rounded), CHW float 0-255
    vocabulary = [class_name, "background"]
    output sem_seg at original image resolution
    target_score = sem_seg[0], background_score = sem_seg[1]

    A3c: threshold = 90th percentile of the image's target_score;
         mask = target_score >= threshold

Installation of SAN / detectron2 is done in the notebook.
"""

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SAN_SHORT_SIDE = 640
A3C_PERCENTILE = 90
SAN_CONFIG_RELPATH = "configs/san_clip_vit_res4_coco.yaml"
SAN_CHECKPOINT_URL = (
    "https://huggingface.co/Mendel192/san/resolve/main/san_vit_b_16.pth"
)


@contextmanager
def _working_directory(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def build_san(san_root, checkpoint_path, device="cuda",
              config_relpath=SAN_CONFIG_RELPATH):
    """Build SAN-ViT-B/16 and load weights (as ABC Steps 2D-2F)."""
    san_root = Path(san_root)
    if str(san_root) not in sys.path:
        sys.path.insert(0, str(san_root))

    with _working_directory(san_root):
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from san.config import add_san_config
        from train_net import Trainer

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_san_config(cfg)
        cfg.merge_from_file(str(san_root / config_relpath))
        cfg.MODEL.DEVICE = device
        cfg.MODEL.WEIGHTS = str(checkpoint_path)
        cfg.freeze()

        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )
        model.eval()

    return model


def prepare_san_input(image, short_side=SAN_SHORT_SIDE):
    """PIL RGB -> CHW float tensor after shortest-side resize (ABC Step 7B)."""
    import torch
    from PIL import Image

    image = image.convert("RGB")
    width, height = image.size
    scale = short_side / min(height, width)
    resized = image.resize(
        (int(round(width * scale)), int(round(height * scale))),
        Image.BILINEAR,
    )
    tensor = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float()
    return tensor, height, width


def run_san(model, image, class_name, device="cuda",
            short_side=SAN_SHORT_SIDE):
    """
    Run SAN once (ABC Step 7C).

    Returns target_score, background_score as (H, W) float32 numpy arrays
    at original image resolution.
    """
    import torch

    tensor, height, width = prepare_san_input(image, short_side)
    inputs = [{
        "image": tensor.to(device),
        "height": height,
        "width": width,
        "vocabulary": [class_name, "background"],
    }]
    with torch.no_grad():
        outputs = model(inputs)

    sem_seg = outputs[0]["sem_seg"].detach().cpu().float()
    assert sem_seg.shape[0] == 2, f"Expected 2 channels, got {tuple(sem_seg.shape)}"
    return (
        sem_seg[0].numpy().astype(np.float32),
        sem_seg[1].numpy().astype(np.float32),
    )


def a3c_mask(target_score, percentile=A3C_PERCENTILE):
    """A3c: target_score >= per-image percentile threshold (ABC A.3)."""
    target_score = np.asarray(target_score, dtype=np.float32)
    threshold = float(np.percentile(target_score, percentile))
    return target_score >= threshold, threshold
