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


# ============================================================
# D5 — CLIP Surgery target similarity map (B1 D notebook, 11H)
# ============================================================
#
# repo   : xmed-lab/CLIP_Surgery @ d4696d47f49cfe70f49140afe5eb94f94c5f59bc
# model  : CS-ViT-B/16
# input  : Resize((768, 768), BICUBIC) -> ToTensor -> CLIP Normalize
# text   : encode_text_with_prompt_ensemble([class_name]);
#          redundant feature = encode_text_with_prompt_ensemble([""])
# map    : clip_feature_surgery -> get_similarity_map(sim[:, 1:, :], (H, W))
#          (already min-max normalised by CLIP Surgery; NOT renormalised)
# peaks  : min_distance = round(0.020 x short side), threshold 0.50
# ============================================================

CLIP_SURGERY_COMMIT = "d4696d47f49cfe70f49140afe5eb94f94c5f59bc"
D5_RESOLUTION = 768
D5_MIN_DISTANCE_FRAC = 0.020
D5_SCORE_THRESHOLD = 0.50
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_clip_surgery(repo_path, device="cuda"):
    """Import CLIP Surgery's `clip` from repo_path and load CS-ViT-B/16."""
    repo_path = str(repo_path)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    if "clip" in sys.modules:
        del sys.modules["clip"]
    import clip  # CLIP Surgery's version

    model, _ = clip.load("CS-ViT-B/16", device=device)
    model.eval()
    return clip, model


def run_clip_surgery(clip_module, model, image, class_name, device="cuda",
                     resolution=D5_RESOLUTION):
    """PIL RGB image -> (H, W) float32 similarity map at original resolution."""
    import torch
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    image = image.convert("RGB")
    width, height = image.size
    preprocess = transforms.Compose([
        transforms.Resize((resolution, resolution),
                          interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    x = preprocess(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        feats = model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        text = clip_module.encode_text_with_prompt_ensemble(model, [class_name], device)
        redundant = clip_module.encode_text_with_prompt_ensemble(model, [""], device)
        sim = clip_module.clip_feature_surgery(feats, text, redundant)
        sim_map = clip_module.get_similarity_map(sim[:, 1:, :], (height, width))
    out = sim_map[0, :, :, 0].detach().float().cpu().numpy().astype(np.float32)
    assert out.shape == (height, width)
    return out


# ============================================================
# D6 — SAN intermediate query outputs (B1 D notebook, extract_d6_candidates)
# ============================================================

def run_san_d6(model, image, class_name, device="cuda", short_side=SAN_SHORT_SIDE):
    """
    One SAN forward reproducing SAN.forward() preprocessing, returning the
    D6 intermediates exactly as cached in the B1 D notebook:

        mask_preds        : (1, Q, h, w) float32  pre-sigmoid query mask logits
        target_query_prob : (1, Q)       float32  P(target | q)
                            softmax over [class_name, "background", no-object],
                            no-object column dropped, column 0 taken
    """
    import torch
    import torch.nn.functional as F
    from detectron2.structures import ImageList

    tensor, height, width = prepare_san_input(image, short_side)
    vocabulary = [class_name, "background"]
    with torch.no_grad():
        ov_w = (model.ov_classifier.logit_scale.exp()
                * model.ov_classifier.get_classifier_by_vocabulary(vocabulary))
        images = [(tensor.to(model.device) - model.pixel_mean) / model.pixel_std]
        images = ImageList.from_tensors(images, model.size_divisibility)
        clip_input = images.tensor
        if model.asymetric_input:
            clip_input = F.interpolate(clip_input, scale_factor=model.clip_resolution,
                                       mode="bilinear")
        clip_feats = model.clip_visual_extractor(clip_input)
        side_feats = model.side_adapter_network.forward_features(images.tensor, clip_feats)
        mask_preds_list, attn_biases_list = model.side_adapter_network.decode_masks(side_feats)
        mask_pred = mask_preds_list[-1]
        attn_bias = attn_biases_list[-1]
        mask_emb = model.clip_rec_head(clip_feats, attn_bias, normalize=True)
        mask_logits = torch.einsum("bqc,nc->bqn", mask_emb, ov_w)
        mask_cls = F.softmax(mask_logits, dim=-1)[..., :-1]
        target_query_prob = mask_cls[:, :, 0]
    return (mask_pred.detach().cpu().float(), target_query_prob.detach().cpu().float())


# ============================================================
# Expanded vocabulary (B3S Step 3)
# ============================================================

# Pre-declared background "stuff" vocabulary: COCO-Stuff stuff classes that
# describe surfaces and scene layout (SAN was trained on COCO-Stuff), with
# object-like stuff classes (fruit, vegetable, flower, food, leaves, stone,
# paper, cloth, ...) excluded because they could describe countable targets.
STUFF_BACKGROUND_VOCAB = [
    "sky", "clouds", "wall", "floor", "ceiling", "ground", "road", "pavement",
    "grass", "sand", "dirt", "gravel", "snow", "water", "sea", "river",
    "mountain", "hill", "fog", "building", "house", "roof", "table", "desk",
    "counter", "shelf", "cabinet", "cupboard", "door", "window", "curtain",
    "carpet", "rug", "mat", "tent", "fence", "railing", "platform",
    "playing field", "stairs", "bridge", "mud", "skyscraper",
]


def expanded_vocabulary(class_name, stuff=STUFF_BACKGROUND_VOCAB):
    """
    [class_name, "background", stuff...] with any stuff term removed if it
    overlaps the class name as a substring either way (e.g. target "windows"
    drops "window"). Target stays at index 0, "background" at index 1.
    """
    cn = class_name.lower()
    kept = [s for s in stuff if s not in cn and cn.rstrip("s") not in s]
    return [class_name, "background"] + kept


def run_san_vocab(model, image, vocabulary, device="cuda", short_side=SAN_SHORT_SIDE):
    """SAN forward with an arbitrary vocabulary -> (C, H, W) float32 score maps."""
    import torch

    tensor, height, width = prepare_san_input(image, short_side)
    inputs = [{"image": tensor.to(device), "height": height, "width": width,
               "vocabulary": list(vocabulary)}]
    with torch.no_grad():
        outputs = model(inputs)
    sem_seg = outputs[0]["sem_seg"].detach().cpu().float().numpy().astype(np.float32)
    assert sem_seg.shape[0] == len(vocabulary), (sem_seg.shape, len(vocabulary))
    return sem_seg
