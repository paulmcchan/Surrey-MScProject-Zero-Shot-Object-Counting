# -*- coding: utf-8 -*-
"""
Geometric evidence — Marigold relative depth.

Reproduces the frozen M1 depth cache:

    prs-eth/marigold-depth-v1-1 (diffusers MarigoldDepthPipeline), fp16
    num_inference_steps   4
    ensemble_size         1
    processing_resolution 768
    generator             torch.Generator(device).manual_seed(0) per image
    input                 PIL RGB image
    output                squeeze(prediction) -> float32 (H, W)

M1 verified this is bit-exact on a T4 (regenerated == cached).
Package versions used by M1: diffusers 0.40.0, transformers 5.16.1,
accelerate 1.14.0.
"""

import numpy as np

MARIGOLD_MODEL_ID = "prs-eth/marigold-depth-v1-1"
MARIGOLD_STEPS = 4
MARIGOLD_ENSEMBLE = 1
MARIGOLD_RESOLUTION = 768
MARIGOLD_SEED = 0


def build_marigold(device="cuda", model_id=MARIGOLD_MODEL_ID):
    import torch
    from diffusers import MarigoldDepthPipeline

    pipe = MarigoldDepthPipeline.from_pretrained(model_id, dtype=torch.float16)
    pipe = pipe.to(device)
    # Per-image progress bars add clutter only; results are unaffected.
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_marigold(pipe, image, device="cuda", seed=MARIGOLD_SEED,
                 steps=MARIGOLD_STEPS, ensemble_size=MARIGOLD_ENSEMBLE,
                 processing_resolution=MARIGOLD_RESOLUTION):
    """PIL RGB image -> (H, W) float32 relative depth (M1 Step 5B)."""
    import torch

    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        output = pipe(
            image,
            num_inference_steps=steps,
            ensemble_size=ensemble_size,
            processing_resolution=processing_resolution,
            generator=generator,
        )
    depth = np.squeeze(np.asarray(output.prediction)).astype(np.float32)
    width, height = image.size
    assert depth.shape == (height, width), (
        f"Depth shape {depth.shape} != image {(height, width)}"
    )
    assert np.isfinite(depth).all(), "Depth contains NaN/Inf"
    return depth
