"""Final counting pipelines of the project (B5.1).

    pipeline_a : GPV_R4+P[MixA]  (primary)
    pipeline_b : GPV_K-n+P[LM0]  (secondary, geometric)

Both are thin wrappers around src.pipeline.candidates_v2 (shared core), so their common steps cannot diverge.
"""
from . import pipeline_a, pipeline_b  # noqa: F401
