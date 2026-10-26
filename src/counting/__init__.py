"""Final counting pipelines of the project (B5.1).

    counting_gpv_r4_p_mixa : GPV_R4+P[MixA]  (primary)
    counting_gpv_kn_p_lm0  : GPV_K-n+P[LM0]  (secondary, geometric)

Both are thin wrappers around src.pipeline.candidates_v2 (shared core, frozen in B5.1),
so their common steps cannot diverge.
"""
from . import counting_gpv_r4_p_mixa, counting_gpv_kn_p_lm0  # noqa: F401
