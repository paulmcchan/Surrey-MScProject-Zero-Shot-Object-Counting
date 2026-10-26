# Notebooks

Grouped by project phase. Every notebook clones the repo and imports `src/`; data and results live on Google Drive.
Final pipelines: `src/counting/counting_gpv_r4_p_mixa.py` (GPV_R4+P[MixA], primary) and `src/counting/counting_gpv_kn_p_lm0.py` (GPV_K-n+P[LM0], secondary), sharing `src/pipeline/candidates_v2.py`.

## `00_datasets/` — Datasets

- `Datasets_Inspection.ipynb` — Audit of FSC-147 and OmniCount-191 (structure, splits, counts, overlaps).
- `Datasets_Loader.ipynb` — Dataset loaders (src/datasets.py) and sanity checks.

## `01_b0_baseline/` — B0 — baseline

- `B0_SAM2_Unguided_Baseline.ipynb` — B0: unguided SAM2 automatic-mask counting baseline on FSC-147 val.
- `B0_SAM2_Unguided_dev100.ipynb` — B0 on the dev100 development subset.

## `02_b1_b2_semantic_evidence/` — B1 / B2 — semantic evidence

- `B1_Experiments_ABC_RegionBased_Evidence.ipynb` — B1 A–C: region-based semantic evidence from SAN.
- `B1_Experiments_D_ActivationBased_Evidence.ipynb` — B1 D: activation-based evidence (SAN intermediate features).
- `B1_Experiments_E_RegionPoint_Relationship_and_Semantic_Integration.ipynb` — B1 E: region–point relationship and semantic integration.
- `B1_Sparse_Scene_Failure_Diagnostic.ipynb` — B1: diagnosis of sparse-scene failures.
- `B1_End_to_End_Semantic_Guided_SAM2_Checkpoint.ipynb` — B1: end-to-end semantic-guided SAM2 checkpoint.
- `B2_B0_Anchored_Semantic_Validation.ipynb` — B2: B0-anchored semantic validation.

## `03_m1_b3a_geometry_and_multiplicity/` — M1 / B3a — geometry and multiplicity (SG, GPV, hybrids)

- `M1_Marigold_Geometric_Diagnostics.ipynb` — M1: Marigold depth as a geometric prior — diagnostics.
- `B3_B0_B2_M_Geometry_Integration.ipynb` — B3: integrating geometry (depth) with the B0/B2 pipeline.
- `B3R_SG_Rebuild_and_Fixes.ipynb` — B3R: SG rebuild, fixes, environment decisions; OmniCount-191 audit (Section 5).
- `B3RK_SG_Multiplicity.ipynb` — B3RK: SG multiplicity.
- `B3G_Guided_SAM2_GPV.ipynb` — B3G: guided SAM2 prompting (GPV family).
- `B3GK_GPV_Multiplicity_and_Prompts.ipynb` — B3GK: GPV multiplicity (GPV-K) and prompt variants.
- `B3H_Hybrid_SG_GPV.ipynb` — B3H: SG–GPV hybrids (H1b).

## `04_b3b_improvement_phase/` — B3b — improvement phase (train_dev400)

- `B3S_Semantic_Improvement.ipynb` — B3S: expanded SAN vocabulary; mask verification (closed).
- `B3U_Unified_Multiplicity.ipynb` — B3U: unified multiplicity variants (closed).
- `B3I_Masked_Input_SAM2.ipynb` — B3I: masked input to SAM2.
- `B3L_SAM2_Large.ipynb` — B3L: SAM2.1-Large.
- `B3N_Negative_Point_Prompts.ipynb` — B3N: negative point prompts.
- `B3NS_Selective_Negative_Prompts.ipynb` — B3NS: selective negative prompts (closed).
- `B3MN_Marigold_Normals.ipynb` — B3MN: Marigold surface normals as a multiplicity cue (K3n).
- `B3Z_Zoomed_SAM2.ipynb` — B3Z: zoomed SAM2 automatic masks (crop_n_layers 1).
- `B3LZ_Local_Zoom.ipynb` — B3LZ: local zoom inside regions.
- `B3MX_Mix_and_Ensemble.ipynb` — B3MX: mixed mask sets (MixA) and ensembles.
- `B3PN_Part_and_Normals_Merge.ipynb` — B3PN: part-mask filter P; normals merge detector.
- `B3AC_Area_Cap_and_P_on_S.ipynb` — B3AC: area cap on K3n; P on the Small base.

## `05_b4_heldout_selection/` — B4 — held-out selection (FSC-147 val)

- `B4_HeldOut_FSC147_Validation.ipynb` — B4: first held-out selection on FSC-147 val (H1b_v1).
- `B4b_HeldOut_Caching_Candidates.ipynb` — B4b: held-out and dev100 caches for the five candidates.
- `B4c_HeldOut_Candidates.ipynb` — B4c: second, pre-declared held-out evaluation; primary GPV_R4+P[MixA], secondary GPV_K-n+P[LM0].

## `06_b5_multilabel/` — B5 — multi-label adaptation, freeze, pre-test validation

- `B5_0_OmniCount_Region_Tests.ipynb` — B5.0: multi-label design on OmniCount-191 (Parts 1–4); U_both selected.
- `B5_1_Candidates_v2_Freeze.ipynb` — B5.1: freeze of candidates_v2 and the two final pipelines (regression checks).
- `B5_2_PreTest_Validation.ipynb` — B5.2: pre-test validation (FSC-147 v1.1 ladder; OmniCount val held-out).

## `07_b6_test_evaluation/` — B6 — test evaluation (FSC-147, CARPK, OmniCount-191, PASCAL VOC 2007)

- (to be added)
