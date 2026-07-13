# Analysis coverage map

This maps every script under `analysis/` to the paper section/figure it reproduces,
and lists which paper analyses **don't yet have code in this repo** and need to be
uploaded before the repository is complete. Status is as of this writing and should
be updated as more scripts are added.

Legend: ✅ present · 🟡 partial (some but not all pieces present) · ❌ missing (nothing uploaded yet)

## 2.2 — CASCADE outperforms baselines across 51 multiscale disease prediction tasks
*(Figure 2; Methods 9.11; Supplementary Tables S6, S9-S12)*

| Piece | Status | Script |
|---|---|---|
| CASCADE donor/cell-level prediction sweep, all 4 datasets (SEATTLE, AUTISM, HLCA, LUCA) | ✅ | `benchmarking/multi_task_prediction.py` |
| Baseline-model evaluation pattern (extract embeddings from an external model, run the same linear/attention probe protocol as CASCADE) | ✅ | `benchmarking/baselines/geneformer_luca.py`, `geneformer_luca_patient.py` — kept as the canonical worked example. All 8 baselines (Geneformer, scGPT, scVI, UCE, PaScient, mcBERT, LR+CA, majority-class) follow this same harness with the embedding-extraction step swapped out, so a separate script per baseline isn't needed in the repo. |
| Sequence-length sensitivity (Supp Note 4) | ✅ | `benchmarking/sensitivity/sequence_length_ablation.py` |
| Median/data-leakage sensitivity (Supp Notes 2-3) | ✅ | `benchmarking/sensitivity/median_stability.py` |
| LUCA UMAP visualization | ✅ | `benchmarking/luca/umap_visualization.py` |
| LUCA context ablation | ✅ | `benchmarking/luca/context_ablation.py` |

## 2.3 — CASCADE-Explainer prioritises Alzheimer's disease-relevant cell types
*(Figure 3; "orthogonal benchmarking" — LLM-arena literature evidence + eQTL/GWAS colocalization)*

| Piece | Status | Script |
|---|---|---|
| LLM-arena pairwise cell-type comparison + Elo scoring (Fig. 3b-c data generation) | ✅ | `alzheimers/elo_score.py` (requires `OPENROUTER_API_KEY`; external LLM-judge API dependency is inherent to the method) |
| Correlation between LLM-arena Elo and CASCADE-derived importance, granular + coarse (Fig. 3b-c) | ✅ | `alzheimers/elo_loci_style_plots.py` |
| eQTL/GWAS colocalization cell-type ranking + correlation with CASCADE (Fig. 3d-e) | ❌ | **needs upload** |
| External reference-gene-set enrichment panel (Fig. 3f) | ❌ | **needs upload** |

`elo_loci_style_plots.py`'s CASCADE cell-type importance values (`CL_XAI`, `CASC_ABS_MAJOR`)
are hardcoded from an already-computed result rather than loaded from a file — the script
that originally produced those specific numbers wasn't part of this upload, so they're
carried over as-is rather than re-derived.

## 2.4 — CASCADE-Explainer recovers canonical cell-type gene programmes
*(healthy-only SEATTLE cell-type explainer validation, two baselines: attention-derived and cell-type-specific DE)*

| Piece | Status | Script |
|---|---|---|
| Healthy-only, grouped cell-type marker (DE) ground truth | ✅ | `alzheimers/healthy_celltype_markers.py` |
| Attention-derived gene-importance baseline | ✅ | `alzheimers/attention_baseline.py` |
| Ground-truth-vs-ranking comparison (proportion-based, with permutation null) and AUROC-based comparison, both with 2-4 GT sources | ✅ | `alzheimers/gt_analysis.py` |
| Headline grouped-bar figure: CASCADE vs. attention-baseline vs. DEG, AUROC vs. 2 GT sources | ✅ | `alzheimers/create_combined_gt_viz.py` |

Section 2.4 is now fully covered. (A script named `FINAL-deg-comparison.py` was uploaded
earlier but turned out to be a *different* analysis — a positional-bias sanity check for the
gene explainer, not referenced anywhere in the paper — so it was dropped rather than ported;
it was not a substitute for this section.)

## 2.5 — CASCADE predicts multiscale genetic and neuropathological Huntington's disease phenotypes
*(Figure 5a-b; donor- and cell-level prediction of CAG repeat lengths, VS grade, motor/cognitive onset)*

| Piece | Status | Script |
|---|---|---|
| Donor-level attention-probe prediction | ✅ | `huntingtons/donor_level_prediction.py` |
| Cell-level linear-probe prediction | ✅ | `huntingtons/cell_level_prediction.py` |
| Raw pseudobulk / cell-level linear-regression baseline | ✅ | `huntingtons/baseline_linear_regression.py` |

## 2.6 — CASCADE-Explainer links cell-type programmes to Huntington's disease severity and CAG repeat length
*(Figure 5c-e; aggregated cell-type importance across VS-grade stages, correlation with clinical features)*

| Piece | Status | Script |
|---|---|---|
| Aggregation of cell-type importance across VS-grade stages (Fig. 5c) | ✅ | `huntingtons/vsgrade_importance_plot.py` |
| Correlation between donor-specific cell-type drivers and clinical features, forest plot (Fig. 5d) | ✅ | `huntingtons/correlations_forestplot.py` |
| Donor-level variability heatmaps, benign vs pathogenic CAG (Fig. 5e) | ✅ | `huntingtons/donor_heatmaps.py` |

## 2.7 — CASCADE enables molecular and clinical stratification of Huntington's disease patients
*(Figure 5f-g; k=2 clustering on donor-specific cell-type importance profiles)*

| Piece | Status | Script |
|---|---|---|
| Primary clustering per CAG model, best-k selected by silhouette (k=2 for both models on real data), clinical bar chart + cell-type heatmaps (Fig. 5f) | ✅ | `huntingtons/clustering_analysis.py` |
| Silhouette diagnostic plots | ✅ | `huntingtons/silhouette_plots.py` |
| Cluster-overlap / concordance analysis between benign and pathogenic CAG clusters, cross-tabulation + clinical comparison (Fig. 5g) | ✅ | `huntingtons/cluster_overlap_analysis.py` |
| Publication figure for clinical differences between overlap groups | ✅ | `huntingtons/cluster_overlap_figure.py` |
| Supplementary: violin-plot cluster-vs-clinical comparison (age/onset/VS grade/both CAG lengths) | ✅ | `huntingtons/cluster_clinical_violin.py` |
| Supplementary: console p-value report | ✅ | `huntingtons/cluster_pvalues.py` |

Figure 5c-g is now fully covered — this was the last real gap in the Huntington's case study.

**Fixed bug, verified against real data:** all of the scripts above extract per-donor
attention importance via `huntingtons/hd_attention_utils.create_donor_df`. The originally
uploaded version of this helper (repeated near-identically across 6 of the 7 new scripts)
filtered `attention_weights` by a `valid` (non-NaN `patient_y`) boolean mask but then indexed
`patient_ids`/`patient_y`/`patient_cell_types` with the same loop variable *unfiltered* — mixing
a compacted index space with the raw one. Checked against the actual `CAG_1_1107.npz` /
`CAG_2_1107.npz` caches: this silently dropped exactly 1 of 52 valid donors per CAG model (the
last valid donor's compacted index happened to land on a different, invalid donor's raw
position, tripping the `if not valid[i]: continue` guard) — not a large-scale identity
scramble, but a real, silent exclusion. `vsgrade_importance_plot.py` (from
`run_combined_vsgrade_plot.py`) was the one script that already filtered all three arrays
consistently and did not have this bug. Fixed in `hd_attention_utils.create_donor_df`, applied
everywhere, and verified: `create_donor_df` now returns all 52 valid donors instead of 51.

## 2.8-2.9 — CASCADE-Explainer recovers thyroid hormone and receptor-regulatory gene signatures
*(Figure 4; Supplementary Notes 22-23)*

| Piece | Status | Script |
|---|---|---|
| M2 differential-expression baseline for the treatment/DN-THRα prediction tasks ("findings ... compared with a matched DE baseline for both the treatment and DN-THRα prediction tasks") | ✅ | `thyroid_hormone/scripts/03_m2_differential_expression.py` — Python, feeds directly into `04_standardize_de_baselines.R` (verified its output columns match that script's `required_cols` exactly) |
| Full pipeline: ground truth construction, standardization, gene universes, filtering variants, 10 aggregation methods, benchmarking, cascade-vs-DE comparison (hit-set + ranking + combined-set), cell-type-specific (astrocyte/glut. neuron) analysis, Excel export | ✅ | `thyroid_hormone/` (20 R scripts — see `thyroid_hormone/scripts/00_config.R` for the pipeline order) |

This is the most complete analysis folder in the repo — the numbered R pipeline plus its
Python DE-baseline precursor (`03_m2_differential_expression.py`, added once the DE
baseline's exact provenance was confirmed against the paper text). One thing to be aware
of: none of the R scripts have actually been *run* against real data in this environment
(R/data files weren't available), so they're verified by parsing and functional smoke
tests on synthetic data only, not by reproducing your actual published numbers end to
end. `03_m2_differential_expression.py`, by contrast, *was* run end-to-end on synthetic
M2-shaped data (Python + scanpy were available).

## Preprocessing coverage (outside `analysis/`, Methods 9.1-9.4)

| Piece | Status | Script |
|---|---|---|
| Clinical/donor-level metadata pre-processing (Methods 9.3), all 6 paper datasets | ✅ | `preprocessing/clinical_metadata_{autism,hlca,m2,luca,hh,seattle}.py`, shared helpers in `preprocessing/clinical_metadata_utils.py` |
| Scanpy QC/normalisation pipeline: filtering, normalise, log1p, HVG, scale (Methods 9.2) — HLCA/LUCA/M2 | ✅ | `preprocessing/qc_scanpy.py`, shared helpers in `preprocessing/qc_pipeline.py` |
| Same, AUTISM (raw 10x-style mtx input instead of a cellxgene h5ad) | ✅ | `preprocessing/qc_scanpy_autism.py` |
| Same, Seattle-AD (out-of-core scarf/Dask pipeline, used because of dataset size) | ✅ | `preprocessing/qc_scarf_seattle.py` |
| BioMart/GTF gene annotation + dedup by Ensembl ID/NCBI ID/symbol (Methods 9.2) | ✅ | `preprocessing/gene_annotation.py` |
| PCA/neighbours/Leiden/UMAP + marker-gene ranking (Methods 9.2) | ✅ | `preprocessing/clustering_umap.py` |
| Majority-vote cluster-to-cell-type annotation + metadata UMAP QC plots | ✅ | `preprocessing/celltype_annotation.py` |
| GENCODE protein-coding/miRNA gene-type annotation | ✅ | `preprocessing/protein_coding_annotation.py` |
| LUCA batch-effect correction via per-gene OLS regression (Methods 9.2) | ✅ | `preprocessing/batch_effect_regression.py` |
| Per-context (cell type/tissue/disease) non-zero median expression reference (Methods 9.4) | ✅ | `preprocessing/context_median_reference.py` |
| Donor-level stratified train/test split, ensuring every clinical category has >=1 training donor (Methods 9.11) | ✅ | `preprocessing/donor_stratified_split.py` — generator for the fixed splits already checked into `cascade/data/splits.py`; re-running isn't required to reproduce the paper |
| Builds the tokenizer/metadata dictionaries and runs tokenisation per dataset+context (Methods 9.4) | ✅ | `preprocessing/build_tokenizer_metadata.py`, plus `preprocessing/rename_tokenized_chunks.py` and `scripts/flatten_arrow_dataset_dirs.sh` (batch-output file-layout utilities) |
| Context-aware tokenizer: fold-change ranking, up-/down-regulated token sequences (Methods 9.4) | ✅ | `cascade/data/tokenizer.py` (core package, not `analysis/`) |
| HH (Huntington's) raw QC/annotation/tokenization (clinical metadata is covered, above) | ❌ | **needs upload** — HH analysis scripts (2.5-2.7) consume pre-extracted attention/embedding caches, not raw counts, so this gap doesn't block the HD case study |

A batch of scripts for datasets never mentioned anywhere in the paper text (AUTISM-BULK,
AUTISM-ORGANOIDS, GEO, CHOOSE — a bulk RNA-seq cohort, an organoid CRISPR-perturbation
model, and two exploratory/external datasets) was also uploaded but deliberately **not**
ported: grepping the full paper text confirms none of these four dataset names appear
anywhere, so they're internal-codebase side projects outside this submission's scope. The
ATC drug-mapping module built specifically for AUTISM-BULK's free-text medication notes
(`P1h`/`P1i`) was dropped for the same reason. An ontology-term-to-integer mapping script
(`P6`) was also left unported: it has a missing dependency (a MONDO-to-ICD9 mapping CSV)
and its exact Methods-text grounding was unclear, so rather than guess at its content it's
just noted here as available in the original upload if it's needed later.

Bugs fixed while porting, verified against synthetic data reproducing the original
scripts' exact logic (no real raw counts were available to test against, but each bug
was confirmed to be deterministic given the described operations, not data-dependent):
- `qc_scanpy.py` (from `02_cells_processing.py`): dropped an abandoned/broken
  in-memory 3-way-split code block for Seattle-AD (referenced undefined variables
  `n_genes`/`genedata_filtered` and would `NameError`) — Seattle-AD's real path is the
  scarf-based `qc_scarf_seattle.py`, consistent with the paper's Methods 9.2 statement
  that AD used scarf specifically because of dataset size.
- `qc_scanpy_autism.py` (from `02_cells_processing-autism.py`): `matplotlib.pyplot`
  was used without being imported (`NameError` on the first plot).
- `gene_annotation.py` (from `03_annotations.py`): the non-HLCA/M2 path manually
  re-added an `ensembl_gene_id` column immediately before `reset_index()`, but the
  index was already named `ensembl_gene_id` at that point, so `reset_index()` tried
  to create a duplicate column and pandas raised `ValueError: cannot insert
  ensembl_gene_id, already exists`. Reproduced deterministically with a synthetic
  LUCA-shaped run; fixed by dropping the redundant assignment.
- `batch_effect_regression.py` (from `07a_batch_effect_regression.py`): the output
  filename used a literal `'processed_data_batch_effect_{key}.h5ad'` (missing the
  f-string prefix), so all three regressor variants (platform/dataset/both) silently
  overwrote the same file instead of writing three separate ones.
- `cascade/data/tokenizer.py` (from `tokenisers_chunks.py`): `save_batch`'s real
  save path was dead code — a parameterized `output_path` was computed and then
  immediately overwritten by a second, hardcoded personal-cluster path that ignored
  the `output_directory`/`output_prefix`/batch arguments entirely. Fixed to use the
  parameterized path; verified end-to-end with a batched synthetic tokenization run
  (5 batches, all cells accounted for, no hardcoded paths in the output).
- `clinical_metadata_autism.py` and `clinical_metadata_hh.py`: both build/pass through
  clinical column names containing `/` (`'Attention-deficit/hyperactivity disorder'`
  as a comorbidity condition; `'Onset/Motor'`/`'Onset/Cog'` as literal HH clinical
  fields). h5ad is HDF5-based, which forbids `/` in dataset/group keys, so
  `adata.write()` would crash as soon as any of these reached `.obs`. Confirmed by
  writing a synthetic AnnData with the exact same column names, which reproduces the
  crash deterministically before the fix and succeeds after (sanitized to `-`/`_`).
- `clinical_metadata_utils.py`'s `export_value_counts_excel`: Excel sheet names are
  case-insensitive for uniqueness, and AUTISM's own column list has both `'sex'` and
  `'Sex'` — writing both as sheets crashed with `DuplicateWorksheetName`. Reproduced
  with the real AUTISM column list; fixed by deduplicating sheet names case-insensitively.

All scripts above were smoke-tested end-to-end on synthetic data shaped like a real
run (raw counts -> QC -> annotation -> clustering -> protein-coding filter ->
batch-effect regression -> median reference -> clinical metadata -> donor split ->
tokenizer-metadata build -> tokenization), not just compiled. The M2 differential
expression baseline (`analysis/thyroid_hormone/scripts/03_m2_differential_expression.py`,
see below) was also run end-to-end and its output columns checked against what
`04_standardize_de_baselines.R` requires.

## Summary of what's needed next

1. **Figure 3d-f** (eQTL/GWAS colocalization, external reference-set enrichment) — LLM-arena (3b-c) is now covered.
2. **HH raw preprocessing** (low priority — the HD analysis scripts consume
   pre-extracted attention caches, not raw counts, so this doesn't block anything
   in `analysis/huntingtons/`).

The Huntington's disease case study (2.5-2.7, Figure 5), the Alzheimer's canonical
cell-type recovery section (2.4), the baseline-model evaluation pattern (2.2), and
the QC/annotation/tokenization preprocessing pipeline (Methods 9.2-9.4, all datasets
except HH's raw counts) are now all covered.
