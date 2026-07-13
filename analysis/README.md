# Reproducing the paper's analyses

This directory contains the code to reproduce the analyses and figures in the CASCADE
paper. Sections below are organized by figure, in the order you'd run them — each
entry names the script(s), the order to run them in, and what each step produces.

Before running anything here, follow the [main README](../README.md) for installation
and set `CASCADE_DATA_ROOT`/`CASCADE_CKPT_ROOT` to point at your copy of the datasets
and trained checkpoints (see [Data](../README.md#data)). Pretrained checkpoints
themselves are produced by `cascade/training/` (see `scripts/*.sh` for SLURM launch
examples) — the analyses below all start from a *frozen* checkpoint or its extracted
embeddings, not from raw data.

## Preprocessing (Methods 9.2-9.4)

Needed once per dataset, before training or any downstream analysis. Run in this
order (all entrypoints are `python -m preprocessing.<script>`, see each script's
docstring for full argument lists):

1. **QC and normalisation** (Methods 9.2) — filter low-quality cells/genes,
   normalise, log-transform, select highly-variable genes, scale.
   - HLCA / LUCA / M2 (mouse-thyroid): [`qc_scanpy.py`](../preprocessing/qc_scanpy.py)
     `--dataset-name LUCA --input-path .../adata_raw.h5ad --output-dir .../LUCA`
   - AUTISM (raw 10x-style mtx input): [`qc_scanpy_autism.py`](../preprocessing/qc_scanpy_autism.py)
   - Seattle-AD (out-of-core scarf/Dask, used because of dataset size): [`qc_scarf_seattle.py`](../preprocessing/qc_scarf_seattle.py)
2. **Gene annotation** — BioMart/GTF lookup, dedup by Ensembl ID/NCBI ID/symbol:
   [`gene_annotation.py`](../preprocessing/gene_annotation.py)
3. **Clustering** — PCA/neighbours/Leiden/UMAP + marker-gene ranking:
   [`clustering_umap.py`](../preprocessing/clustering_umap.py), then majority-vote
   cluster-to-cell-type labelling: [`celltype_annotation.py`](../preprocessing/celltype_annotation.py)
4. **Protein-coding filter** — GENCODE gene-type annotation:
   [`protein_coding_annotation.py`](../preprocessing/protein_coding_annotation.py)
5. **Batch-effect correction** (LUCA only) — per-gene OLS regression on
   platform/dataset covariates: [`batch_effect_regression.py`](../preprocessing/batch_effect_regression.py)
6. **Clinical/donor-level metadata** (Methods 9.3) — one script per dataset:
   [`clinical_metadata_autism.py`](../preprocessing/clinical_metadata_autism.py),
   [`clinical_metadata_hlca.py`](../preprocessing/clinical_metadata_hlca.py),
   [`clinical_metadata_m2.py`](../preprocessing/clinical_metadata_m2.py),
   [`clinical_metadata_luca.py`](../preprocessing/clinical_metadata_luca.py),
   [`clinical_metadata_hh.py`](../preprocessing/clinical_metadata_hh.py),
   [`clinical_metadata_seattle.py`](../preprocessing/clinical_metadata_seattle.py)
7. **Donor-level stratified split** (Methods 9.11) — generates the train/test donor
   lists already checked into `cascade/data/splits.py`; only needed if you want to
   regenerate them: [`donor_stratified_split.py`](../preprocessing/donor_stratified_split.py)
8. **Per-context median expression reference** (Methods 9.4) — non-zero median
   expression per cell type/tissue/disease, used to compute each cell's context-specific
   fold-change ranking: [`context_median_reference.py`](../preprocessing/context_median_reference.py)
9. **Tokenisation** (Methods 9.4) — builds the tokenizer/metadata dictionaries and
   runs `cascade.data.tokenizer.TranscriptomeTokenizer`:
   [`build_tokenizer_metadata.py`](../preprocessing/build_tokenizer_metadata.py)
   `--dataset-name LUCA --context CELLS --clinical-dir ... --data-dir ... --output-dir ... --output-prefix tokenized_data_CELLS`,
   then flatten the batched output with
   [`rename_tokenized_chunks.py`](../preprocessing/rename_tokenized_chunks.py) and
   [`../scripts/flatten_arrow_dataset_dirs.sh`](../scripts/flatten_arrow_dataset_dirs.sh).

HH (Huntington's) raw QC/annotation/tokenization is not yet included (only its
clinical-metadata step is) — the Huntington's analyses below consume pre-extracted
attention/embedding caches rather than raw counts, so this doesn't block anything.

## Figure 2 (CASCADE outperforms baselines across 51 multiscale disease prediction tasks)
*(Methods 9.11; Supplementary Tables S6, S9-S12)*

**Main prediction sweep**: donor- and cell-level prediction across all 4 benchmark
datasets (SEATTLE, AUTISM, HLCA, LUCA), against a `PatientAggregator`-based attention
model (donor-level) or linear probe (cell-level):
[`benchmarking/multi_task_prediction.py`](benchmarking/multi_task_prediction.py)
```
python -m analysis.benchmarking.multi_task_prediction --dataset SEATTLE
python -m analysis.benchmarking.multi_task_prediction --dataset all --output-dir results/
```

**Baselines**: extract embeddings from an external model and run the same probe
protocol as CASCADE. [`benchmarking/baselines/geneformer_luca.py`](benchmarking/baselines/geneformer_luca.py)
and [`geneformer_luca_patient.py`](benchmarking/baselines/geneformer_luca_patient.py)
are the worked example — all 8 baselines (Geneformer, scGPT, scVI, UCE, PaScient,
mcBERT, LR+CA, majority-class) follow this same harness with the embedding-extraction
step swapped out:
```
python -m analysis.benchmarking.baselines.geneformer_luca --seed 1 \
    --geneformer-h5ad $CASCADE_DATA_ROOT/LUCA/BASELINE/geneformer/adata_geneformer.h5ad
```

**Sensitivity analyses** (supplementary, run independently of the main sweep):
- Sequence-length ablation (Supp Note 4): [`benchmarking/sensitivity/sequence_length_ablation.py`](benchmarking/sensitivity/sequence_length_ablation.py) `--dataset LUCA`
- Median/data-leakage stability (Supp Notes 2-3): [`benchmarking/sensitivity/median_stability.py`](benchmarking/sensitivity/median_stability.py) `--h5ad /path/to/data.h5ad --output-dir ./results`

**LUCA-specific supplementary panels**:
[`benchmarking/luca/umap_visualization.py`](benchmarking/luca/umap_visualization.py) and
[`benchmarking/luca/context_ablation.py`](benchmarking/luca/context_ablation.py).

## Figure 3 (CASCADE-Explainer prioritises and validates Alzheimer's disease-relevant cell types)

**Panels b-c — LLM-arena orthogonal benchmarking**: pairwise cell-type comparisons
judged by an LLM, scored with Elo, correlated against CASCADE-derived importance.
1. [`alzheimers/elo_score.py`](alzheimers/elo_score.py) — generates the pairwise
   comparisons and Elo scores (requires `OPENROUTER_API_KEY`):
   ```
   python -m analysis.alzheimers.elo_score \
       --output-dir $CASCADE_DATA_ROOT/SEATTLE/gene_explainer_validation
   ```
   Output naming matches what step 2 expects to read
   (`results_{intermediate,major}_{ad,control}_elo_elo.csv`).
2. [`alzheimers/elo_loci_style_plots.py`](alzheimers/elo_loci_style_plots.py) — the
   correlation plots themselves:
   ```
   python -m analysis.alzheimers.elo_loci_style_plots \
       --elo-dir $CASCADE_DATA_ROOT/SEATTLE/gene_explainer_validation \
       --output-dir $CASCADE_DATA_ROOT/SEATTLE/gene_explainer_validation
   ```

**Panels d-f — eQTL/GWAS colocalization and external reference-set enrichment**: not
yet included.

**Canonical cell-type gene-programme recovery** (healthy-only SEATTLE validation,
restricted to non-diseased participants to rule out disease confounding):
1. [`alzheimers/healthy_celltype_markers.py`](alzheimers/healthy_celltype_markers.py) —
   builds the grouped cell-type marker (DE) ground truth from healthy donors:
   ```
   python -m analysis.alzheimers.healthy_celltype_markers \
       --h5ad-dir $CASCADE_DATA_ROOT/SEATTLE/adata_objects \
       --output-dir $CASCADE_DATA_ROOT/SEATTLE/healthy_grouped_cell_type_markers \
       --cell-type-col cell_type --health-col disease --healthy-values 0 \
       --gene-col ensembl_gene_id --h5ad-pattern "adata_annotated_protein_clinical_fin_*.h5ad"
   ```
2. [`alzheimers/attention_baseline.py`](alzheimers/attention_baseline.py) — the
   attention-derived gene-importance baseline:
   ```
   python -m analysis.alzheimers.attention_baseline \
       --output_path ./attention_outputs/attention_baseline_SEATTLE_cell_type.pkl \
       --dataset_name SEATTLE --task cell_type \
       --transformer_checkpoint /path/to/checkpoint.pt
   ```
3. [`alzheimers/gt_analysis.py`](alzheimers/gt_analysis.py) — compares CASCADE,
   attention, and DE rankings against the ground truth (proportion-based and
   AUROC-based, both with 2-4 ground-truth sources):
   ```
   python -m analysis.alzheimers.gt_analysis \
       --data-dir $CASCADE_DATA_ROOT/SEATTLE/gene_explainer_validation \
       --ground-truth-sources txt paper
   ```
4. [`alzheimers/create_combined_gt_viz.py`](alzheimers/create_combined_gt_viz.py) —
   the headline grouped-bar figure (CASCADE vs. attention-baseline vs. DEG, AUROC vs.
   2 ground-truth sources):
   ```
   python -m analysis.alzheimers.create_combined_gt_viz \
       --data-dir $CASCADE_DATA_ROOT/SEATTLE/gene_explainer_validation \
       --cascade-ranks $CASCADE_DATA_ROOT/SEATTLE/median_gene_ranks_corrected.csv
   ```

## Figure 4 (CASCADE-Explainer recovers thyroid hormone and receptor-regulatory gene signatures)
*(Supplementary Notes 22-23)*

Mixed Python + R pipeline. Run the Python DE baseline first, then the numbered R
pipeline (each R script is self-documenting — see its header comment for inputs/outputs).

1. **M2 differential-expression baseline** for the treatment/DN-THRα prediction
   tasks: [`thyroid_hormone/scripts/03_m2_differential_expression.py`](thyroid_hormone/scripts/03_m2_differential_expression.py)
   ```
   python -m analysis.thyroid_hormone.scripts.03_m2_differential_expression \
       --h5ad-path $CASCADE_DATA_ROOT/M2/adata_objects/adata_annotated_protein_coding_clinical.h5ad
   ```
   Produces the `de_pooled_*.csv`/`de_cell_type_specific_*.csv` files consumed by
   step 2 below.
2. **R pipeline** (ground truth construction, standardisation, gene universes,
   filtering variants, 10 aggregation methods, benchmarking, CASCADE-vs-DE comparison,
   cell-type-specific analysis, Excel export) — 20 scripts under
   [`thyroid_hormone/scripts/`](thyroid_hormone/scripts/), starting from
   [`00_config.R`](thyroid_hormone/scripts/00_config.R) (source it at the top of
   every downstream script) and running in numeric-prefix order.

## Figure 5 (Huntington's disease case study)

**Panels a-b — multiscale phenotype prediction** (CAG repeat lengths, VS grade,
motor/cognitive onset):
- Donor-level attention-probe prediction: [`huntingtons/donor_level_prediction.py`](huntingtons/donor_level_prediction.py) `--embeddings-path /path/to/embeddings.pkl`
- Cell-level linear-probe prediction: [`huntingtons/cell_level_prediction.py`](huntingtons/cell_level_prediction.py) `--embeddings-path /path/to/embeddings.pkl`
- Raw pseudobulk/cell-level linear-regression baseline: [`huntingtons/baseline_linear_regression.py`](huntingtons/baseline_linear_regression.py) `--adata-path /path/to/HH.h5ad`

**Panels c-e — cell-type programmes vs. severity and CAG repeat length** (all read the
`HD_models_scale/{CAG_1,CAG_2,VSGRADE}_1107.npz` attention caches):
- Aggregated cell-type importance across VS-grade stages (5c): [`huntingtons/vsgrade_importance_plot.py`](huntingtons/vsgrade_importance_plot.py) `--vsgrade-npz .../VSGRADE_1107.npz --output-dir .`
- Correlation between donor-specific cell-type drivers and clinical features, forest
  plot (5d): [`huntingtons/correlations_forestplot.py`](huntingtons/correlations_forestplot.py)
- Donor-level variability heatmaps, benign vs. pathogenic CAG (5e): [`huntingtons/donor_heatmaps.py`](huntingtons/donor_heatmaps.py)

**Panels f-g — molecular and clinical patient stratification** (k=2 clustering on
donor-specific cell-type importance profiles), run in this order:
1. [`huntingtons/silhouette_plots.py`](huntingtons/silhouette_plots.py) — diagnostic
   plots for selecting k.
2. [`huntingtons/clustering_analysis.py`](huntingtons/clustering_analysis.py) — primary
   clustering per CAG model, clinical bar chart + cell-type heatmaps (5f).
3. [`huntingtons/cluster_overlap_analysis.py`](huntingtons/cluster_overlap_analysis.py) —
   concordance between benign and pathogenic CAG clusters, cross-tabulation + clinical
   comparison (5g), followed by [`huntingtons/cluster_overlap_figure.py`](huntingtons/cluster_overlap_figure.py)
   `--overlap-csv cluster_overlap_donors.csv --output-dir .` for the publication figure.
4. Supplementary: [`huntingtons/cluster_clinical_violin.py`](huntingtons/cluster_clinical_violin.py)
   (violin-plot cluster-vs-clinical comparison) and [`huntingtons/cluster_pvalues.py`](huntingtons/cluster_pvalues.py)
   (console p-value report).

All four scripts in this group take the same `--cag1-npz`/`--cag2-npz` arguments,
e.g. `HD_models_scale/CAG_1_1107.npz` / `HD_models_scale/CAG_2_1107.npz`.
