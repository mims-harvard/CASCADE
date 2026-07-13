# Reproducing the paper's analyses

This directory contains the code to reproduce the analyses and figures in the CASCADE
paper. Tables below are organized by figure, ordered top-to-bottom the way you'd run
them. See each script's own docstring for the full argument list — commands below show
only the required/most relevant flags.

Before running anything here, follow the [main README](../README.md) for installation
and set `CASCADE_DATA_ROOT`/`CASCADE_CKPT_ROOT` to point at your copy of the datasets
and trained checkpoints (see [Data](../README.md#data)). Pretrained checkpoints
themselves are produced by `cascade/training/` (see `scripts/*.sh` for SLURM launch
examples) — the analyses below all start from a *frozen* checkpoint or its extracted
embeddings, not from raw data.

## Preprocessing (Methods 9.2-9.4)

Needed once per dataset, before training or any downstream analysis. Run top-to-bottom.

| # | Step | Script | Run |
|---|---|---|---|
| 1 | QC & normalisation — HLCA / LUCA / M2 | [`qc_scanpy.py`](../preprocessing/qc_scanpy.py) | `python -m preprocessing.qc_scanpy --dataset-name LUCA --input-path .../adata_raw.h5ad --output-dir .../LUCA` |
| 1 | QC & normalisation — AUTISM (raw 10x-style mtx) | [`qc_scanpy_autism.py`](../preprocessing/qc_scanpy_autism.py) | `python -m preprocessing.qc_scanpy_autism --input-dir $CASCADE_DATA_ROOT/raw/AUTISM --output-dir $CASCADE_DATA_ROOT/processed/AUTISM` |
| 1 | QC & normalisation — Seattle-AD (out-of-core scarf/Dask; used because of dataset size) | [`qc_scarf_seattle.py`](../preprocessing/qc_scarf_seattle.py) | `python -m preprocessing.qc_scarf_seattle --h5ad-path .../adata_raw.h5ad --zarr-path .../adata_raw.zarr --biomart-file .../biomart_cleaned.csv --gencode-file .../gencode....gff3 --output-dir .../SEA_AD` |
| 2 | Gene annotation — BioMart/GTF lookup, dedup by Ensembl ID/NCBI ID/symbol | [`gene_annotation.py`](../preprocessing/gene_annotation.py) | `python -m preprocessing.gene_annotation --dataset-name LUCA --input-path .../adata_scaled.h5ad --biomart-file .../results.txt` |
| 3 | Clustering — PCA/neighbours/Leiden/UMAP + marker genes | [`clustering_umap.py`](../preprocessing/clustering_umap.py) | `python -m preprocessing.clustering_umap --dataset-name LUCA --input-path .../adata_annotated.h5ad --output-dir .../LUCA` |
| 3 | Cell-type labelling — majority-vote cluster-to-cell-type | [`celltype_annotation.py`](../preprocessing/celltype_annotation.py) | `python -m preprocessing.celltype_annotation --dataset-name LUCA --input-path .../adata_marker_genes.h5ad --output-dir .../LUCA` |
| 4 | Protein-coding filter — GENCODE gene-type annotation | [`protein_coding_annotation.py`](../preprocessing/protein_coding_annotation.py) | `python -m preprocessing.protein_coding_annotation --dataset-name LUCA --input-path .../adata_marker_genes.h5ad --gencode-file .../gencode....gff3` |
| 5 | Batch-effect correction (LUCA only) — per-gene OLS on platform/dataset covariates | [`batch_effect_regression.py`](../preprocessing/batch_effect_regression.py) | `python -m preprocessing.batch_effect_regression --input-path .../adata_annotated_protein_coding.h5ad --output-dir .../LUCA` |
| 6 | Clinical/donor-level metadata (Methods 9.3) — AUTISM | [`clinical_metadata_autism.py`](../preprocessing/clinical_metadata_autism.py) | `python -m preprocessing.clinical_metadata_autism --input-path .../adata_annotated_protein_coding.h5ad --output-path .../adata_annotated_protein_coding_clinical.h5ad` |
| 6 | Clinical/donor-level metadata — HLCA | [`clinical_metadata_hlca.py`](../preprocessing/clinical_metadata_hlca.py) | `python -m preprocessing.clinical_metadata_hlca --input-path .../adata_annotated_protein_coding.h5ad --output-path .../adata_annotated_protein_coding_clinical.h5ad` |
| 6 | Clinical/donor-level metadata — M2 | [`clinical_metadata_m2.py`](../preprocessing/clinical_metadata_m2.py) | `python -m preprocessing.clinical_metadata_m2 --input-path .../adata_annotated_protein_coding.h5ad --output-path .../adata_annotated_protein_coding_clinical.h5ad` |
| 6 | Clinical/donor-level metadata — LUCA | [`clinical_metadata_luca.py`](../preprocessing/clinical_metadata_luca.py) | `python -m preprocessing.clinical_metadata_luca --batch-corrected-path .../processed_data_batch_effect_both.h5ad --annotated-path .../adata_annotated_protein_coding.h5ad` |
| 6 | Clinical/donor-level metadata — HH (Huntington's) | [`clinical_metadata_hh.py`](../preprocessing/clinical_metadata_hh.py) | `python -m preprocessing.clinical_metadata_hh --input-path .../adata_annotated.h5ad --output-path .../adata_annotated_protein_coding_clinical.h5ad` |
| 6 | Clinical/donor-level metadata — Seattle-AD | [`clinical_metadata_seattle.py`](../preprocessing/clinical_metadata_seattle.py) | `python -m preprocessing.clinical_metadata_seattle --annotations-csv .../SEATTLE_full_annotations.csv --chunks-dir .../SEA_AD` |
| 7 | Donor-level stratified split (Methods 9.11) — regenerates the fixed splits already in `cascade/data/splits.py` | [`donor_stratified_split.py`](../preprocessing/donor_stratified_split.py) | `python -m preprocessing.donor_stratified_split --dataset-name LUCA --input-path .../adata_annotated_protein_coding_clinical.h5ad` |
| 8 | Per-context median expression reference — feeds the tokenizer's fold-change ranking | [`context_median_reference.py`](../preprocessing/context_median_reference.py) | `python -m preprocessing.context_median_reference --dataset-name LUCA --input-path .../adata_annotated_protein_coding.h5ad --output-dir .../LUCA` |
| 9 | Tokenisation — builds the tokenizer/metadata dictionaries and runs `TranscriptomeTokenizer` | [`build_tokenizer_metadata.py`](../preprocessing/build_tokenizer_metadata.py) | `python -m preprocessing.build_tokenizer_metadata --dataset-name LUCA --context CELLS --clinical-dir ... --data-dir ... --output-dir ... --output-prefix tokenized_data_CELLS` |
| 9 | Tokenisation — flatten the batched output | [`rename_tokenized_chunks.py`](../preprocessing/rename_tokenized_chunks.py), [`flatten_arrow_dataset_dirs.sh`](../scripts/flatten_arrow_dataset_dirs.sh) | `python -m preprocessing.rename_tokenized_chunks --dataset-dir .../CELLS` then `bash flatten_arrow_dataset_dirs.sh` |

HH (Huntington's) raw QC/annotation/tokenization is not yet included (only its
clinical-metadata step is) — the Huntington's analyses below consume pre-extracted
attention/embedding caches rather than raw counts, so this doesn't block anything.

## Figure 2 — CASCADE outperforms baselines across 51 multiscale disease prediction tasks
*(Methods 9.11; Supplementary Tables S6, S9-S12)*

| Analysis | Script | Run |
|---|---|---|
| Main prediction sweep: donor-/cell-level prediction across all 4 benchmark datasets (SEATTLE, AUTISM, HLCA, LUCA) | [`multi_task_prediction.py`](benchmarking/multi_task_prediction.py) | `python -m analysis.benchmarking.multi_task_prediction --dataset all --output-dir results/` |
| Baselines: extract embeddings from an external model, run the same probe protocol as CASCADE. Worked example for Geneformer/LUCA — all 8 baselines (Geneformer, scGPT, scVI, UCE, PaScient, mcBERT, LR+CA, majority-class) follow this same harness with the embedding-extraction step swapped out | [`baselines/geneformer_luca.py`](benchmarking/baselines/geneformer_luca.py), [`geneformer_luca_patient.py`](benchmarking/baselines/geneformer_luca_patient.py) | `python -m analysis.benchmarking.baselines.geneformer_luca --seed 1 --geneformer-h5ad .../adata_geneformer.h5ad` |
| Sequence-length sensitivity (Supp Note 4) | [`sensitivity/sequence_length_ablation.py`](benchmarking/sensitivity/sequence_length_ablation.py) | `python -m analysis.benchmarking.sensitivity.sequence_length_ablation --dataset LUCA` |
| Median/data-leakage sensitivity (Supp Notes 2-3) | [`sensitivity/median_stability.py`](benchmarking/sensitivity/median_stability.py) | `python -m analysis.benchmarking.sensitivity.median_stability --h5ad /path/to/data.h5ad --output-dir ./results` |
| LUCA UMAP visualization (supplementary) | [`luca/umap_visualization.py`](benchmarking/luca/umap_visualization.py) | `python -m analysis.benchmarking.luca.umap_visualization --embeddings-file /path/to/embeddings.pkl --output-dir .` |
| LUCA context ablation (supplementary) | [`luca/context_ablation.py`](benchmarking/luca/context_ablation.py) | `python -m analysis.benchmarking.luca.context_ablation --embeddings-base /path/to/luca_embeddings_dir --output-dir .` |

## Figure 3 — CASCADE-Explainer prioritises and validates Alzheimer's disease-relevant cell types

**Panels b-c — LLM-arena orthogonal benchmarking**: pairwise cell-type comparisons judged by an LLM, scored with Elo, correlated against CASCADE-derived importance.

| # | Analysis | Script | Run |
|---|---|---|---|
| 1 | Pairwise LLM comparisons + Elo scoring (requires `OPENROUTER_API_KEY`); output naming matches what step 2 reads (`results_{intermediate,major}_{ad,control}_elo_elo.csv`) | [`elo_score.py`](alzheimers/elo_score.py) | `python -m analysis.alzheimers.elo_score --output-dir $CASCADE_DATA_ROOT/SEATTLE/gene_explainer_validation` |
| 2 | Correlation plots: LLM-arena Elo vs. CASCADE importance | [`elo_loci_style_plots.py`](alzheimers/elo_loci_style_plots.py) | `python -m analysis.alzheimers.elo_loci_style_plots --elo-dir $CASCADE_DATA_ROOT/SEATTLE/gene_explainer_validation --output-dir $CASCADE_DATA_ROOT/SEATTLE/gene_explainer_validation` |

**Panels d-f — eQTL/GWAS colocalization and external reference-set enrichment**: not yet included.

**Canonical cell-type gene-programme recovery** (healthy-only SEATTLE validation, restricted to non-diseased participants to rule out disease confounding):

| # | Analysis | Script | Run |
|---|---|---|---|
| 1 | Grouped cell-type marker (DE) ground truth from healthy donors | [`healthy_celltype_markers.py`](alzheimers/healthy_celltype_markers.py) | `python -m analysis.alzheimers.healthy_celltype_markers --h5ad-dir .../adata_objects --output-dir .../healthy_grouped_cell_type_markers --cell-type-col cell_type --health-col disease --healthy-values 0 --gene-col ensembl_gene_id` |
| 2 | Attention-derived gene-importance baseline | [`attention_baseline.py`](alzheimers/attention_baseline.py) | `python -m analysis.alzheimers.attention_baseline --output_path .../attention_baseline_SEATTLE_cell_type.pkl --dataset_name SEATTLE --task cell_type --transformer_checkpoint /path/to/checkpoint.pt` |
| 3 | CASCADE vs. attention vs. DE rankings against ground truth (proportion-based + AUROC-based, 2-4 GT sources) | [`gt_analysis.py`](alzheimers/gt_analysis.py) | `python -m analysis.alzheimers.gt_analysis --data-dir $CASCADE_DATA_ROOT/SEATTLE/gene_explainer_validation --ground-truth-sources txt paper` |
| 4 | Headline grouped-bar figure: CASCADE vs. attention-baseline vs. DEG, AUROC vs. 2 GT sources | [`create_combined_gt_viz.py`](alzheimers/create_combined_gt_viz.py) | `python -m analysis.alzheimers.create_combined_gt_viz --data-dir $CASCADE_DATA_ROOT/SEATTLE/gene_explainer_validation --cascade-ranks .../median_gene_ranks_corrected.csv` |

## Figure 4 — CASCADE-Explainer recovers thyroid hormone and receptor-regulatory gene signatures
*(Supplementary Notes 22-23)*

Mixed Python + R pipeline: run the Python DE baseline first, then the numbered R
pipeline (each R script is self-documenting — see its header comment for inputs/outputs).

| # | Analysis | Script | Run |
|---|---|---|---|
| 1 | M2 differential-expression baseline for the treatment/DN-THRα prediction tasks; produces the `de_pooled_*.csv`/`de_cell_type_specific_*.csv` files step 2 consumes | [`03_m2_differential_expression.py`](thyroid_hormone/scripts/03_m2_differential_expression.py) | `python -m analysis.thyroid_hormone.scripts.03_m2_differential_expression --h5ad-path .../adata_annotated_protein_coding_clinical.h5ad` |
| 2 | R pipeline: ground truth construction, standardisation, gene universes, filtering variants, 10 aggregation methods, benchmarking, CASCADE-vs-DE comparison, cell-type-specific analysis, Excel export — 20 scripts, numeric-prefix order | [`thyroid_hormone/scripts/`](thyroid_hormone/scripts/) | `source("scripts/00_config.R")` at the top of every downstream script, then run in numeric-prefix order |

## Figure 5 — Huntington's disease case study

**Panels a-b — multiscale phenotype prediction** (CAG repeat lengths, VS grade, motor/cognitive onset):

| Analysis | Script | Run |
|---|---|---|
| Donor-level attention-probe prediction | [`donor_level_prediction.py`](huntingtons/donor_level_prediction.py) | `python -m analysis.huntingtons.donor_level_prediction --embeddings-path /path/to/embeddings.pkl` |
| Cell-level linear-probe prediction | [`cell_level_prediction.py`](huntingtons/cell_level_prediction.py) | `python -m analysis.huntingtons.cell_level_prediction --embeddings-path /path/to/embeddings.pkl` |
| Raw pseudobulk/cell-level linear-regression baseline | [`baseline_linear_regression.py`](huntingtons/baseline_linear_regression.py) | `python -m analysis.huntingtons.baseline_linear_regression --adata-path /path/to/HH.h5ad` |

**Panels c-e — cell-type programmes vs. severity and CAG repeat length** (all read the `HD_models_scale/{CAG_1,CAG_2,VSGRADE}_1107.npz` attention caches):

| Analysis | Script | Run |
|---|---|---|
| Aggregated cell-type importance across VS-grade stages (5c) | [`vsgrade_importance_plot.py`](huntingtons/vsgrade_importance_plot.py) | `python -m analysis.huntingtons.vsgrade_importance_plot --vsgrade-npz .../VSGRADE_1107.npz --output-dir .` |
| Correlation between donor-specific cell-type drivers and clinical features, forest plot (5d) | [`correlations_forestplot.py`](huntingtons/correlations_forestplot.py) | `python -m analysis.huntingtons.correlations_forestplot --cag1-npz .../CAG_1_1107.npz --cag2-npz .../CAG_2_1107.npz` |
| Donor-level variability heatmaps, benign vs. pathogenic CAG (5e) | [`donor_heatmaps.py`](huntingtons/donor_heatmaps.py) | `python -m analysis.huntingtons.donor_heatmaps --cag1-npz .../CAG_1_1107.npz --cag2-npz .../CAG_2_1107.npz --output-dir .` |

**Panels f-g — molecular and clinical patient stratification** (k=2 clustering on donor-specific cell-type importance profiles), run in this order — all take the same `--cag1-npz`/`--cag2-npz` arguments (e.g. `HD_models_scale/CAG_1_1107.npz` / `HD_models_scale/CAG_2_1107.npz`):

| # | Analysis | Script | Run |
|---|---|---|---|
| 1 | Silhouette diagnostic plots for selecting k | [`silhouette_plots.py`](huntingtons/silhouette_plots.py) | `python -m analysis.huntingtons.silhouette_plots --cag1-npz .../CAG_1_1107.npz --cag2-npz .../CAG_2_1107.npz --output-dir .` |
| 2 | Primary clustering per CAG model, clinical bar chart + cell-type heatmaps (5f) | [`clustering_analysis.py`](huntingtons/clustering_analysis.py) | `python -m analysis.huntingtons.clustering_analysis --cag1-npz .../CAG_1_1107.npz --cag2-npz .../CAG_2_1107.npz` |
| 3 | Concordance between benign/pathogenic CAG clusters, cross-tab + clinical comparison (5g) | [`cluster_overlap_analysis.py`](huntingtons/cluster_overlap_analysis.py) | `python -m analysis.huntingtons.cluster_overlap_analysis --cag1-npz .../CAG_1_1107.npz --cag2-npz .../CAG_2_1107.npz` |
| 3 | Publication figure for the clusters found above | [`cluster_overlap_figure.py`](huntingtons/cluster_overlap_figure.py) | `python -m analysis.huntingtons.cluster_overlap_figure --overlap-csv cluster_overlap_donors.csv --output-dir .` |
| 4 | Supplementary: violin-plot cluster-vs-clinical comparison | [`cluster_clinical_violin.py`](huntingtons/cluster_clinical_violin.py) | `python -m analysis.huntingtons.cluster_clinical_violin --cag1-npz .../CAG_1_1107.npz --cag2-npz .../CAG_2_1107.npz` |
| 4 | Supplementary: console p-value report | [`cluster_pvalues.py`](huntingtons/cluster_pvalues.py) | `python -m analysis.huntingtons.cluster_pvalues --cag1-npz .../CAG_1_1107.npz --cag2-npz .../CAG_2_1107.npz` |
