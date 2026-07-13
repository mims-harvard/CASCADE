<p align="center">
  <a href="https://valegiunchiglia.github.io/cascade-website/">
    <img src="assets/banner.png" alt="CASCADE — Cross-System, Multi-Scale Single-Cell Foundation Model with Clinical Applications" width="100%">
  </a>
</p>

<p align="center">
  <a href="https://valegiunchiglia.github.io/cascade-website/"><img src="https://img.shields.io/badge/Project%20Page-valegiunchiglia.github.io%2Fcascade--website-14b8a6" alt="Project Page"></a>
  <a href="https://github.com/mims-harvard/CASCADE"><img src="https://img.shields.io/badge/Code-mims--harvard%2FCASCADE-181717?logo=github&logoColor=white" alt="Code"></a>
  <img src="https://img.shields.io/badge/Paper-coming%20soon-lightgrey" alt="Paper (coming soon)">
  <a href="https://huggingface.co/mims-harvard"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-mims--harvard-FFD21E" alt="HuggingFace"></a>
</p>

CASCADE is a context-aware, multiscale single-cell foundation model. During pre-training it
conditions each cell's representation on the biological context in which it was observed —
disease state, tissue, cell type, and treatment condition — by tokenizing cells as
context-dependent up-/down-regulated gene sequences rather than raw expression. A
patient-representation module aggregates cell embeddings per donor via cross-attention, linking
single-cell molecular states to tissue-, disease-, and patient-level phenotypes. CASCADE-Explainer
provides two complementary explanations: an attention-based **cell explainer** (which cells drive a
patient-level prediction) and a perturbation-based **gene explainer** (which genes drive a
prediction, via in-silico knockdown and KL-divergence scoring).

<p align="center">
  <img src="assets/figure1_overview.png" alt="CASCADE model overview: context-aware tokenization, cell encoder, context-specific projectors, patient representation module, and the explainability module" width="900">
</p>

CASCADE was evaluated across four large-scale disease cohorts (HLCA, LuCA, SEATTLE-Alzheimer,
Autism; 51 supervised tasks against 8 baselines), and in three case studies: Alzheimer's disease
cell-type and gene-programme recovery, a Huntington's disease patient-stratification study, and
recovery of thyroid hormone / THRα receptor-signalling gene programmes.

<p align="center">
  <img src="assets/figure2_datasets_stratification.png" alt="Disease-specific context-aware single-cell modelling across six cohorts, and patient-specific molecular-to-clinical phenotype modelling with patient stratification" width="900">
</p>

## Repository structure

```
CASCADE/
├── cascade/                 # core model package
│   ├── model/                 # architecture: TransformerGenerator, GeneEncoder,
│   │                           #   PatientAggregator / AttentionClassifier / AttentionRegressor
│   ├── explainer/              # CASCADE-Explainer: cell + gene explainer, embedding extraction,
│   │                           #   donor/cell-level downstream prediction heads
│   ├── data/                   # context-aware tokenizer (Methods 9.4), contrastive sampler,
│   │                           #   collator, donor splits
│   └── training/                # DDP training loop, Sinkhorn domain adaptation
├── preprocessing/            # raw -> model-ready preprocessing: QC/normalisation, gene
│                              #   annotation, clustering, batch-effect correction, clinical/
│                              #   donor-level metadata cleaning, donor-level stratified splitting,
│                              #   the per-context median expression reference, and the scripts
│                              #   that build the tokenizer's dictionaries (Methods 9.2-9.4;
│                              #   see analysis/README.md for the run order per dataset)
├── analysis/
│   ├── benchmarking/           # 4-dataset, 51-task benchmark vs. baselines; sensitivity analyses
│   ├── alzheimers/              # CASCADE-Explainer cell-type / gene-programme recovery (SEATTLE)
│   ├── huntingtons/             # patient stratification, CAG repeat burden, neuropathology
│   ├── thyroid_hormone/         # hormone-response / receptor-signalling perturbation (R pipeline)
│   └── README.md                # maps every script to a paper figure and documents run order
└── scripts/                   # SLURM/job launchers (.sh) for the entrypoints above
```

`cascade/` is the reusable package; everything under `analysis/` is a standalone script (or,
for `thyroid_hormone/`, an R pipeline) that consumes frozen CASCADE embeddings or checkpoints
and is meant to be run, not imported.

## Installation

_TBD — a proper `environment.yml`/`requirements.txt` and R `renv` lockfile still need to be
generated. For reference, the Python side currently depends on:_ `torch`, `datasets`
(HuggingFace), `scanpy`, `anndata`, `scikit-learn`, `scipy`, `statsmodels`, `pandas`, `numpy`,
`matplotlib`, `seaborn`, `wandb` (optional, training/eval logging). _The R side
(`analysis/thyroid_hormone/`) depends on:_ `here`, `dplyr`, `tidyr`, `readr`, `tibble`,
`stringr`, `data.table`, `janitor`, `pROC`, `openxlsx`, `readxl`, `Seurat`, `AnnotationDbi`,
`org.Mm.eg.db`, `biomaRt`.

## Data

Data are not included in this repository. Paths are resolved via the `CASCADE_DATA_ROOT` and
`CASCADE_CKPT_ROOT` environment variables (see `scripts/*.sh` for examples) rather than being
hardcoded, so the repo is portable across environments. Pretrained checkpoints are on
HuggingFace (see below); the underlying raw datasets and tokenized Arrow shards still need to
be sourced separately per dataset.

## Pretrained models

Pretrained CASCADE checkpoints, one per pretraining cohort, are hosted on HuggingFace under
[mims-harvard](https://huggingface.co/mims-harvard). Each model page also documents the
dataset it was pretrained on.

| Cohort | Model |
|---|---|
| Seattle-AD (Alzheimer's) | [mims-harvard/CASCADE-Alzheimer](https://huggingface.co/mims-harvard/CASCADE-Alzheimer) |
| Autism | [mims-harvard/CASCADE-AUTISM](https://huggingface.co/mims-harvard/CASCADE-AUTISM) |
| Mouse-thyroid | [mims-harvard/CASCADE-THYROID](https://huggingface.co/mims-harvard/CASCADE-THYROID) |
| LuCA (lung cancer atlas) | [mims-harvard/CASCADE-LUCA](https://huggingface.co/mims-harvard/CASCADE-LUCA) |
| HLCA (Human Lung Cell Atlas) | [mims-harvard/CASCADE-HLCA](https://huggingface.co/mims-harvard/CASCADE-HLCA) |

## Reproducing the analyses

Each subfolder under `analysis/` corresponds to one or more paper figures/Results sections.
Start with [`analysis/README.md`](analysis/README.md) for the full script-to-figure map, then
look at `scripts/*.sh` for example SLURM launch commands for the entrypoints that need one
(training, embedding extraction, gene explanation).

## Figures

Some analysis scripts already produce publication-style figures directly (e.g.
`analysis/benchmarking/luca/umap_visualization.py`,
`analysis/huntingtons/cluster_overlap_figure.py`, and the `--plot` mode of the sensitivity
scripts in `analysis/benchmarking/sensitivity/`). Most other scripts currently emit tables
(CSV/RDS) rather than rendered plots — the final multi-panel paper figures (Figs. 1-5) are
composite and were assembled downstream of these tables; that assembly step is not yet in the
repo. If you want the repo to reproduce figure panels directly rather than just the underlying
statistics, the relevant plotting code for each panel would need to be uploaded alongside its
analysis script.

## Citation

_TBD_

## Contact

_TBD_
