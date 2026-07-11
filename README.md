# CASCADE

**Context-aware single-cell modelling links cellular programmes to patient-level disease phenotypes**

CASCADE is a context-aware, multiscale single-cell foundation model. During pre-training it
conditions each cell's representation on the biological context in which it was observed —
disease state, tissue, cell type, and treatment condition — by tokenizing cells as
context-dependent up-/down-regulated gene sequences rather than raw expression. A
patient-representation module aggregates cell embeddings per donor via cross-attention, linking
single-cell molecular states to tissue-, disease-, and patient-level phenotypes. CASCADE-Explainer
provides two complementary explanations: an attention-based **cell explainer** (which cells drive a
patient-level prediction) and a perturbation-based **gene explainer** (which genes drive a
prediction, via in-silico knockdown and KL-divergence scoring).

CASCADE was evaluated across four large-scale disease cohorts (HLCA, LuCA, SEATTLE-Alzheimer,
Autism; 51 supervised tasks against 8 baselines), and in three case studies: Alzheimer's disease
cell-type and gene-programme recovery, a Huntington's disease patient-stratification study, and
recovery of thyroid hormone / THRα receptor-signalling gene programmes.

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
│                              #   annotation, clustering, batch-effect correction, and the
│                              #   per-context median expression reference used by the tokenizer
│                              #   (Methods 9.2-9.3; see analysis/README.md for per-dataset status)
├── analysis/
│   ├── benchmarking/           # 4-dataset, 51-task benchmark vs. baselines; sensitivity analyses
│   ├── alzheimers/              # CASCADE-Explainer cell-type / gene-programme recovery (SEATTLE)
│   ├── huntingtons/             # patient stratification, CAG repeat burden, neuropathology
│   ├── thyroid_hormone/         # hormone-response / receptor-signalling perturbation (R pipeline)
│   └── README.md                # maps every script here to a paper section/figure, and lists
│                                 #   what's present vs. still needs to be uploaded
└── scripts/                   # SLURM/job launchers (.sh) for the entrypoints above
```

`cascade/` is the reusable package; everything under `analysis/` is a standalone script (or,
for `thyroid_hormone/`, an R pipeline) that consumes frozen CASCADE embeddings or checkpoints
and is meant to be run, not imported.

## Status

This repository is being assembled incrementally, paper section by paper section. **See
[`analysis/README.md`](analysis/README.md) for the authoritative map of which scripts exist,
which paper analyses they correspond to, and which are still missing.** In short:

- **Core model, training, and explainer package (`cascade/`)** — complete.
- **Benchmarking (Fig. 2)** — CASCADE-side prediction sweep complete; only 1 of 8 baselines
  (Geneformer, and only for LUCA) has been ported so far.
- **Alzheimer's disease case study (Fig. 3)** — LLM-arena literature-evidence comparison (3b-c) is
  in place; eQTL/GWAS colocalization (3d-e) and the external reference-set panel (3f) are missing.
- **Alzheimer's disease case study (canonical cell-type recovery)** — complete: ground truth,
  attention baseline, and the CASCADE-vs-baseline AUROC comparison are all in place.
- **Huntington's disease case study (Fig. 5)** — complete: prediction (5a-b), cell-type-importance
  vs. severity (5c-e), and clustering/stratification (5f-g) are all in place.
- **Thyroid hormone case study (Fig. 4)** — complete, most mature analysis folder in the repo.
- **Preprocessing (Methods 9.2-9.4)** — QC/normalisation, gene annotation, clustering,
  batch-effect correction, per-context median reference, and the context-aware tokenizer are
  all in place, covering every dataset except HH (whose analysis scripts consume pre-extracted
  attention caches rather than raw counts, so this doesn't block anything downstream).

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
hardcoded, so the repo is portable across environments — but the underlying datasets, tokenized
Arrow shards, and trained checkpoints still need to be sourced/downloaded separately per dataset.

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
