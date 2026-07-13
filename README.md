<p align="center">
  <a href="https://valegiunchiglia.github.io/cascade-website/">
    <img src="assets/banner.png" alt="CASCADE — Cross-System, Multi-Scale Single-Cell Foundation Model with Clinical Applications" width="100%">
  </a>
</p>

<h2 align="center">CASCADE: context-aware single-cell modelling links cellular programmes to patient-level disease phenotypes</h2>

<p align="center">
  <a href="https://valegiunchiglia.github.io/cascade-website/"><img src="https://img.shields.io/badge/Project%20Page-valegiunchiglia.github.io%2Fcascade--website-14b8a6" alt="Project Page"></a>
  <a href="https://github.com/mims-harvard/CASCADE"><img src="https://img.shields.io/badge/Code-mims--harvard%2FCASCADE-181717?logo=github&logoColor=white" alt="Code"></a>
  <img src="https://img.shields.io/badge/Paper-bioRxiv%20coming%20soon-b31b1b?logo=arxiv&logoColor=white" alt="Paper (bioRxiv, coming soon)">
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

## Pretraining

CASCADE is pretrained per-cohort with the context-specific contrastive objective (Methods
9.10-9.11) via [`cascade/training/train_ddp.py`](cascade/training/train_ddp.py), which wraps
the shared training loop in [`cascade/training/trainer.py`](cascade/training/trainer.py). It's
a DDP entrypoint, so it's always launched with `torchrun` — with `--nproc_per_node=1` on a
single GPU, or scaled up across nodes as in the SLURM example below. `--list_data` points at
one or more tokenized dataset directories (output of the
[preprocessing pipeline](analysis/README.md#preprocessing-methods-92-94)); `--dataset` must be
a key in `cascade.data.splits.SPLITS_BY_DATASET`.

```bash
CASCADE_DATA_ROOT=/path/to/DATASET CASCADE_CKPT_ROOT=/path/to/checkpoints \
torchrun --nproc_per_node=1 -m cascade.training.train_ddp \
    --list_data "$CASCADE_DATA_ROOT/HLCA/DISEASE/DISEASE-ID.dataset" \
    --dataset HLCA --batch 32 --nlayers 12 --cell_emb_style avg-pool \
    --context_specific_projections --donors --DA --nepochs 50
```

For multi-node training, see [`scripts/hlca_multinode.sh`](scripts/hlca_multinode.sh) for a
full SLURM launcher (checkpoint frequency, W&B logging, NCCL/rendezvous setup included).

## Inference

Downstream analyses consume *frozen* CASCADE embeddings rather than running the model live.
Extract them from a trained (or [downloaded](#pretrained-models)) checkpoint with
[`cascade/explainer/get_embeddings_parallel.py`](cascade/explainer/get_embeddings_parallel.py),
also a `torchrun` entrypoint:

```bash
torchrun --nproc_per_node=1 -m cascade.explainer.get_embeddings_parallel \
    --dataset HLCA --checkpoint /path/to/ckpt.pt --output-path /path/to/embeddings.pkl
```

This writes chunked embedding pickles (`*_rank_*_chunk_*.pkl`) that the donor-/cell-level
prediction heads and CASCADE-Explainer scripts under `analysis/` read directly. See
[`scripts/get_emb_parallel.sh`](scripts/get_emb_parallel.sh) for the multi-node SLURM version.

For the exact analyses and figures reported in the paper, see
[`analysis/README.md`](analysis/README.md).

## Citation

```bibtex
@article{giunchiglia2026cascade,
  title   = {CASCADE: context-aware single-cell modelling links cellular programmes to patient-level disease phenotypes},
  author  = {Giunchiglia, Valentina and Queen, Owen and Lin, Xiang and Matuszek, Zaneta
             and Hochbaum, Daniel and Venkat, Aarthi and Abbadessa, Gianmarco
             and Rickord, Walker and Nicholas, Richard and Arlotta, Paola and Zitnik, Marinka},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {TBD},
  url     = {TBD}
}
```

## Contact

For any questions or feedback, please open an issue in the
[GitHub repository](https://github.com/mims-harvard/CASCADE) or contact
[Valentina Giunchiglia](mailto:v.giunchiglia20@imperial.ac.uk) and
[Marinka Zitnik](mailto:marinka@hms.harvard.edu).
