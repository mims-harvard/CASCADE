"""
Shared single-cell QC/normalisation helpers used by the per-dataset preprocessing
entrypoints (Methods 9.2): mitochondrial/total-count filtering, total-count
normalisation, log transform, highly-variable-gene selection, and scaling.

Every dataset except Seattle-AD was preprocessed with this Scanpy-based pipeline;
Seattle-AD used a separate out-of-core (scarf/dask) pipeline for the same steps
because of its size (`qc_scarf_seattle.py`).
"""
import numpy as np
import scanpy as sc

# Per-dataset QC thresholds actually used for the paper's runs (Supplementary Table
# S1), applied via `max_genes_by_counts`/`max_pct_counts_mt` below. These were
# determined by manual inspection of QC scatterplots per dataset, not derived
# programmatically.
QC_THRESHOLDS = {
    'HLCA': {'max_genes_by_counts': 8000, 'max_pct_counts_mt': 40},
    'LUCA': {'max_genes_by_counts': 5000, 'max_pct_counts_mt': 3},
    'M2': {'max_genes_by_counts': 800, 'max_pct_counts_mt': 3},
    'AUTISM': {'max_genes_by_counts': 10000, 'max_pct_counts_mt': 4},
}


def compute_qc_metrics(adata, mt_prefix='MT-', gene_symbol_col=None):
    """Flag mitochondrial genes and compute per-cell QC metrics (n_genes_by_counts,
    total_counts, pct_counts_mt) via `sc.pp.calculate_qc_metrics`."""
    symbols = adata.var[gene_symbol_col] if gene_symbol_col else adata.var_names.to_series()
    adata.var["mt"] = symbols.str.startswith(mt_prefix).values
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    return adata


def filter_low_quality_cells(adata, max_genes_by_counts, max_pct_counts_mt,
                              min_genes=200, min_cells=3, mt_prefix='MT-', gene_symbol_col=None):
    """Genes in <3 cells, cells with <200 genes, and dataset-specific upper bounds
    on n_genes_by_counts / pct_counts_mt (Methods 9.2)."""
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    adata = compute_qc_metrics(adata, mt_prefix=mt_prefix, gene_symbol_col=gene_symbol_col)
    adata = adata[adata.obs["total_counts"] > 0]
    adata = adata[adata.obs.n_genes_by_counts < max_genes_by_counts, :]
    adata = adata[adata.obs.pct_counts_mt < max_pct_counts_mt, :]
    return adata


def normalize_and_log(adata, target_sum=1e4):
    """Total-count normalise to `target_sum` and log1p-transform (Methods 9.2)."""
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def select_highly_variable(adata, hvg_fraction=0.9):
    """Keep the top `hvg_fraction` most variable genes (seurat_v3), i.e. drop the
    bottom 10% least variable genes, matching Methods 9.2."""
    n_top_genes = int(np.round(adata.shape[1] * hvg_fraction))
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=n_top_genes)
    return adata[:, adata.var.highly_variable].copy()


def scale(adata, max_value=10):
    """Scale to unit variance, clipping values beyond `max_value` s.d. (Methods 9.2).
    Stores the pre-scaling matrix in `.raw` since scaled data is not directly usable
    for downstream median/fold-change tokenisation (Methods 9.4)."""
    adata = adata.copy()
    adata.raw = adata
    sc.pp.scale(adata, max_value=max_value)
    return adata
