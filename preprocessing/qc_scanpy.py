#!/usr/bin/env python3
"""
Generic Scanpy-based single-cell QC/normalisation pipeline (Methods 9.2), used for
every dataset except Seattle-AD (which used the out-of-core scarf pipeline in
`qc_scarf_seattle.py` because of its size). Covers HLCA, LUCA, and the mouse-thyroid
(M2) dataset; AUTISM has its own entrypoint (`qc_scanpy_autism.py`) because its raw
data ships as loose 10x-style mtx/barcodes/genes files rather than a cellxgene h5ad.

Steps: (1) restore raw counts, (2) [M2 only] map Ensembl IDs to gene symbols via a
pre-cleaned BioMart export, (3) filter genes in <3 cells and cells with <200 genes,
(4) filter cells on dataset-specific n_genes_by_counts / pct_counts_mt thresholds
(Supplementary Table S1), (5) total-count normalise + log1p, (6) select the top 90%
highly-variable genes, (7) scale to unit variance (clipped at 10 s.d.).

Usage:
    python -m preprocessing.qc_scanpy --dataset-name LUCA \
        --input-path $CASCADE_DATA_ROOT/raw/LUCA/adata_raw.h5ad \
        --output-dir $CASCADE_DATA_ROOT/processed/LUCA
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

from preprocessing.qc_pipeline import (
    QC_THRESHOLDS, filter_low_quality_cells, normalize_and_log, scale, select_highly_variable,
)

CASCADE_DATA_ROOT = Path(os.environ.get("CASCADE_DATA_ROOT", "/n/data1/hms/dbmi/zitnik/lab/datasets/2024-11-CASCADE"))


def map_m2_ensembl_ids(adata, biomart_file):
    """M2 (mouse-thyroid) ships with Ensembl IDs as the var index; map to gene
    symbols ('feature_name') via a pre-exported BioMart table before QC."""
    adata.var = adata.var.reset_index().rename(columns={'index': 'ensembl_gene_id'})
    bio = pd.read_csv(biomart_file, sep='\t').rename(
        columns={'Gene stable ID': 'ensembl_gene_id', 'Gene name': 'feature_name'})
    adata.var = pd.merge(adata.var, bio, on='ensembl_gene_id', how='left')
    return adata[:, ~adata.var.feature_name.isna()]


def run(dataset_name, input_path, output_dir, figures_dir=None, biomart_file=None,
        max_genes_by_counts=None, max_pct_counts_mt=None, min_genes=200, min_cells=3):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(figures_dir) if figures_dir else output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    defaults = QC_THRESHOLDS.get(dataset_name, {})
    max_genes_by_counts = max_genes_by_counts or defaults.get('max_genes_by_counts')
    max_pct_counts_mt = max_pct_counts_mt or defaults.get('max_pct_counts_mt')
    if max_genes_by_counts is None or max_pct_counts_mt is None:
        raise ValueError(
            f"No default QC thresholds for '{dataset_name}'; pass --max-genes-by-counts/--max-pct-counts-mt.")

    print("1. Retrieve dataset of interest and start pre-processing")
    adata = sc.read_h5ad(input_path)
    adata.X = adata.raw.X
    print(adata.shape)

    if dataset_name == 'M2':
        if biomart_file is None:
            raise ValueError("M2 requires --biomart-file (Ensembl ID -> gene symbol export).")
        adata = map_m2_ensembl_ids(adata, biomart_file)

    print("2. Identify highest expressed genes")
    sc.pl.highest_expr_genes(adata, n_top=20)
    plt.savefig(figures_dir / f"high_expressed_{dataset_name}.png")

    print("3. Filter genes in <3 cells, cells with <200 genes, and QC outlier cells")
    mt_prefix = 'mt-' if dataset_name == 'M2' else 'MT-'
    gene_symbol_col = 'feature_name' if dataset_name == 'M2' else None
    adata = filter_low_quality_cells(
        adata, max_genes_by_counts, max_pct_counts_mt,
        min_genes=min_genes, min_cells=min_cells, mt_prefix=mt_prefix, gene_symbol_col=gene_symbol_col,
    )
    print(adata.shape)

    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.1, multi_panel=True)
    plt.savefig(figures_dir / f"violinplot_{dataset_name}.png")
    sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
    plt.savefig(figures_dir / f"total_pct_plot_{dataset_name}.png")
    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
    plt.savefig(figures_dir / f"total_ngenes_plot_{dataset_name}.png")

    print("4. Total count normalize and log-transform")
    adata = normalize_and_log(adata)
    adata.write(output_dir / 'adata_normalized_log.h5ad')

    print("5. Detect highly variable genes and remove the bottom 10% least variable")
    adata = select_highly_variable(adata)
    adata.write(output_dir / 'adata_varable_genes.h5ad')
    print(adata.shape)

    print("6. Scaling (NB: scaled data is not used for downstream median/fold-change tokenisation)")
    adata_scaled = scale(adata)
    adata_scaled.write(output_dir / 'adata_scaled.h5ad')

    if 'disease' in adata_scaled.obs:
        print("Final number of healthy and disease cells")
        print(adata_scaled.obs.disease.value_counts())
    print("Final number of donors")
    if 'donor_id' in adata_scaled.obs:
        print(len(adata_scaled.obs.donor_id.unique()))
    return adata_scaled


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-name", required=True, choices=sorted(set(QC_THRESHOLDS) | {'OTHER'}))
    parser.add_argument("--input-path", type=Path, required=True, help="Raw cellxgene-format h5ad file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--biomart-file", type=Path, default=None, help="Required for --dataset-name M2")
    parser.add_argument("--max-genes-by-counts", type=float, default=None)
    parser.add_argument("--max-pct-counts-mt", type=float, default=None)
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--min-cells", type=int, default=3)
    args = parser.parse_args()

    run(args.dataset_name, args.input_path, args.output_dir, figures_dir=args.figures_dir,
        biomart_file=args.biomart_file, max_genes_by_counts=args.max_genes_by_counts,
        max_pct_counts_mt=args.max_pct_counts_mt, min_genes=args.min_genes, min_cells=args.min_cells)


if __name__ == "__main__":
    main()
