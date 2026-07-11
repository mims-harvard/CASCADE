#!/usr/bin/env python3
"""
AUTISM-specific entry point into the Scanpy QC/normalisation pipeline (Methods
9.2). AUTISM ships as loose 10x-style Matrix Market files (matrix.mtx, barcodes.tsv,
genes.tsv, meta.txt) rather than a cellxgene h5ad, so it gets its own loader; the
QC/normalise/HVG/scale steps that follow are the same as `qc_scanpy.py` (shared via
`qc_pipeline.py`).

Fixes vs. the original lab script: `matplotlib.pyplot` was used without being
imported (would raise NameError on the first `plt.savefig` call).

Usage:
    python -m preprocessing.qc_scanpy_autism \
        --input-dir $CASCADE_DATA_ROOT/raw/AUTISM \
        --output-dir $CASCADE_DATA_ROOT/processed/AUTISM
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from scipy import io
from scipy.sparse import csr_matrix, issparse, isspmatrix_coo

from preprocessing.qc_pipeline import filter_low_quality_cells, normalize_and_log, scale, select_highly_variable

DATASET_NAME = 'AUTISM'
MAX_GENES_BY_COUNTS = 10_000
MAX_PCT_COUNTS_MT = 4


def load_raw(input_dir):
    """Assemble an AnnData from loose matrix.mtx/barcodes.tsv/genes.tsv/meta.txt."""
    input_dir = Path(input_dir)
    matrix = io.mmread(input_dir / 'matrix.mtx').T.tocsr()
    barcodes = pd.read_csv(input_dir / 'barcodes.tsv', header=None, sep='\t')
    meta = pd.read_csv(input_dir / 'meta.txt', sep='\t')
    genes = pd.read_csv(input_dir / 'genes.tsv', header=None, sep='\t')

    adata = sc.AnnData(X=matrix)
    adata.obs['barcode'] = barcodes[0].values
    adata.var['ensembl_gene_id'] = genes[0].values
    adata.var['approved_symbol'] = genes[1].values

    adata.obs = adata.obs.rename(columns={'barcode': 'cell'})
    adata.obs = adata.obs.join(meta.set_index('cell'), on='cell')
    return adata


def run(input_dir, output_dir, figures_dir=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(figures_dir) if figures_dir else output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    adata = load_raw(input_dir)
    adata.write(output_dir / 'adata_original.h5ad')

    sc.pl.highest_expr_genes(adata, n_top=20)
    plt.savefig(figures_dir / f"high_expressed_{DATASET_NAME}.png")

    print("Filter genes in <3 cells, cells with <200 genes, and QC outlier cells")
    adata = filter_low_quality_cells(
        adata, MAX_GENES_BY_COUNTS, MAX_PCT_COUNTS_MT, gene_symbol_col='approved_symbol')
    print(adata.shape)

    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.1, multi_panel=True)
    plt.savefig(figures_dir / f"violinplot_{DATASET_NAME}.png")
    sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
    plt.savefig(figures_dir / f"total_pct_plot_{DATASET_NAME}.png")
    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
    plt.savefig(figures_dir / f"total_ngenes_plot_{DATASET_NAME}.png")

    print("Total count normalize and log-transform")
    adata = normalize_and_log(adata)
    adata.write(output_dir / 'adata_normalized_log.h5ad')

    print("Detect highly variable genes")
    adata = select_highly_variable(adata)
    adata.write(output_dir / 'adata_varable_genes.h5ad')
    print(adata.shape)

    print("Scaling")
    adata_scaled = scale(adata)
    adata_scaled.write(output_dir / 'adata_scaled.h5ad')

    adata_scaled.obs = adata_scaled.obs.rename(columns={'diagnosis': 'disease', 'individual': 'donor_id'})
    print("Final number of healthy and disease cells")
    print(adata_scaled.obs.disease.value_counts())
    print("Final number of donors")
    print(len(adata_scaled.obs.donor_id.unique()))
    return adata_scaled


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True,
                         help="Directory containing matrix.mtx, barcodes.tsv, genes.tsv, meta.txt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, default=None)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, figures_dir=args.figures_dir)


if __name__ == "__main__":
    main()
