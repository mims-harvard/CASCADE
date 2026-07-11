#!/usr/bin/env python3
"""
Batch-effect correction via linear regression (Methods 9.2): "in case of the LUCA
dataset, we performed an additional correction step using linear regression. Batch
covariates, including platform and dataset, were used as regressors, and gene
expression values as the outcome. The residuals of this model were used as corrected
gene expression values in downstream analyses." This is only applied to LUCA; the
other datasets did not show strong enough batch effects on UMAP inspection to warrant
it.

For each of three regressor sets (platform only, dataset only, both), fits one OLS
model per gene against one-hot-encoded batch covariates and replaces the expression
matrix with the residuals, then plots UMAPs (by cell_type/platform/dataset) to verify
biological structure survived the correction.

Fixes vs. the original lab script: the output filename used a literal
'processed_data_batch_effect_{key}.h5ad' (missing the f-string prefix), so every
regressor set silently overwrote the same file instead of writing three separate ones.

Usage:
    python -m preprocessing.batch_effect_regression \
        --input-path $CASCADE_DATA_ROOT/processed/LUCA/adata_annotated_protein_coding.h5ad \
        --output-dir $CASCADE_DATA_ROOT/processed/LUCA
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.api as sm
from tqdm import tqdm

DATASET_NAME = 'LUCA'


def regress_out_batch(adata, batch_dummies):
    """Fit one OLS model per gene against `batch_dummies` and return the residual
    expression matrix."""
    design_matrix = sm.add_constant(batch_dummies).astype(np.float64)
    X = adata.X
    X_residuals = np.zeros_like(X)
    for i in tqdm(range(X.shape[1])):
        y = np.asarray(X[:, i], dtype=np.float64)
        model = sm.OLS(y, np.asarray(design_matrix))
        X_residuals[:, i] = model.fit().resid
    return X_residuals


def run(input_path, output_dir, figures_dir=None, batch_columns=('platform', 'dataset')):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(figures_dir) if figures_dir else output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    adata_orig = sc.read_h5ad(input_path)
    metadata = adata_orig.obs[list(batch_columns)]

    regressor_sets = {
        'both': pd.get_dummies(metadata, drop_first=True),
        **{col: pd.get_dummies(adata_orig.obs[[col]], drop_first=True) for col in batch_columns},
    }

    for key, batch_dummies in regressor_sets.items():
        adata = adata_orig.copy()
        adata.X = regress_out_batch(adata, batch_dummies)

        sc.tl.pca(adata, svd_solver='arpack')
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        for color in ('cell_type', 'platform', 'dataset'):
            sc.pl.umap(adata, color=[color])
            plt.tight_layout()
            plt.savefig(figures_dir / f"umap_plot_regress_{key}_{color}_{DATASET_NAME}.png")

        adata.write_h5ad(output_dir / f'processed_data_batch_effect_{key}.h5ad')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-path", type=Path, required=True,
                         help="adata_annotated_protein_coding.h5ad from protein_coding_annotation.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, default=None)
    args = parser.parse_args()
    run(args.input_path, args.output_dir, figures_dir=args.figures_dir)


if __name__ == "__main__":
    main()
