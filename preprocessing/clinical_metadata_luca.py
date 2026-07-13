#!/usr/bin/env python3
"""
LUCA clinical/donor-level metadata pre-processing (Methods 9.3): merges the
batch-corrected expression matrix (from `batch_effect_regression.py`) back with the
full clinical obs columns dropped during batch correction, combines stage III/IV
into one category, bins age, converts "unknown"/"nan" placeholders to true missing
values, and integer-encodes the remaining categorical columns.

Usage:
    python -m preprocessing.clinical_metadata_luca \
        --batch-corrected-path $CASCADE_DATA_ROOT/processed/LUCA/processed_data_batch_effect_both.h5ad \
        --annotated-path $CASCADE_DATA_ROOT/processed/LUCA/adata_annotated_protein_coding.h5ad \
        --output-path $CASCADE_DATA_ROOT/processed/LUCA/adata_annotated_protein_coding_clinical.h5ad
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from preprocessing.clinical_metadata_utils import export_value_counts_excel, integer_encode_columns

VALUE_COUNT_COLUMNS = [
    'uicc_stage', 'ever_smoker', 'age', 'origin', 'dataset', 'tumor_stage', 'EGFR_mutation',
    'TP53_mutation', 'ALK_mutation', 'BRAF_mutation', 'ERBB2_mutation', 'KRAS_mutation', 'ROS_mutation',
    'origin_fine', 'study', 'platform', 'assay', 'sex', 'uicc_stage_combined', 'age-binned',
]


def run(batch_corrected_path, annotated_path, output_path, value_counts_xlsx=None):
    adata = sc.read_h5ad(batch_corrected_path)
    adata_original = sc.read_h5ad(annotated_path)

    # Batch-regression residuals are centered at zero; add the original log-scale
    # mean back so downstream tokenisation's expm1/log1p assumptions still hold.
    original_log_mean = np.mean(adata_original.X)
    adata.X = adata.X + original_log_mean

    cols_merge = list(set(adata_original.obs) - set(adata.obs))
    obs_merged = pd.merge(
        adata.obs, adata_original.obs[cols_merge + ['observation_joinid']], on='observation_joinid', how='left')
    adata.obs = obs_merged

    print('1. Combine stage III and IV into one category')
    adata.obs['uicc_stage_combined'] = adata.obs.uicc_stage.replace({'III': 'III or IV', 'IV': 'III or IV'})

    print('2. Bin age into 5 year bins')
    bins = range(20, 91, 5)
    adata.obs['age-binned'] = pd.cut(adata.obs['age'], bins=bins, right=False)
    adata.obs['age-binned'] = adata.obs['age-binned'].apply(lambda x: f'{x.left}-{x.right - 1}')

    print('3-5. Convert placeholder strings to true missing values')
    adata.obs.origin = adata.obs.origin.replace({'nan': np.nan})
    adata.obs.origin_fine = adata.obs.origin_fine.replace({'nan': np.nan})
    adata.obs.sex = adata.obs.sex.replace({'unknown': np.nan})

    print('6. Integer-encode categorical columns')
    obj_cols = adata.obs.select_dtypes(include=['object']).columns
    categorical_cols = adata.obs.select_dtypes(include=['category']).columns
    integer_encoding_cols = [c for c in (categorical_cols.tolist() + obj_cols.tolist()) if c in VALUE_COUNT_COLUMNS]
    label_encoders = integer_encode_columns(adata.obs, integer_encoding_cols)

    adata.write(output_path)

    if value_counts_xlsx:
        export_value_counts_excel(adata.obs, VALUE_COUNT_COLUMNS, label_encoders, value_counts_xlsx)
    return adata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-corrected-path", type=Path, required=True,
                         help="Output of batch_effect_regression.py")
    parser.add_argument("--annotated-path", type=Path, required=True,
                         help="adata_annotated_protein_coding.h5ad, for the obs columns dropped during batch correction")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--value-counts-xlsx", type=Path, default=None)
    args = parser.parse_args()
    run(args.batch_corrected_path, args.annotated_path, args.output_path, value_counts_xlsx=args.value_counts_xlsx)


if __name__ == "__main__":
    main()
