#!/usr/bin/env python3
"""
Mouse-thyroid (M2) clinical/donor-level metadata pre-processing (Methods 9.3):
integer-encode categorical columns while preserving missing values. `Cre` and
`THR_expr` are left as continuous (used for regression tasks), not encoded.

Usage:
    python -m preprocessing.clinical_metadata_m2 \
        --input-path $CASCADE_DATA_ROOT/processed/M2/adata_annotated_protein_coding.h5ad \
        --output-path $CASCADE_DATA_ROOT/processed/M2/adata_annotated_protein_coding_clinical.h5ad
"""
import argparse
from pathlib import Path

import scanpy as sc

from preprocessing.clinical_metadata_utils import export_value_counts_excel, integer_encode_columns

VALUE_COUNT_COLUMNS = ['donor_id', 'treatment', 'Cre', 'THR', 'THR_expr', 'batch']


def run(input_path, output_path, value_counts_xlsx=None):
    adata = sc.read_h5ad(input_path)

    categorical_cols = [
        col for col in adata.obs.select_dtypes(include=['category']).columns
        if col in VALUE_COUNT_COLUMNS and col != 'donor_id'
    ]
    print('Integer-encode categorical columns')
    label_encoders = integer_encode_columns(adata.obs, categorical_cols)
    adata.write(output_path)

    if value_counts_xlsx:
        export_value_counts_excel(adata.obs, VALUE_COUNT_COLUMNS, label_encoders, value_counts_xlsx)
    return adata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-path", type=Path, required=True, help="adata_annotated_protein_coding.h5ad")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--value-counts-xlsx", type=Path, default=None)
    args = parser.parse_args()
    run(args.input_path, args.output_path, value_counts_xlsx=args.value_counts_xlsx)


if __name__ == "__main__":
    main()
