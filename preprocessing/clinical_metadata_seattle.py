#!/usr/bin/env python3
"""
Seattle-AD clinical/donor-level metadata pre-processing (Methods 9.3): replaces the
ACT study's "Reference" placeholder (used for donors in the non-diseased reference
group) with true missing values, integer-encodes the remaining categorical columns,
then merges the cleaned donor-level clinical table back into each per-chunk AnnData
object produced by `qc_scarf_seattle.py` (Seattle-AD is processed in chunks because
of its size).

Usage:
    python -m preprocessing.clinical_metadata_seattle \
        --annotations-csv $CASCADE_DATA_ROOT/data/SEATTLE/SEATTLE_full_annotations.csv \
        --chunks-dir $CASCADE_DATA_ROOT/processed/SEA_AD \
        --output-dir $CASCADE_DATA_ROOT/processed/SEA_AD
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

from preprocessing.clinical_metadata_utils import export_value_counts_excel, integer_encode_columns

VALUE_COUNT_COLUMNS = [
    'ADNC', 'APOE4 status', 'Age at death', 'Braak stage', 'CERAD score', 'Class', 'Cognitive status',
    'Continuous Pseudo-progression Score', 'LATE-NC stage', 'Lewy body disease pathology',
    'Microinfarct pathology', 'PMI', 'Specimen ID', 'Subclass', 'Supertype', 'Thal phase',
    'Years of education', 'assay', 'batch', 'cell_type', 'development_stage', 'disease', 'donor_id',
    'self_reported_ethnicity', 'sex',
]
REFERENCE_PLACEHOLDER_COLUMNS = [
    'ADNC', 'APOE4 status', 'Braak stage', 'CERAD score', 'Cognitive status', 'LATE-NC stage',
    'Lewy body disease pathology', 'Microinfarct pathology', 'PMI', 'Thal phase', 'Years of education',
]
INTEGER_ENCODING_COLUMNS = [
    'ADNC', 'APOE4 status', 'Age at death', 'Braak stage', 'CERAD score', 'Class', 'Cognitive status',
    'LATE-NC stage', 'Lewy body disease pathology', 'Microinfarct pathology', 'PMI', 'Specimen ID',
    'Subclass', 'Supertype', 'Thal phase', 'Years of education', 'assay', 'batch', 'development_stage',
    'donor_id', 'self_reported_ethnicity', 'sex',
]


def clean_annotations(annotations_csv, output_dir, value_counts_xlsx=None):
    df = pd.read_csv(annotations_csv)

    print('1. Replace ACT "Reference" placeholder (and an unclassifiable LATE-NC label) with true missing values')
    for col in REFERENCE_PLACEHOLDER_COLUMNS:
        df[col] = df[col].replace({'Reference': np.nan})
    df['LATE-NC stage'] = df['LATE-NC stage'].replace(
        {'Staging Precluded by FTLD with TDP43 or ALS/MND or TDP-43 pathology is unclassifiable': np.nan})

    print('2. Integer-encode categorical columns')
    label_encoders = integer_encode_columns(df, INTEGER_ENCODING_COLUMNS)

    df = df.drop(columns=[c for c in ('batch', 'Unnamed: 0') if c in df.columns]).drop_duplicates()
    clean_csv_path = Path(output_dir) / 'SEATTLE_full_annotations_clean.csv'
    df.to_csv(clean_csv_path, index=False)

    if value_counts_xlsx:
        export_value_counts_excel(df, VALUE_COUNT_COLUMNS, label_encoders, value_counts_xlsx)
    return df


def merge_into_chunks(df, chunks_dir, output_dir):
    """Merge the cleaned donor-level clinical table into each per-chunk AnnData
    object (matched on the `ids` cell identifier column)."""
    chunk_files = sorted(Path(chunks_dir).glob("adata_annotated_protein_coding_fin_*.h5ad"))
    for chunk_file in tqdm(chunk_files):
        new_name = chunk_file.name.replace("coding", "clinical")
        out_path = Path(output_dir) / new_name
        if out_path.exists():
            continue
        adata = sc.read_h5ad(chunk_file)
        merged = pd.merge(adata.obs[['ids']], df, on='ids')
        adata.obs = merged
        adata.write(out_path)


def run(annotations_csv, chunks_dir, output_dir, value_counts_xlsx=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = clean_annotations(annotations_csv, output_dir, value_counts_xlsx=value_counts_xlsx)
    merge_into_chunks(df, chunks_dir, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations-csv", type=Path, required=True, help="SEATTLE_full_annotations.csv")
    parser.add_argument("--chunks-dir", type=Path, required=True,
                         help="Directory containing adata_annotated_protein_coding_fin_*.h5ad chunks")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--value-counts-xlsx", type=Path, default=None)
    args = parser.parse_args()
    run(args.annotations_csv, args.chunks_dir, args.output_dir, value_counts_xlsx=args.value_counts_xlsx)


if __name__ == "__main__":
    main()
