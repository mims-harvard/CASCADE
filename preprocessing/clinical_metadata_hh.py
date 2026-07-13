#!/usr/bin/env python3
"""
Huntington's disease (HH) clinical/donor-level metadata pre-processing (Methods 9.3):
derives cell ontology term IDs, renames the disease status column, bins age into
decades and post-mortem interval into 5-hour bins, splits the biallelic CAG repeat
string into max/min/diff allele-specific measures, and integer-encodes the remaining
categorical columns.

Fixes vs. the original lab script: `Onset/Motor` and `Onset/Cog` are literal clinical
column names, but h5ad (HDF5-based) forbids '/' in keys, so `adata.write()` would
crash as soon as those columns reached .obs. Renamed to `Onset_Motor`/`Onset_Cog`.

Usage:
    python -m preprocessing.clinical_metadata_hh \
        --input-path $CASCADE_DATA_ROOT/processed/HH/adata_annotated.h5ad \
        --output-path $CASCADE_DATA_ROOT/processed/HH/adata_annotated_protein_coding_clinical.h5ad
"""
import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc

from preprocessing.clinical_metadata_utils import export_value_counts_excel, integer_encode_columns

VALUE_COUNT_COLUMNS = [
    'batch', 'cell_type', 'Donor', 'Age', 'Age_decade', 'Sex', 'disease', 'CAG', 'PMI', 'PMI-binned',
    'VS_Grade', 'Onset_Motor', 'Onset_Cog', 'CAP', 'CAP-100',
]
# h5ad (HDF5-based) forbids '/' in column names.
SLASH_COLUMN_RENAME = {'Onset/Motor': 'Onset_Motor', 'Onset/Cog': 'Onset_Cog'}
INTEGER_ENCODING_COLUMNS = ['batch', 'Donor', 'Age_decade', 'Sex', 'CAG', 'PMI-binned', 'VS_Grade', 'CAP', 'CAP-100']

# Cell type / disease / tissue -> Cell Ontology, MONDO/PATO, and UBERON term IDs
# (Methods 9.1: anterior caudate nucleus of the striatum).
CELL_TYPE_ONTOLOGY_MAP = {
    'interneuron': 'CL:0000099', 'SPN': 'CL:0000540', 'astrocyte': 'CL:0000127',
    'polydendrocyte': 'CL:0002453', 'oligodendrocyte': 'CL:0000128', 'endothelia': 'CL:0000115',
    'microglia': 'CL:0000129',
}
DISEASE_ONTOLOGY_MAP = {'Case': 'MONDO:0007739', 'Control': 'PATO:0000461'}


def run(input_path, output_path, value_counts_xlsx=None):
    adata = sc.read_h5ad(input_path)
    adata.obs.rename(columns={'Status': 'disease', **SLASH_COLUMN_RENAME}, inplace=True)

    print('1. Derive ontology term IDs for cell type, disease, and tissue')
    adata.obs['cell_type_ontology_term_id'] = adata.obs['cell_type'].replace(CELL_TYPE_ONTOLOGY_MAP)
    adata.obs['disease_ontology_term_id'] = adata.obs['disease'].replace(DISEASE_ONTOLOGY_MAP)
    adata.obs['tissue_ontology_term_id'] = adata.obs['tissue'].replace({'caudate': 'UBERON:0001873'})

    print('2. Bin age into decades')
    adata.obs['Age'] = adata.obs['Age'].replace('>89', '90')
    adata.obs['Age'] = pd.to_numeric(adata.obs['Age'], errors='coerce')
    adata.obs['Age_decade'] = pd.cut(
        adata.obs['Age'], bins=[20, 30, 40, 50, 60, 70, 80, 90, 101],
        labels=['20s', '30s', '40s', '50s', '60s', '70s', '80s', '90+'], right=False)

    print('3. Bin post-mortem interval into 5 hour bins')
    pmi_bins = list(range(0, 50, 5))
    pmi_labels = [f"{pmi_bins[i]}-{pmi_bins[i + 1]}" for i in range(len(pmi_bins) - 1)]
    adata.obs['PMI-binned'] = pd.cut(adata.obs['PMI'], bins=pmi_bins, labels=pmi_labels, right=False)

    print('4. Split biallelic CAG repeat length into max/min/diff allele measures')
    cag_split = adata.obs["CAG"].str.split("/", expand=True).astype(float)
    adata.obs["CAG_1"] = cag_split[0]
    adata.obs["CAG_2"] = cag_split[1]
    adata.obs["CAG_max"] = adata.obs[["CAG_1", "CAG_2"]].max(axis=1)
    adata.obs["CAG_min"] = adata.obs[["CAG_1", "CAG_2"]].min(axis=1)
    adata.obs["CAG_diff"] = adata.obs["CAG_max"] - adata.obs["CAG_min"]

    print('5. Integer-encode categorical columns')
    label_encoders = integer_encode_columns(adata.obs, INTEGER_ENCODING_COLUMNS)
    adata.write(output_path)

    if value_counts_xlsx:
        export_value_counts_excel(adata.obs, VALUE_COUNT_COLUMNS, label_encoders, value_counts_xlsx)
    return adata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-path", type=Path, required=True, help="adata_annotated.h5ad")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--value-counts-xlsx", type=Path, default=None)
    args = parser.parse_args()
    run(args.input_path, args.output_path, value_counts_xlsx=args.value_counts_xlsx)


if __name__ == "__main__":
    main()
