#!/usr/bin/env python3
"""
AUTISM clinical/donor-level metadata pre-processing (Methods 9.3): age and
post-mortem-interval binning, decomposing free-text comorbidity/medication fields
into binary indicator columns, correcting duplicate cause-of-death labels, and
integer-encoding the remaining categorical columns.

Usage:
    python -m preprocessing.clinical_metadata_autism \
        --input-path $CASCADE_DATA_ROOT/processed/AUTISM/adata_annotated_protein_coding.h5ad \
        --output-path $CASCADE_DATA_ROOT/processed/AUTISM/adata_annotated_protein_coding_clinical.h5ad
"""
import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc

from preprocessing.clinical_metadata_utils import export_value_counts_excel, integer_encode_columns
from preprocessing.seattle_ad_metadata import generate_binary_columns, mapping_data, medications_mapping_dict

VALUE_COUNT_COLUMNS = [
    'sample', 'donor_id', 'age', 'sex', 'Capbatch', 'Seqbatch', 'post-mortem interval (hours)',
    'Patient ID', 'Diagnosis', 'Age', 'Sex', 'PMI', 'Other diagnoses', 'Medications',
    'ADI-R-A', 'ADI-R-Bverbal', 'ADI-R-Bnonverbal', 'ADI-R-C', 'ADI-R-D', 'Cause of death',
    'Epilepsy', 'Attention-deficit/hyperactivity disorder', 'Cardiac malformation', 'Depression',
    'Pneumonia', 'Lead poisoning', 'Cerebellar Heterotopia', 'Developmental Delay',
    'binned', 'post-mortem-binned',
]
COMORBIDITY_CONDITIONS = [
    'Epilepsy', 'Attention-deficit/hyperactivity disorder', 'Cardiac malformation', 'Depression',
    'Pneumonia', 'Lead poisoning', 'Cerebellar Heterotopia', 'Developmental Delay',
]
INTEGER_ENCODING_COLUMNS = [
    'Cause of death', 'sample', 'sex', 'Capbatch', 'Seqbatch', 'Diagnosis', 'Sex',
    'Other diagnoses', 'Medications', 'binned', 'post-mortem-binned',
]


def run(input_path, output_path, value_counts_xlsx=None):
    adata = sc.read_h5ad(input_path)

    print('1. Binning of age into 5 years bins')
    adata.obs['binned'] = pd.cut(adata.obs['age'], bins=[0, 5, 10, 15, 20, 25],
                                  labels=['0-5', '5-10', '10-15', '15-20', '20-25'], right=False)

    print('2. Binning of post-mortem interval into 5 hour bins')
    pmi_bins = list(range(0, 50, 5))
    pmi_labels = [f"{pmi_bins[i]}-{pmi_bins[i + 1]}" for i in range(len(pmi_bins) - 1)]
    adata.obs['post-mortem-binned'] = pd.cut(
        adata.obs['post-mortem interval (hours)'], bins=pmi_bins, labels=pmi_labels, right=False)

    print('3. Convert comorbidity conditions into binary columns')
    binary_columns = adata.obs.apply(
        generate_binary_columns, axis=1, conditions=COMORBIDITY_CONDITIONS, column='Other diagnoses')
    # h5ad (HDF5-based) forbids '/' in column names, which "Attention-deficit/hyperactivity
    # disorder" would otherwise hit at adata.write() time; sanitize just the column labels.
    binary_df = pd.DataFrame(binary_columns.tolist(), columns=[c.replace('/', '-') for c in COMORBIDITY_CONDITIONS])
    binary_df.index = binary_df.index.astype(str)
    df_final = pd.merge(binary_df, adata.obs, right_index=True, left_index=True)

    print('4. Correct duplicate cause-of-death labels')
    label_mapping = {"accidental drowning": "drowning"}
    df_final['Cause of death'] = df_final['Cause of death'].str.lower().str.strip().replace(label_mapping)
    death_conditions = list(set(df_final['Cause of death'].dropna()))
    binary_columns = df_final.apply(
        generate_binary_columns, axis=1, conditions=death_conditions, column='Cause of death')
    binary_df = pd.DataFrame(binary_columns.tolist(), columns=death_conditions)
    binary_df.index = binary_df.index.astype(str)
    df_final = pd.merge(binary_df, df_final, right_index=True, left_index=True)

    print('5. Decompose medications into binary columns + ATC hierarchy levels')
    all_medications = sorted(set(sum(medications_mapping_dict.values(), [])))
    for med in all_medications:
        df_final[med] = df_final["Medications"].apply(lambda meds: 1 if med in meds else 0)

    mapping_df = pd.DataFrame(mapping_data)
    act_code_columns = ["Level 5", "Level 4", "Level 3", "Level 2", "Level 1"]
    for _, row in mapping_df.iterrows():
        med_name = row["Medications_mapping"]
        for level in act_code_columns:
            for code in row[level]:
                df_final.loc[df_final[med_name] == 1, code] = 1
                df_final.loc[df_final[med_name] == 0, code] = 0

    print('6. Integer-encode categorical columns')
    label_encoders = integer_encode_columns(df_final, INTEGER_ENCODING_COLUMNS)
    adata.obs = df_final
    adata.write(output_path)

    if value_counts_xlsx:
        export_value_counts_excel(df_final, VALUE_COUNT_COLUMNS, label_encoders, value_counts_xlsx)
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
