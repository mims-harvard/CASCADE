#!/usr/bin/env python3
"""
HLCA clinical/donor-level metadata pre-processing (Methods 9.3): BMI binning,
cause-of-death label standardisation and decomposition into binary columns,
ambiguous "unknown" placeholders converted to true missing values, developmental
stage grouped into age bins, and integer-encoding the remaining categorical columns.

Usage:
    python -m preprocessing.clinical_metadata_hlca \
        --input-path $CASCADE_DATA_ROOT/processed/HLCA/adata_annotated_protein_coding.h5ad \
        --output-path $CASCADE_DATA_ROOT/processed/HLCA/adata_annotated_protein_coding_clinical.h5ad
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from preprocessing.clinical_metadata_utils import export_value_counts_excel, integer_encode_columns

VALUE_COUNT_COLUMNS = [
    'BMI', 'age_range', 'anatomical_region_ccf_score', 'cause_of_death', 'dataset', 'lung_condition',
    'mixed_ancestry', 'sequencing_platform', 'smoking_status', 'study', 'subject_type', 'assay', 'sex',
    'sample', 'self_reported_ethnicity', 'development_stage', 'BMI_Groups', 'Head Trauma',
    'intracranial hemorrhage', 'stroke', 'Anoxic Brain Injury', 'Drug Overdose', 'Accident',
    'Suspected Suicide', 'age_group', 'suspension_type',
]
CAUSE_OF_DEATH_MAPPING = {
    "Head Trauma": ["head trauma", "head trauma from gunshot wound"],
    "intracranial hemorrhage": ["intracranial hemorrhage", "Intracranial haemorrhage", "Intracerebral hemorrhage"],
    "stroke": ["stroke", "CVA/stroke"],
    "Anoxic Brain Injury": ["anoxic brain injury", "anoxic brain injury after seizure"],
    "Drug Overdose": ["meth overdose", "drug overdose"],
    "Accident": ["MVA", "accident"],
    "Suspected Suicide": ["known or suspected suicide"],
}
INTEGER_ENCODING_COLUMNS = [
    'age_range', 'dataset', 'lung_condition', 'mixed_ancestry', 'sequencing_platform', 'smoking_status',
    'study', 'subject_type', 'assay', 'sex', 'self_reported_ethnicity', 'BMI_Groups', 'age_group',
    'suspension_type',
]
AGE_GROUP_BINS = [
    "0-28 days", "1-3 months", "4-6 months", "7-12 months", "1-4 years", "5-9 years", "10-19 years",
    "20-24 years", "25-29 years", "30-34 years", "35-39 years", "40-44 years", "45-49 years",
    "50-54 years", "55-59 years", "60-64 years", "65-69 years", "70-74 years", "75+ years",
]


def _standardize_cause_of_death(diagnosis, mapping=CAUSE_OF_DEATH_MAPPING):
    for standard_name, variations in mapping.items():
        if diagnosis.lower() in [v.lower() for v in variations]:
            return standard_name
    return diagnosis


def _generate_binary_columns(row, conditions):
    if pd.isna(row['cause_of_death']):
        return [np.nan] * len(conditions)
    condition_name = row['cause_of_death']
    return [1 if condition.lower() in condition_name.lower() else 0 for condition in conditions]


def _map_development_stage_to_age_group(stage):
    stage = stage.lower()
    if "newborn" in stage:
        return "0-28 days"
    if "month" in stage:
        if any(f"{m}-month" in stage for m in (1, 2, 3)):
            return "1-3 months"
        if any(f"{m}-month" in stage for m in (4, 5, 6)):
            return "4-6 months"
        return "7-12 months"
    if "year" in stage:
        age = int(stage.split("-")[0])
        for lo, hi, label in [(1, 4, "1-4 years"), (5, 9, "5-9 years"), (10, 19, "10-19 years"),
                               (20, 24, "20-24 years"), (25, 29, "25-29 years"), (30, 34, "30-34 years"),
                               (35, 39, "35-39 years"), (40, 44, "40-44 years"), (45, 49, "45-49 years"),
                               (50, 54, "50-54 years"), (55, 59, "55-59 years"), (60, 64, "60-64 years"),
                               (65, 69, "65-69 years"), (70, 74, "70-74 years")]:
            if lo <= age <= hi:
                return label
        return "75+ years"
    if "decade" in stage:
        return {"fourth": "40-49 years", "fifth": "50-59 years", "sixth": "60-69 years",
                "seventh": "70-79 years"}.get(
            next((k for k in ("fourth", "fifth", "sixth", "seventh") if k in stage), None), "80+ years")
    return "Unknown"


def run(input_path, output_path, value_counts_xlsx=None):
    adata = sc.read_h5ad(input_path)

    print('1. Bin BMI into standard clinical categories')
    adata.obs['BMI_Groups'] = pd.cut(
        adata.obs['BMI'], bins=[0, 18.5, 25.0, 30.0, 35.0, float('inf')],
        labels=['Underweight', 'Normal weight', 'Overweight', 'Obesity Class 1', 'Obesity Class 2+'], right=False)

    print('2. Standardise cause-of-death labels')
    adata.obs['cause_of_death'] = adata.obs['cause_of_death'].apply(_standardize_cause_of_death)

    print('3. Convert cause of death into binary columns')
    conditions = list(CAUSE_OF_DEATH_MAPPING.keys())
    binary_columns = adata.obs.apply(_generate_binary_columns, axis=1, conditions=conditions)
    binary_df = pd.DataFrame(binary_columns.tolist(), columns=conditions)
    binary_df.index = adata.obs.index.astype(str)
    df_final = pd.merge(binary_df, adata.obs, right_index=True, left_index=True)

    print('4-6. Replace "unknown" placeholders with true missing values')
    df_final.sex = df_final.sex.replace({'unknown': np.nan})
    df_final.self_reported_ethnicity = df_final.self_reported_ethnicity.replace({'unknown': np.nan})
    df_final.development_stage = df_final.development_stage.replace({'unknown': np.nan})

    print('7. Group developmental stage into age bins')
    df_final['age_group'] = df_final['development_stage'].apply(_map_development_stage_to_age_group)

    print('8. Convert string "nan" in age_range to true missing values')
    df_final.age_range = df_final.age_range.replace({'nan': np.nan})

    print('9. Integer-encode categorical columns')
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
