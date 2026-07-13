#!/usr/bin/env python3
"""
Donor-level stratified train/test split (Methods 9.11, Supplementary Note 11): splits
each dataset by donor (never splitting a donor's cells across train and test) while
ensuring every category of every clinical/demographic column has at least one donor
in the training set, then caps the training set at `train_cap` fraction of donors.
This is the generator for the fixed splits hardcoded in `cascade/data/splits.py`
(SPLITS_BY_DATASET) — re-running it is not required to reproduce the paper, since the
resulting donor lists are already checked in, but it documents how they were derived.

Usage:
    python -m preprocessing.donor_stratified_split --dataset-name LUCA \
        --input-path $CASCADE_DATA_ROOT/processed/LUCA/adata_annotated_protein_coding_clinical.h5ad
"""
import argparse
from pathlib import Path

import scanpy as sc
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Per-dataset columns whose categories must each be represented by at least one
# donor in the training set (Methods 9.11).
STRATIFICATION_COLUMNS_BY_DATASET = {
    'AUTISM': [
        'A08AA51', 'A11HA02', 'C02LC01', 'C02LC51', 'N02AA02', 'N02CX02', 'N03AF01', 'N03AG01',
        'N03AX14', 'N05AH04', 'N05CH01', 'N06AB03', 'N06AB06', 'N06AB08', 'N06CA03', 'S01EA04',
        'A08AA', 'A11HA', 'C02LC', 'N02AA', 'N02CX', 'N03AF', 'N03AG', 'N03AX', 'N05AH', 'N05CH',
        'N06AB', 'N06CA', 'S01EA', 'A08A', 'A11H', 'C02L', 'N02A', 'N02C', 'N03A', 'N05A', 'N05C',
        'N06A', 'N06C', 'S01E', 'A08', 'A11', 'C02', 'N02', 'N03', 'N05', 'N06', 'S01', 'A', 'C', 'N', 'S',
    ],
    'HLCA': [
        'Suspected Suicide', 'age_range', 'lung_condition', 'mixed_ancestry', 'dataset',
        'sequencing_platform', 'smoking_status', 'study', 'subject_type', 'assay', 'sex',
        'self_reported_ethnicity', 'BMI_Groups', 'Head Trauma', 'intracranial hemorrhage', 'stroke',
        'Anoxic Brain Injury', 'Drug Overdose', 'Accident', 'age_group', 'suspension_type',
    ],
    'M2': ['treatment', 'THR', 'batch'],
    'LUCA': [
        'ever_smoker', 'origin', 'dataset', 'tumor_stage', 'disease', 'tissue', 'EGFR_mutation',
        'TP53_mutation', 'ALK_mutation', 'BRAF_mutation', 'ERBB2_mutation', 'KRAS_mutation',
        'ROS_mutation', 'origin_fine', 'study', 'platform', 'assay', 'sex', 'uicc_stage_combined',
        'age-binned',
    ],
    'HH': ['batch', 'cell_type', 'disease', 'Age_decade', 'Sex', 'CAG', 'PMI-binned', 'VS_Grade', 'CAP', 'CAP-100'],
    'SEATTLE': [
        'ADNC', 'APOE4 status', 'Age at death', 'Braak stage', 'CERAD score', 'Class', 'Cognitive status',
        'LATE-NC stage', 'Lewy body disease pathology', 'Microinfarct pathology', 'PMI', 'Subclass',
        'Supertype', 'Thal phase', 'Years of education', 'assay', 'batch', 'cell_type',
        'development_stage', 'disease', 'donor_id', 'self_reported_ethnicity', 'sex',
    ],
}


def stratified_train_split(df, categorical_cols, donor_col, train_cap=0.7, test_size=0.3, random_state=42):
    """Split `df` into train/test by `donor_col`, ensuring every category in
    `categorical_cols` has at least one donor in train, then capping the training
    set at `train_cap` fraction of all donors."""
    train_donors = set()
    all_donors = df[donor_col].unique()
    total_donors = len(all_donors)

    for col in tqdm(categorical_cols):
        for value in df[col].dropna().unique():
            unique_donors = df[df[col] == value][donor_col].unique()
            if len(unique_donors) > 1:
                train_donors.add(unique_donors[0])

    max_train_donors = int(train_cap * total_donors)
    remaining_donors = list(set(all_donors) - train_donors)
    additional_train_needed = max_train_donors - len(train_donors)

    if additional_train_needed > 0:
        additional_train_donors, test_donors = train_test_split(
            remaining_donors, test_size=test_size, random_state=random_state)
        additional_train_donors = additional_train_donors[:additional_train_needed]
        train_donors.update(additional_train_donors)
        test_donors = list(set(remaining_donors) - set(additional_train_donors))
    else:
        test_donors = remaining_donors

    for col in tqdm(categorical_cols):
        for value in df[col].dropna().unique():
            if value not in df[df[donor_col].isin(train_donors)][col].values:
                missing_donors = df[(df[col] == value) & (~df[donor_col].isin(train_donors))][donor_col].unique()
                missing_donors = [d for d in missing_donors if d not in test_donors]
                if missing_donors:
                    train_donors.add(missing_donors[0])

    train_df = df[df[donor_col].isin(train_donors)]
    test_df = df[df[donor_col].isin(test_donors)]
    return train_df, test_df


def run(dataset_name, input_path, donor_col='donor_id', train_cap=0.7, test_size=0.3, random_state=42):
    if dataset_name not in STRATIFICATION_COLUMNS_BY_DATASET:
        raise ValueError(f"No stratification columns configured for '{dataset_name}'")

    adata = sc.read_h5ad(input_path)
    train_set, test_set = stratified_train_split(
        adata.obs, STRATIFICATION_COLUMNS_BY_DATASET[dataset_name], donor_col,
        train_cap=train_cap, test_size=test_size, random_state=random_state)

    assert not (set(train_set[donor_col]) & set(test_set[donor_col])), "donor leaked across train/test"
    print('Train donors', sorted(set(train_set[donor_col])))
    print('Test donors', sorted(set(test_set[donor_col])))
    return train_set, test_set


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-name", required=True, choices=sorted(STRATIFICATION_COLUMNS_BY_DATASET))
    parser.add_argument("--input-path", type=Path, required=True,
                         help="Output of the corresponding clinical_metadata_*.py script")
    parser.add_argument("--donor-col", default="donor_id")
    parser.add_argument("--train-cap", type=float, default=0.7)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    run(args.dataset_name, args.input_path, donor_col=args.donor_col, train_cap=args.train_cap,
        test_size=args.test_size, random_state=args.random_state)


if __name__ == "__main__":
    main()
