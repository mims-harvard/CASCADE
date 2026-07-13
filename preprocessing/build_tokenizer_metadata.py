#!/usr/bin/env python3
"""
Builds the two pickle files `cascade.data.tokenizer.TranscriptomeTokenizer` needs
(Methods 9.4) and then runs tokenisation for one context:

- `tokenizer_dictionary_{dataset}.pkl`: special tokens (<pad>, <ctx>, <cls>, <ref>,
  <up>, <down> - fixed at IDs 0-5, matching the defaults in
  `cascade.data.collator.DataCollatorContrastiveLearning`) plus one token ID per gene
  in the context's median-reference gene universe (from `context_median_reference.py`).
- `metadata_dictionary_{dataset}.pkl`: maps each obs column that should be preserved
  as a custom attribute in the tokenized dataset to a filesystem/Arrow-safe column
  name (spaces replaced with underscores), merged with the common context columns
  (cell_type, disease, tissue and their ontology term IDs).

Usage:
    python -m preprocessing.build_tokenizer_metadata --dataset-name LUCA --context CELLS \
        --clinical-dir $CASCADE_DATA_ROOT/processed/LUCA \
        --data-dir $CASCADE_DATA_ROOT/processed/LUCA/adata_objects \
        --output-dir $CASCADE_DATA_ROOT/processed/LUCA \
        --output-prefix tokenized_data_CELLS_out-ref
"""
import argparse
import pickle
from pathlib import Path

from cascade.data.tokenizer import TranscriptomeTokenizer

SPECIAL_TOKENS = ["<pad>", "<ctx>", "<cls>", "<ref>", "<up>", "<down>"]

# Per-dataset obs columns preserved as custom attributes in the tokenized dataset,
# beyond the common context columns (Methods 9.3 clinical metadata columns, one list
# per dataset actually used in the paper).
DATASET_METADATA_COLUMNS = {
    'AUTISM': [
        'sample', 'donor_id', 'age', 'sex', 'Capbatch', 'Seqbatch', 'Diagnosis', 'Age', 'Sex', 'PMI',
        'Other diagnoses', 'Medications', 'ADI-R-A', 'ADI-R-Bverbal', 'ADI-R-Bnonverbal', 'ADI-R-C',
        'ADI-R-D', 'Cause of death', 'Epilepsy', 'Attention-deficit/hyperactivity disorder',
        'Cardiac malformation', 'Depression', 'Pneumonia', 'Lead poisoning', 'Cerebellar Heterotopia',
        'Developmental Delay', 'binned', 'post-mortem-binned',
    ],
    'HLCA': [
        'donor_id', 'BMI', 'age_range', 'anatomical_region_ccf_score', 'cause_of_death', 'dataset',
        'BMI_Groups', 'Head Trauma', 'intracranial hemorrhage', 'stroke', 'Anoxic Brain Injury',
        'Drug Overdose', 'Accident', 'Suspected Suicide', 'age_group', 'suspension_type', 'sample',
        'sequencing_platform', 'smoking_status', 'study', 'subject_type', 'assay', 'sex',
        'self_reported_ethnicity', 'development_stage',
    ],
    'M2': ['donor_id', 'treatment', 'Cre', 'THR', 'THR_expr', 'batch'],
    'LUCA': [
        'sample', 'uicc_stage', 'ever_smoker', 'age', 'donor_id', 'origin', 'dataset', 'tumor_stage',
        'EGFR_mutation', 'TP53_mutation', 'ALK_mutation', 'BRAF_mutation', 'ERBB2_mutation',
        'KRAS_mutation', 'ROS_mutation', 'origin_fine', 'study', 'platform', 'suspension_type', 'assay',
        'sex', 'uicc_stage_combined', 'age-binned',
    ],
    'HH': [
        'batch', 'sample', 'Donor', 'Age', 'Sex', 'CAG', 'PMI', 'VS_Grade', 'Onset/Motor', 'Onset/Cog',
        'CAP', 'CAP-100', 'Age_decade', 'PMI-binned', 'CAG_1', 'CAG_2', 'CAG_max', 'CAG_min', 'CAG_diff',
    ],
    'SEATTLE': [
        'ADNC', 'APOE4 status', 'Age at death', 'Braak stage', 'CERAD score', 'Class', 'Cognitive status',
        'Continuous Pseudo-progression Score', 'LATE-NC stage', 'Lewy body disease pathology',
        'Microinfarct pathology', 'PMI', 'Specimen ID', 'Subclass', 'Supertype', 'Thal phase',
        'Years of education', 'assay', 'donor_id',
    ],
}
COMMON_CONTEXT_COLUMNS = {
    "cell_type": "cell_type", "disease": "disease", "tissue": "tissue",
    "cell_type_ontology_term_id": "cell_type_ontology_term_id",
    "disease_ontology_term_id": "disease_ontology_term_id",
    "tissue_ontology_term_id": "tissue_ontology_term_id",
}


def build_custom_attr_dict(dataset_name):
    dataset_cols = DATASET_METADATA_COLUMNS[dataset_name]
    dataset_dict = {col: col.replace(" ", "_") for col in dataset_cols}
    return {**dataset_dict, **COMMON_CONTEXT_COLUMNS}


def build_token_dictionary(median_reference_dict):
    """Special tokens (fixed IDs 0-5) followed by one token per gene in the
    context's median-reference gene universe."""
    special_dict = {name: idx for idx, name in enumerate(SPECIAL_TOKENS)}
    first_key = next(iter(median_reference_dict))
    gene_dict = {
        gene: idx + len(SPECIAL_TOKENS)
        for idx, gene in enumerate(median_reference_dict[first_key]["out-ref"].keys())
    }
    return {**special_dict, **gene_dict}


def run(dataset_name, context, clinical_dir, data_dir, output_dir, output_prefix,
        reference="out-ref", chunk_size=5000, batch_size=20000, file_format="h5ad"):
    clinical_dir = Path(clinical_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    custom_attr_dict = build_custom_attr_dict(dataset_name)
    with open(output_dir / f"metadata_dictionary_{dataset_name}.pkl", "wb") as f:
        pickle.dump(custom_attr_dict, f)

    context_file_by_name = {'DISEASE': 'disease', 'TISSUE': 'tissue', 'CELLS': 'cells', 'ALL': 'all'}
    median_reference_path = clinical_dir / f"median_genes_{context_file_by_name[context]}_all_{dataset_name}.pkl"
    with open(median_reference_path, "rb") as f:
        median_reference_dict = pickle.load(f)

    token_dictionary = build_token_dictionary(median_reference_dict)
    token_dictionary_file = output_dir / f"tokenizer_dictionary_{dataset_name}.pkl"
    with open(token_dictionary_file, "wb") as f:
        pickle.dump(token_dictionary, f)

    tokenizer = TranscriptomeTokenizer(
        output_dir=clinical_dir, dataset_name=dataset_name, custom_attr_name_dict=custom_attr_dict,
        nproc=1, chunk_size=chunk_size, batch_size=batch_size,
        token_dictionary_file=token_dictionary_file, context=context, reference=reference,
    )
    tokenizer.tokenize_data(
        data_directory=data_dir, output_directory=output_dir, output_prefix=output_prefix, file_format=file_format)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-name", required=True, choices=sorted(DATASET_METADATA_COLUMNS))
    parser.add_argument("--context", required=True, choices=["ALL", "DISEASE", "TISSUE", "CELLS"])
    parser.add_argument("--clinical-dir", type=Path, required=True,
                         help="Directory with the clinical_metadata_*.py output and median_genes_*.pkl files")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory of per-chunk .h5ad files to tokenize")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--reference", default="out-ref")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--file-format", default="h5ad", choices=["h5ad", "loom"])
    args = parser.parse_args()
    run(args.dataset_name, args.context, args.clinical_dir, args.data_dir, args.output_dir, args.output_prefix,
        reference=args.reference, chunk_size=args.chunk_size, batch_size=args.batch_size, file_format=args.file_format)


if __name__ == "__main__":
    main()
