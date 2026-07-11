#!/usr/bin/env python3
"""
Compute per-context non-zero median gene expression references (Methods 9.4): "For
each gene g and context k, we computed a reference expression value as the non-zero
median expression of gene g across cells in the corresponding reference group,"
where the reference group for context k is all cells *not* in that context category
(the "out-ref" convention below). These medians are the normalisation factors used by
the tokenizer (`cascade/data/tokenizer.py`) to compute each cell's context-specific
fold-change ranking.

Restricts to protein-coding + miRNA genes (Geneformer convention; requires
`protein_coding_annotation.py` to have been run first) and computes one median dict
per context type: cell type, tissue, disease, and an overall ("ALL") pooled dict.
Writes `median_genes_{context}_all_{dataset}.pkl` for each, matching the filenames
`TranscriptomeTokenizer` expects.

Usage:
    python -m preprocessing.context_median_reference --dataset-name LUCA \
        --input-path $CASCADE_DATA_ROOT/processed/LUCA/adata_annotated_protein_coding.h5ad \
        --output-dir $CASCADE_DATA_ROOT/processed/LUCA
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import scanpy as sc
from tqdm import tqdm

CONTEXT_COLUMNS = {'cells': 'cell_type_ontology_term_id', 'tissue': 'tissue', 'disease': 'disease'}


def restrict_to_coding_genes(adata, coding_mirna_only=True):
    if not coding_mirna_only:
        return adata, adata.var["ensembl_gene_id"].tolist()
    loc = np.where((adata.var.gene_type == "protein_coding") | (adata.var.gene_type == "miRNA"))[0]
    coding_genes = adata.var["ensembl_gene_id"].values[loc].tolist()
    return adata[:, adata.var["ensembl_gene_id"].isin(coding_genes)], coding_genes


def nonzero_median(adata_subset, genes):
    """Non-zero median expression per gene, i.e. median over cells with detected
    (non-zero) expression of that gene."""
    data = np.where(adata_subset.X == 0, np.nan, adata_subset.X)
    medians = np.nanmedian(data, axis=0)
    return dict(zip(genes, medians))


def compute_out_ref_medians_per_context(adata, context_col, genes):
    """For each category in `context_col`, compute the non-zero median using all
    *other* cells as the reference group ("out-ref")."""
    result = {}
    for context in tqdm(sorted(set(adata.obs[context_col].dropna()))):
        subset = adata[adata.obs[context_col] != context, :]
        print(f"Out context {context} there are {subset.shape[0]} cells")
        result[context] = {"out-ref": nonzero_median(subset, genes)}
    return result


def run(dataset_name, input_path, output_dir, coding_mirna_only=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(input_path)
    adata, coding_genes = restrict_to_coding_genes(adata, coding_mirna_only=coding_mirna_only)

    # Data was log1p-normalised during QC (`qc_scanpy.py`); undo that here so medians
    # are computed on the target_sum=10,000-normalised (not log-scaled) counts.
    adata.X = (np.expm1(adata.X) * 10_000).astype(np.float32)

    for name, col in CONTEXT_COLUMNS.items():
        if col not in adata.obs:
            print(f"Skipping '{name}' context: column '{col}' not present")
            continue
        medians = compute_out_ref_medians_per_context(adata, col, coding_genes)
        with open(output_dir / f"median_genes_{name}_all_{dataset_name}.pkl", "wb") as f:
            pickle.dump(medians, f)

    print("Derive overall (ALL) reference median")
    overall = {"out-ref": nonzero_median(adata, coding_genes)}
    with open(output_dir / f"median_genes_all_all_{dataset_name}.pkl", "wb") as f:
        pickle.dump(overall, f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--input-path", type=Path, required=True,
                         help="adata_annotated_protein_coding.h5ad from protein_coding_annotation.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-coding-mirna-filter", action="store_true",
                         help="Use all genes instead of restricting to protein-coding + miRNA")
    args = parser.parse_args()
    run(args.dataset_name, args.input_path, args.output_dir, coding_mirna_only=not args.no_coding_mirna_filter)


if __name__ == "__main__":
    main()
