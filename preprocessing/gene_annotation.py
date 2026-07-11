#!/usr/bin/env python3
"""
Standardise gene annotation across datasets via BioMart (Methods 9.2): "we retrieved
metadata from BioMart to obtain standardized GENE SYMBOL, ENSEMBL GENE ID, and NCBI
GENE ID for each gene ... We removed genes with duplicated ENSEMBL GENE ID, missing
NCBI GENE ID, and missing or duplicated GENE SYMBOL."

Reads `adata_scaled.h5ad` (output of `qc_scanpy.py`/`qc_scanpy_autism.py`), merges in
BioMart annotation, drops genes failing the above criteria, and re-indexes by gene
symbol. HLCA is processed in three memory-bounded chunks and re-concatenated (it OOMs
otherwise); M2 (mouse) uses a GTF-derived annotation table instead of human BioMart.

Fixes vs. the original lab script: the non-HLCA/M2 path re-added an 'ensembl_gene_id'
column via `var['ensembl_gene_id'] = var.index` immediately before `reset_index()` -
since the index was already named 'ensembl_gene_id' at that point, `reset_index()`
alone tries to create a column of the same name, and pandas raises
`ValueError: cannot insert ensembl_gene_id, already exists`. Confirmed with a
synthetic LUCA-shaped run (the original would crash deterministically, not just on
edge-case data). Fixed by dropping the redundant assignment.

Usage:
    python -m preprocessing.gene_annotation --dataset-name LUCA \
        --input-path $CASCADE_DATA_ROOT/processed/LUCA/adata_scaled.h5ad \
        --biomart-file $CASCADE_DATA_ROOT/reference/results.txt \
        --output-dir $CASCADE_DATA_ROOT/processed/LUCA
"""
import argparse
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


def load_or_build_biomart(biomart_file, cleaned_cache):
    """Load a pre-cleaned BioMart export, or build+cache one from a raw BioMart
    `results.txt` export (deduplicated by Ensembl ID, missing NCBI IDs dropped,
    columns >50% missing dropped)."""
    if cleaned_cache.is_file():
        return pd.read_csv(cleaned_cache)

    mart = pd.read_csv(biomart_file, sep="\t", low_memory=False)
    mart.columns = [re.sub(r'\s+', '_', re.sub(r'\s\([^)]+\)', '', c.lower())) for c in mart.columns]

    mart = mart.drop_duplicates(['ensembl_gene_id']).reset_index(drop=True)
    print("Removing duplicate Ensembl IDs: {}".format(mart.shape))

    mart = mart.dropna(subset=['ncbi_gene_id'])
    mart['ncbi_gene_id'] = mart['ncbi_gene_id'].astype(int)
    print("Removing missing NCBI gene IDs: {}".format(mart.shape))

    threshold = int(mart.shape[0] * 0.5)
    mart = mart.dropna(thresh=threshold, axis=1)
    mart.to_csv(cleaned_cache, index=False)
    return mart


def build_m2_mart(gtf_file):
    """M2 (mouse) has no BioMart export; build an equivalent annotation table by
    parsing gene_id/gene_name attributes out of the Ensembl GTF directly."""
    keys_to_extract = [
        "gene_id", "gene_version", "gene_name", "transcript_id", "transcript_version",
        "exon_number", "ccds_id", "protein_id", "protein_version",
    ]
    fixed_columns = ["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attributes"]
    data = pd.read_csv(gtf_file, sep="\t", comment="#", names=fixed_columns, header=None, low_memory=False)

    patterns = {key: re.compile(fr'{key} "([^"]+)"') for key in keys_to_extract}
    extracted = {key: [] for key in keys_to_extract}
    for attr in data["attributes"]:
        for key, pattern in patterns.items():
            match = pattern.search(attr)
            extracted[key].append(match.group(1) if match else None)

    final_df = pd.concat([data.drop(columns=["attributes"]), pd.DataFrame(extracted)], axis=1)
    mart = final_df.rename(columns={'gene_id': 'ensembl_gene_id'})
    threshold = int(mart.shape[0] * 0.5)
    mart = mart.dropna(thresh=threshold, axis=1)
    return mart.drop_duplicates(subset=['ensembl_gene_id'])


def _clean_chunk(chunk, symbol_col='approved_symbol'):
    """Drop genes with missing/duplicated gene symbols from one AnnData chunk."""
    pre = chunk.n_vars
    chunk = chunk[:, ~chunk.var[symbol_col].isnull()]
    print(f'Removing missing gene symbols: {pre - chunk.n_vars}')
    pre = chunk.n_vars
    chunk = chunk[:, ~chunk.var[symbol_col].duplicated()]
    print(f'Removing duplicated gene symbols: {pre - chunk.n_vars}')
    for col in ('refseq_accession', 'pubmed_id'):
        if col in chunk.var:
            del chunk.var[col]
    chunk.var.reset_index(inplace=True)
    chunk.var.set_index(symbol_col, inplace=True)
    return chunk.copy()


def run(dataset_name, input_path, output_dir, biomart_file=None, gtf_file=None, n_hlca_chunks=3):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    genedata = sc.read_h5ad(input_path)

    if dataset_name == 'M2':
        mart = build_m2_mart(gtf_file)
    else:
        mart = load_or_build_biomart(biomart_file, output_dir / "biomart_cleaned.csv")

    genedata.var.index = [re.sub(r'\.[0-9]+', '', x) for x in genedata.var.index]
    if dataset_name in ('AUTISM', 'M2'):
        genedata.var = genedata.var.set_index('ensembl_gene_id')

    genedata_genes = set(genedata.var_names)
    mart_genes = set(mart['ensembl_gene_id'])
    print('Number of genes in adata:', len(genedata_genes))
    print('Number of genes in annotation:', len(mart_genes))
    print('Number of genes in both:', len(mart_genes.intersection(genedata_genes)))

    if dataset_name == 'AUTISM':
        mart = mart.drop(columns='approved_symbol')

    if dataset_name == 'M2':
        merged_var = pd.merge(genedata.var, mart, on='ensembl_gene_id', how='left')
    else:
        merged_var = genedata.var.merge(mart, how='left', left_index=True, right_on='ensembl_gene_id')
        assert np.all(genedata.var.index == merged_var['ensembl_gene_id'])

    genedata.var = merged_var.set_index('ensembl_gene_id')

    pre_filtered_genes = genedata.n_vars
    print('Number of genes before filtering: {:d}'.format(pre_filtered_genes))

    if dataset_name != 'M2':
        genedata = genedata[:, ~genedata.var['ncbi_gene_id'].isnull()]
        print('Removing missing NCBI gene IDs: {:d}'.format(pre_filtered_genes - genedata.n_vars))

    symbol_col = 'feature_name' if dataset_name == 'M2' else 'approved_symbol'

    if dataset_name == 'HLCA':
        # HLCA is too large to clean in memory in one pass; split into `n_hlca_chunks`
        # random cell chunks, clean each independently, and re-concatenate.
        n_obs = genedata.n_obs
        random_indices = np.random.permutation(n_obs)
        split_points = [n_obs * i // n_hlca_chunks for i in range(1, n_hlca_chunks)]
        chunks = np.split(random_indices, split_points)

        cleaned = []
        for i, idx in enumerate(chunks):
            print(f'Cleaning chunk {i}')
            chunk = _clean_chunk(genedata[idx], symbol_col=symbol_col)
            chunk.write(output_dir / f'adata_annotated_chunk_{i}.h5ad')
            cleaned.append(chunk)
        genedata = cleaned[0]
        for chunk in cleaned[1:]:
            genedata = ad.concat([genedata, chunk], axis=0)
    else:
        pre = genedata.n_vars
        genedata = genedata[:, ~genedata.var[symbol_col].isnull()]
        print('Removing missing gene symbols: {:d}'.format(pre - genedata.n_vars))
        pre = genedata.n_vars
        genedata = genedata[:, ~genedata.var[symbol_col].duplicated()]
        print('Removing duplicated gene symbols: {:d}'.format(pre - genedata.n_vars))

        genedata.var.reset_index(inplace=True)  # var.index is already named 'ensembl_gene_id'
        genedata.var = genedata.var.set_index(symbol_col)
        if dataset_name == 'M2':
            genedata.var = genedata.var.astype(str).drop(columns='transcript_id')

    print('Number of genes after filtering: {:d}'.format(genedata.n_vars))
    genedata.write(output_dir / 'adata_annotated.h5ad')
    return genedata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--input-path", type=Path, required=True, help="adata_scaled.h5ad from qc_scanpy.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--biomart-file", type=Path, default=None, help="Raw BioMart results.txt export")
    parser.add_argument("--gtf-file", type=Path, default=None, help="Required for --dataset-name M2")
    parser.add_argument("--n-hlca-chunks", type=int, default=3)
    args = parser.parse_args()
    run(args.dataset_name, args.input_path, args.output_dir, biomart_file=args.biomart_file,
        gtf_file=args.gtf_file, n_hlca_chunks=args.n_hlca_chunks)


if __name__ == "__main__":
    main()
