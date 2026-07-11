#!/usr/bin/env python3
"""
Seattle-AD-specific out-of-core QC/normalisation pipeline (Methods 9.2): "In case of
AD, pre-processing was completed using scarf due to the dataset size." Seattle-AD
(2.77M nuclei) doesn't fit the in-memory Scanpy pipeline used for the other datasets
(`qc_scanpy.py`), so filtering, gene annotation, mitochondrial QC, normalisation and
scaling are all done via scarf's zarr/Dask-backed DataStore instead.

This assumes a scarf zarr store has already been built from the raw h5ad (one-time
conversion, `convert_h5ad_to_zarr`) and gene annotation references (a cleaned BioMart
export and the GENCODE GFF3) are available locally.

Usage:
    python -m preprocessing.qc_scarf_seattle \
        --h5ad-path $CASCADE_DATA_ROOT/data/SEA_AD/adata_raw.h5ad \
        --zarr-path $CASCADE_DATA_ROOT/data/SEA_AD/adata_raw.zarr \
        --biomart-file $CASCADE_DATA_ROOT/reference/biomart_cleaned.csv \
        --gencode-file $CASCADE_DATA_ROOT/reference/gencode.v44.chr_patch_hapl_scaff.annotation.gff3 \
        --output-dir $CASCADE_DATA_ROOT/processed/SEA_AD
"""
import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc
import scarf
import scipy.sparse as sp
from tqdm import tqdm

DATASET_NAME = 'SEA_AD'


def convert_h5ad_to_zarr(h5ad_path, zarr_path, chunk_size=(2000, 1000)):
    """One-time conversion of the raw cellxgene h5ad into a scarf-compatible zarr
    store. Only needs to be run once per dataset; skip if `zarr_path` already exists."""
    reader = scarf.H5adReader(
        h5ad_path,
        cell_ids_key='index',
        feature_ids_key='ensembl_gene_id',
        feature_name_key='gene_names',
    )
    writer = scarf.H5adToZarr(reader, zarr_loc=zarr_path, chunk_size=chunk_size)
    writer.dump(batch_size=10000)


def gene_info(attribute):
    """Parse a GENCODE GFF3 attribute string into (gene_id, gene_name, gene_type,
    gene_status, gene_level)."""
    info_list = attribute.split(";")

    def field(key):
        matches = [entry for entry in info_list if key in entry]
        return matches[0].split("=")[1] if matches else None

    return field("ID"), field("gene_name"), field("gene_type"), field("gene_status"), field("level")


def load_gencode_genes(gencode_file):
    gencode = pd.read_table(
        gencode_file, comment="#", sep="\t",
        names=['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'],
    )
    genes = gencode[gencode.feature == "gene"][['seqname', 'start', 'end', 'attribute']].copy()
    genes = genes.reset_index(drop=True)
    print(f"In total there is information about {genes.shape[0]} genes")

    parsed = genes.attribute.apply(gene_info)
    (genes["ensembl_gene"], genes["gene_name"], genes["gene_type"],
     genes["gene_status"], genes["gene_level"]) = zip(*parsed)
    genes["_index"] = [x[0] for x in genes.ensembl_gene.str.split(".")]
    return genes


def run(h5ad_path, zarr_path, biomart_file, gencode_file, output_dir,
        min_cells_per_gene=200, max_counts=110_000, max_features=12_000, max_pct_mt=1.3):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata_obs = sc.read_h5ad(h5ad_path, backed='r').obs
    ds = scarf.DataStore(zarr_path)
    for col in adata_obs.columns:
        if col not in ds.cells.columns:
            ds.cells.insert(col, adata_obs[col].values)

    # Restrict to genes detected in enough cells (feature-level QC; note this
    # threshold operates on gene-level `nCells`, unlike the <3-cell gene filter used
    # in the Scanpy pipeline for the other datasets).
    feats_df = ds.RNA.feats.to_pandas_dataframe(columns=ds.RNA.feats.columns)
    genes_to_keep = feats_df.nCells >= min_cells_per_gene
    ds.RNA.rawData = ds.RNA.rawData[:, genes_to_keep]
    feats_df = feats_df[genes_to_keep]
    print(ds.RNA.rawData.shape)

    mart = pd.read_csv(biomart_file).rename(columns={'ensembl_gene_id': '_index'})
    merged_var = feats_df.merge(mart, how='left', on='_index').set_index('_index')

    for col in ('ncbi_gene_id', 'approved_symbol'):
        keep = ~merged_var[col].isnull()
        ds.RNA.rawData = ds.RNA.rawData[:, keep]
        merged_var = merged_var[keep]
        print(ds.RNA.rawData.shape)

    keep = ~merged_var['approved_symbol'].duplicated()
    ds.RNA.rawData = ds.RNA.rawData[:, keep]
    merged_var = merged_var[keep]
    print(ds.RNA.rawData.shape)

    gencode_genes = load_gencode_genes(gencode_file)
    merged_var = merged_var.reset_index()
    merged_var = pd.merge(merged_var, gencode_genes, on="_index", how="left").drop(columns="gene_status")
    ds.RNA.feats = ds.RNA.feats.__class__(merged_var)

    # Mitochondrial QC and cell filtering, matching the thresholds in Supplementary
    # Table S1 for Seattle-AD.
    ds.RNA.add_percent_feature(feat_pattern='MT-', name='RNA_percent')
    ds.plot_cells_dists(cols=['RNA_nCounts', 'RNA_nFeatures', 'RNA_percent'])
    ds.filter_cells(
        attrs=['RNA_nCounts', 'RNA_nFeatures', 'RNA_percent'],
        highs=[max_counts, max_features, max_pct_mt],
        lows=[200, 3, 0],
    )

    obs_df = ds.RNA.cells.to_pandas_dataframe(columns=ds.RNA.cells.columns)
    ds.RNA.rawData = ds.RNA.normed(log_transform=True)
    obs_df = obs_df.loc[obs_df.I == True, :]

    ds.RNA.rawData = (ds.RNA.rawData - ds.RNA.rawData.mean(axis=0)) / ds.RNA.rawData.std(axis=0)
    ds.RNA.cells = ds.RNA.cells.__class__(obs_df)
    ds.RNA.rawData = ds.RNA.rawData.astype('float32')
    ds.RNA.rawData = ds.RNA.rawData.rechunk((40000, ds.RNA.rawData.chunks[1]))

    print('Converting data to per-block AnnData chunks')
    var = ds.RNA.feats.to_pandas_dataframe(ds.RNA.feats.columns)
    obs = ds.RNA.cells.to_pandas_dataframe(ds.RNA.cells.columns)
    if "_index" in var.columns:
        var = var.rename(columns={"_index": "index_backup"})

    for i, block in tqdm(enumerate(ds.RNA.rawData.blocks), total=ds.RNA.rawData.numblocks[0]):
        expression_matrix = sp.csr_matrix(block)
        adata_chunk = sc.AnnData(
            X=expression_matrix, var=var, obs=obs[i * block.shape[0]: (i + 1) * block.shape[0]])
        adata_chunk.write_h5ad(output_dir / f"adata_annotated_protein_coding_fin_{i}.h5ad")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5ad-path", type=Path, required=True)
    parser.add_argument("--zarr-path", type=Path, required=True,
                         help="Pre-built scarf zarr store (see convert_h5ad_to_zarr)")
    parser.add_argument("--biomart-file", type=Path, required=True)
    parser.add_argument("--gencode-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-cells-per-gene", type=int, default=200)
    parser.add_argument("--max-counts", type=int, default=110_000)
    parser.add_argument("--max-features", type=int, default=12_000)
    parser.add_argument("--max-pct-mt", type=float, default=1.3)
    args = parser.parse_args()
    run(args.h5ad_path, args.zarr_path, args.biomart_file, args.gencode_file, args.output_dir,
        min_cells_per_gene=args.min_cells_per_gene, max_counts=args.max_counts,
        max_features=args.max_features, max_pct_mt=args.max_pct_mt)


if __name__ == "__main__":
    main()
