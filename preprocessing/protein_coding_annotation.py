#!/usr/bin/env python3
"""
Annotate each gene with its GENCODE gene_type/gene_level (protein_coding, miRNA,
etc.), used downstream to restrict tokenisation to protein-coding + miRNA genes
(Methods 9.4, following the Geneformer convention).

Reads the marker-gene-annotated AnnData (`adata_marker_genes.h5ad`, from
`clustering_umap.py`/`celltype_annotation.py`) and writes
`adata_annotated_protein_coding.h5ad`.

Usage:
    python -m preprocessing.protein_coding_annotation --dataset-name LUCA \
        --input-path $CASCADE_DATA_ROOT/processed/LUCA/adata_marker_genes.h5ad \
        --gencode-file $CASCADE_DATA_ROOT/reference/gencode.v44.chr_patch_hapl_scaff.annotation.gff3 \
        --output-dir $CASCADE_DATA_ROOT/processed/LUCA
"""
import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc


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
    genes["ensembl_gene_id"] = [x[0] for x in genes.ensembl_gene.str.split(".")]
    return genes


def run(input_path, gencode_file, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(input_path)
    gencode_genes = load_gencode_genes(gencode_file)

    missing_genes = len(set(adata.var["ensembl_gene_id"].values) - set(gencode_genes["ensembl_gene_id"]))
    print(f"{missing_genes} genes are missing gene_status and gene_type information")

    adata.var = pd.merge(adata.var, gencode_genes, on="ensembl_gene_id", how="left")
    adata.var = adata.var.drop(columns="gene_status")

    adata.write(output_dir / 'adata_annotated_protein_coding.h5ad')
    return adata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--gencode-file", type=Path, required=True,
                         help="Use the mouse GENCODE GFF3 (gencode.vM36...) for M2")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input_path, args.gencode_file, args.output_dir)


if __name__ == "__main__":
    main()
