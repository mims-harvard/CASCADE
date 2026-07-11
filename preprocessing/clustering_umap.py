#!/usr/bin/env python3
"""
PCA, neighbour graph, Leiden clustering, and UMAP embedding, plus marker-gene ranking
per Leiden cluster (Methods 9.2: "Dimensionality reduction was performed using UMAP
to visualize cell-type identities, donor labels, and associated metadata, allowing
for qualitative assessment of batch and biological variation.").

Reads `adata_annotated.h5ad` (output of `gene_annotation.py`) and writes
`adata_marker_genes.h5ad` for use by `celltype_annotation.py`.

Usage:
    python -m preprocessing.clustering_umap --dataset-name LUCA \
        --input-path $CASCADE_DATA_ROOT/processed/LUCA/adata_annotated.h5ad \
        --output-dir $CASCADE_DATA_ROOT/processed/LUCA
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc


def run(dataset_name, input_path, output_dir, figures_dir=None, skip_marker_genes=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(figures_dir) if figures_dir else output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print('Load data')
    genedata = sc.read_h5ad(input_path)

    if dataset_name == 'M2':
        genedata.var.index = genedata.var.index.astype(str)
        genedata.var_names = genedata.var_names.astype(str)
        genedata.var['highly_variable'] = genedata.var['highly_variable'].astype(bool)

    # `log1p` bookkeeping gets dropped/mangled by some AnnData round-trips; scanpy
    # needs it set for downstream functions that check whether data is log-scaled.
    genedata.uns.setdefault('log1p', {})
    genedata.uns['log1p']["base"] = None
    genedata.raw = None

    print("1. Compute PCA")
    sc.tl.pca(genedata, svd_solver='arpack')
    sc.pl.pca(genedata)
    plt.tight_layout()
    plt.savefig(figures_dir / f"pca_plot_{dataset_name}.png")
    sc.pl.pca_variance_ratio(genedata, log=True)
    plt.tight_layout()
    plt.savefig(figures_dir / f"pca_plot_var_{dataset_name}.png")

    print("2. Calculate neighbouring graph")
    sc.pp.neighbors(genedata)

    print("3. Compute UMAP embedding")
    sc.tl.leiden(genedata)
    sc.tl.paga(genedata)
    sc.pl.paga(genedata, plot=False)
    sc.tl.umap(genedata, init_pos='paga')
    sc.pl.umap(genedata)
    plt.tight_layout()
    plt.savefig(figures_dir / f"umap_plot_{dataset_name}.png")

    print("4. Cluster the neighbour graph")
    sc.pl.umap(genedata, color=['leiden'])
    plt.tight_layout()
    plt.savefig(figures_dir / f"umap_plot_clusters_{dataset_name}.png")

    if not skip_marker_genes:
        print("5. Find marker genes")
        sc.tl.rank_genes_groups(genedata, 'leiden', method='t-test')
        sc.pl.rank_genes_groups(genedata, n_genes=25, sharey=False)
        plt.tight_layout()
        plt.savefig(figures_dir / f"ranked_genes_ttest_{dataset_name}.png")

        genedata.uns['log1p']["base"] = None
        sc.settings.verbosity = 2
        sc.tl.rank_genes_groups(genedata, 'leiden', method='wilcoxon')
        sc.pl.rank_genes_groups(genedata, n_genes=25, sharey=False)
        plt.tight_layout()
        plt.savefig(figures_dir / f"ranked_genes_wilcoxon_{dataset_name}.png")

    if dataset_name == 'AUTISM':
        genedata.obs = genedata.obs.rename(
            columns={'diagnosis': 'disease', 'individual': 'donor_id', 'cluster': 'cell_type'})
        genedata.obs['donor_id'] = genedata.obs['donor_id'].astype(str)

    genedata.write(output_dir / 'adata_marker_genes.h5ad')

    print("6. Visualise donor effect")
    if 'donor_id' in genedata.obs:
        sc.pl.umap(genedata, color=['donor_id'])
        plt.tight_layout()
        plt.savefig(figures_dir / f"umap_plot_donors_{dataset_name}.png")
    return genedata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--input-path", type=Path, required=True, help="adata_annotated.h5ad from gene_annotation.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--skip-marker-genes", action="store_true",
                         help="Skip rank_genes_groups (matches original HLCA runs, which skipped this step)")
    args = parser.parse_args()
    run(args.dataset_name, args.input_path, args.output_dir,
        figures_dir=args.figures_dir, skip_marker_genes=args.skip_marker_genes)


if __name__ == "__main__":
    main()
