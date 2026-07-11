#!/usr/bin/env python3
"""
Assign a majority-vote cell-type label to each Leiden cluster (from `clustering_umap.py`)
by taking, for each cluster, the existing per-cell `cell_type` annotation that occurs
most often within it (Methods 9.2 UMAP/clustering QA step). Also renders one UMAP per
obs column for visual inspection of batch/biological variation.

Usage:
    python -m preprocessing.celltype_annotation --dataset-name LUCA \
        --input-path $CASCADE_DATA_ROOT/processed/LUCA/adata_marker_genes.h5ad \
        --output-dir $CASCADE_DATA_ROOT/processed/LUCA
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from tqdm import tqdm

AUTISM_METADATA_COLUMNS = [
    'cell_type', 'sample', 'donor_id', 'tissue', 'age', 'sex', 'disease',
    'Capbatch', 'Seqbatch', 'post-mortem interval (hours)',
]
SKIP_COLUMNS = {'is_primary_data', 'observation_joinid'}


def assign_majority_vote_celltype(genedata):
    common = genedata.obs.groupby(["leiden", "cell_type"]).size()
    ranked = pd.DataFrame(common.groupby(["leiden"]).rank(ascending=False).reset_index())
    ranked.columns = ["leiden", "cell_type", "rank"]
    top = ranked[ranked["rank"] == 1]
    names = dict(zip(top.leiden, top.cell_type))
    genedata.obs["clusters"] = genedata.obs["leiden"].replace(names)
    return genedata


def run(dataset_name, input_path, output_dir, figures_dir=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(figures_dir) if figures_dir else output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = figures_dir

    genedata = sc.read_h5ad(input_path)
    genedata.uns.setdefault('log1p', {})
    genedata.uns['log1p']["base"] = None

    if dataset_name == 'AUTISM':
        genedata.obs = genedata.obs.rename(
            columns={'diagnosis': 'disease', 'individual': 'donor_id', 'cluster': 'cell_type', 'region': 'tissue'})
        genedata.write(output_dir / 'adata_marker_genes.h5ad')

    genedata = assign_majority_vote_celltype(genedata)

    sc.pl.umap(genedata, color=["leiden", "clusters"], legend_fontsize="xx-small")
    plt.tight_layout()
    plt.savefig(figures_dir / f"umap_cells_{dataset_name}.png")

    metadata_plots = AUTISM_METADATA_COLUMNS if dataset_name == 'AUTISM' else genedata.obs.columns.tolist()
    for meta in tqdm(metadata_plots):
        if meta in SKIP_COLUMNS:
            continue
        print(meta)
        sc.pl.umap(genedata, color=[meta], save=f"_{meta}_{dataset_name}.png")

    return genedata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--input-path", type=Path, required=True, help="adata_marker_genes.h5ad from clustering_umap.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, default=None)
    args = parser.parse_args()
    run(args.dataset_name, args.input_path, args.output_dir, figures_dir=args.figures_dir)


if __name__ == "__main__":
    main()
