#!/usr/bin/env python3
"""
Silhouette plots for the k=2 attention-profile clustering (supplementary to
Figure 5f): per-donor silhouette coefficients for benign and pathogenic CAG
clusters, individually and combined.

Usage:
    python -m analysis.huntingtons.silhouette_plots \
        --cag1-npz HD_models_scale/CAG_1_1107.npz \
        --cag2-npz HD_models_scale/CAG_2_1107.npz --output-dir .
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from analysis.huntingtons.hd_attention_utils import cluster_k2, create_donor_df, normalize_per_donor

CLUSTER_COLORS = ['#AA3377', '#4477AA']


def build_feature_matrix(df):
    pivot = df.pivot_table(index='donor_id', columns='cell_type', values='importance', aggfunc='mean')
    return StandardScaler().fit_transform(np.nan_to_num(pivot.values, nan=0.0)), pivot.index.tolist()


def silhouette_panel(ax, x_scaled, labels, k, title):
    sil_vals = silhouette_samples(x_scaled, labels)
    avg_sil = silhouette_score(x_scaled, labels)

    y_lower = 10
    for cl in range(k):
        cl_sil = np.sort(sil_vals[labels == cl])
        size = len(cl_sil)
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cl_sil, facecolor=CLUSTER_COLORS[cl],
                         alpha=0.85, edgecolor='none')
        cl_name = 'A' if cl == 0 else 'B'
        ax.text(-0.07, y_lower + size / 2, f'Cluster {cl_name}\n(n={size})', ha='right', va='center',
                fontsize=11, color=CLUSTER_COLORS[cl], fontweight='bold')
        y_lower = y_upper + 10

    ax.axvline(avg_sil, color='#333333', linestyle='--', linewidth=1.5)
    ax.text(avg_sil + 0.01, y_lower * 0.5, f'avg = {avg_sil:.3f}', va='center', fontsize=11, color='#333333')

    ax.set_xlim(-0.25, 1.0)
    ax.set_ylim(0, y_lower + 10)
    ax.set_xlabel('Silhouette coefficient', fontsize=13, labelpad=6)
    ax.set_yticks([])
    ax.set_title(title, fontsize=15, fontweight='bold', pad=8)
    ax.axvline(0, color='#888888', linewidth=0.8)
    ax.grid(axis='x', alpha=0.2, linewidth=0.6)


def main(cag1_npz, cag2_npz, output_dir, seed=42):
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
        'xtick.labelsize': 12, 'ytick.labelsize': 11,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    data = {'Benign CAG': np.load(cag1_npz, allow_pickle=True), 'Pathogenic CAG': np.load(cag2_npz, allow_pickle=True)}
    fname_map = {'Benign CAG': 'silhouette_benign_cag', 'Pathogenic CAG': 'silhouette_pathogenic_cag'}

    output_dir = Path(output_dir)
    paper_ready_dir = output_dir / 'plots' / 'paper_ready'
    paper_ready_dir.mkdir(parents=True, exist_ok=True)

    for tag, data_dict in data.items():
        df = normalize_per_donor(create_donor_df(data_dict))
        x_scaled, _ = build_feature_matrix(df)
        labs = cluster_k2(x_scaled, seed=seed)

        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        silhouette_panel(ax, x_scaled, labs, 2, tag)
        legend_handles = [mpatches.Patch(facecolor=CLUSTER_COLORS[i], label=f'Cluster {"A" if i == 0 else "B"}', alpha=0.85)
                          for i in range(2)]
        ax.legend(handles=legend_handles, loc='lower right', fontsize=11, frameon=False)

        plt.tight_layout(pad=1.0)
        fname = fname_map[tag]
        for out in (output_dir / 'plots' / f'{fname}.png', paper_ready_dir / f'{fname}.png', paper_ready_dir / f'{fname}.pdf'):
            plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.12)
            print('Saved:', out)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
    plt.subplots_adjust(wspace=0.32)
    for ax, (tag, data_dict) in zip(axes, data.items()):
        df = normalize_per_donor(create_donor_df(data_dict))
        x_scaled, _ = build_feature_matrix(df)
        labs = cluster_k2(x_scaled, seed=seed)
        silhouette_panel(ax, x_scaled, labs, 2, tag)

    handles = [mpatches.Patch(facecolor=CLUSTER_COLORS[i], label=f'Cluster {i}', alpha=0.85) for i in range(2)]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    for out in (output_dir / 'plots' / 'silhouette_combined.png', paper_ready_dir / 'silhouette_combined.png',
                paper_ready_dir / 'silhouette_combined.pdf'):
        plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.12)
        print('Saved:', out)
    plt.close(fig)
    print('Done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cag1-npz", type=str, required=True)
    parser.add_argument("--cag2-npz", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.cag1_npz, args.cag2_npz, args.output_dir, seed=args.seed)
