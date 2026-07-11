#!/usr/bin/env python3
"""
Donor-level variability in cell-type attention importance for benign and
pathogenic CAG prediction (Figure 5e): per-donor heatmaps (cell type x
donor, donors sorted by CAG length), with cluster-membership and CAG-length
annotation strips, showing that different cell types drive predictions in
different donors.

Usage:
    python -m analysis.huntingtons.donor_heatmaps \
        --cag1-npz HD_models_scale/CAG_1_1107.npz \
        --cag2-npz HD_models_scale/CAG_2_1107.npz --output-dir .
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler

from analysis.huntingtons.hd_attention_utils import cluster_best_k, create_donor_df, normalize_per_donor

warnings.filterwarnings('ignore')

HM_CMAP = 'RdYlGn_r'
CLUSTER_COLORS = ['#AA3377', '#4477AA', '#EE7733', '#33BBEE', '#EE3377', '#BBBBBB', '#228833', '#000000']
N_CAG_BINS = 8
CAG_STRIP_CMAP = 'YlGn'


def cluster_donors(df_norm, seed=42):
    pivot = df_norm.pivot_table(index='donor_id', columns='cell_type', values='importance', aggfunc='mean')
    x_scaled = StandardScaler().fit_transform(np.nan_to_num(pivot.values, nan=0.0))
    labels, best_k, best_sil = cluster_best_k(x_scaled, seed=seed)
    print(f'  Optimal k={best_k} (silhouette={best_sil:.3f})')
    return dict(zip(pivot.index, labels)), best_k


def build_heatmap(df):
    """Returns (hm, y_map): hm is cell_type x donor, sorted by label value."""
    pivot = df.groupby(['donor_id', 'cell_type'])['importance'].mean().reset_index()
    hm = pivot.pivot(index='cell_type', columns='donor_id', values='importance')
    y_map = df.groupby('donor_id')['y_value'].first()
    hm = hm[sorted(hm.columns, key=lambda d: y_map.get(d, 0))]
    for col in hm.columns:
        v = hm[col].dropna()
        mn, mx = v.min(), v.max()
        hm[col] = (hm[col] - mn) / (mx - mn) if mx > mn else 0.5
    return hm, y_map


def annotation_rows(hm, cluster_map, y_map):
    donors = list(hm.columns)
    cl_row = np.array([float(cluster_map.get(d, 0)) for d in donors]).reshape(1, -1)
    cag_row = np.array([float(y_map.get(d, np.nan)) for d in donors]).reshape(1, -1)
    return cl_row, cag_row


def main(cag1_npz, cag2_npz, output_dir, seed=42):
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 18, 'axes.titlesize': 20, 'axes.labelsize': 18,
        'xtick.labelsize': 15, 'ytick.labelsize': 17,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })

    print("DONOR-SPECIFIC HEATMAPS: Benign CAG, Pathogenic CAG")
    data_cag1, data_cag2 = np.load(cag1_npz, allow_pickle=True), np.load(cag2_npz, allow_pickle=True)

    print("Processing Benign CAG...")
    df1 = normalize_per_donor(create_donor_df(data_cag1))
    cluster_map1, k1 = cluster_donors(df1, seed=seed)
    hm1, y1 = build_heatmap(df1)

    print("Processing Pathogenic CAG...")
    df2 = normalize_per_donor(create_donor_df(data_cag2))
    cluster_map2, k2 = cluster_donors(df2, seed=seed)
    hm2, y2 = build_heatmap(df2)

    all_ct = sorted(set(hm1.index) | set(hm2.index))
    all_ct_disp = [ct[0].upper() + ct[1:] for ct in all_ct]
    n_ct = len(all_ct)

    cl1, cag1 = annotation_rows(hm1, cluster_map1, y1)
    cl2, cag2 = annotation_rows(hm2, cluster_map2, y2)

    all_cag = np.concatenate([cag1.flatten(), cag2.flatten()])
    all_cag = all_cag[~np.isnan(all_cag)]
    cag_vmin, cag_vmax = all_cag.min(), all_cag.max()
    cag_bin_edges = np.linspace(cag_vmin, cag_vmax, N_CAG_BINS + 1)

    def discretise_cag(cag_row):
        return np.digitize(cag_row, cag_bin_edges[1:-1]).astype(float)

    cag1_disc, cag2_disc = discretise_cag(cag1), discretise_cag(cag2)

    cag_colors_arr = plt.get_cmap(CAG_STRIP_CMAP)(np.linspace(0.05, 0.95, N_CAG_BINS))
    cag_listed = mcolors.ListedColormap(cag_colors_arr)
    cag_norm = mcolors.BoundaryNorm(np.arange(-0.5, N_CAG_BINS + 0.5, 1), N_CAG_BINS)

    fig = plt.figure(figsize=(16, 7.5), dpi=300)
    gs = GridSpec(3, 3, figure=fig, height_ratios=[n_ct, 0.35, 0.35], width_ratios=[10, 10, 1.8],
                  hspace=0.04, wspace=0.12)

    panels = [('Benign CAG', hm1, cl1, cag1_disc, k1), ('Pathogenic CAG', hm2, cl2, cag2_disc, k2)]
    im_hm_last, im_cag_last = None, None

    for ci, (title, hm, cl_row, cag_disc_row, k) in enumerate(panels):
        ax_hm, ax_cl, ax_cag = fig.add_subplot(gs[0, ci]), fig.add_subplot(gs[1, ci]), fig.add_subplot(gs[2, ci])

        hm_arr = hm.reindex(all_ct).values
        im_hm = ax_hm.imshow(hm_arr, aspect='auto', cmap=HM_CMAP, vmin=0, vmax=1, interpolation='none')
        im_hm_last = im_hm

        ax_hm.set_title(title, fontsize=20, fontweight='bold', pad=6)
        ax_hm.set_xticks([])
        ax_hm.set_yticks(range(n_ct))
        if ci == 0:
            ax_hm.set_yticklabels(all_ct_disp, fontsize=17)
            ax_hm.set_ylabel('Cell type', fontsize=18, labelpad=6)
        else:
            ax_hm.set_yticks([])
        for sp in ax_hm.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color('#cccccc')

        n_cl = max(k, 2)
        cl_cmap = mcolors.ListedColormap(CLUSTER_COLORS[:n_cl])
        cl_norm = mcolors.BoundaryNorm(np.arange(-0.5, n_cl + 0.5, 1), n_cl)
        ax_cl.imshow(cl_row, aspect='auto', cmap=cl_cmap, norm=cl_norm, interpolation='none')
        ax_cl.set_xticks([])
        ax_cl.set_yticks([0])
        if ci == 0:
            ax_cl.set_yticklabels(['Cluster'], fontsize=15, va='center')
        else:
            ax_cl.set_yticks([])
        for sp in ax_cl.spines.values():
            sp.set_linewidth(0.8)
            sp.set_color('#888888')

        im_cag = ax_cag.imshow(cag_disc_row, aspect='auto', cmap=cag_listed, norm=cag_norm, interpolation='none')
        im_cag_last = im_cag
        ax_cag.set_xticks([])
        ax_cag.set_yticks([0])
        if ci == 0:
            ax_cag.set_yticklabels(['CAG length'], fontsize=15, va='center')
        else:
            ax_cag.set_yticks([])
        ax_cag.set_xlabel('Patients', fontsize=16, labelpad=5)
        for sp in ax_cag.spines.values():
            sp.set_linewidth(0.8)
            sp.set_color('#888888')

    ax_cb0 = fig.add_subplot(gs[0, 2])
    ax_cb0.axis('off')
    cbar_main = fig.colorbar(im_hm_last, ax=ax_cb0, fraction=0.85, shrink=0.85, aspect=22)
    cbar_main.set_label('Importance', fontsize=15)
    cbar_main.set_ticks([0, 0.5, 1])
    cbar_main.ax.tick_params(labelsize=13)

    ax_cb1 = fig.add_subplot(gs[1, 2])
    ax_cb1.axis('off')
    cluster_handles = [mpatches.Patch(color=CLUSTER_COLORS[i], label=f'C{i}') for i in range(2)]
    ax_cb1.legend(handles=cluster_handles, loc='center left', fontsize=13, frameon=False,
                 handlelength=1.2, handletextpad=0.4, borderpad=0, labelspacing=0.3)

    ax_cb2 = fig.add_subplot(gs[2, 2])
    ax_cb2.axis('off')
    cbar_cag = fig.colorbar(im_cag_last, ax=ax_cb2, fraction=0.85, shrink=0.85, aspect=4, orientation='vertical')
    cbar_cag.set_label('CAG length', fontsize=13, labelpad=4)
    cbar_cag.set_ticks([0, N_CAG_BINS - 1])
    cbar_cag.set_ticklabels([f'{cag_vmin:.0f}', f'{cag_vmax:.0f}'])
    cbar_cag.ax.tick_params(labelsize=11)

    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / 'donor_heatmaps_notebook_style.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print(f'Saved: {out}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cag1-npz", type=str, required=True)
    parser.add_argument("--cag2-npz", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.cag1_npz, args.cag2_npz, args.output_dir, seed=args.seed)
