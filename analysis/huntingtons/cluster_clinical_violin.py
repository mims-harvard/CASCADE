#!/usr/bin/env python3
"""
Cluster clinical comparison, violin-plot version (supplementary to Figure 5f):
age, motor/cognitive onset, VS grade, and both CAG lengths, organised as
variables (rows) x CAG group (columns), cluster 0 vs cluster 1 per panel.

Usage:
    python -m analysis.huntingtons.cluster_clinical_violin \
        --cag1-npz HD_models_scale/CAG_1_1107.npz \
        --cag2-npz HD_models_scale/CAG_2_1107.npz \
        --donor-info donors_hh_info.csv --output-dir .
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from analysis.huntingtons.hd_attention_utils import cluster_k2, create_donor_df, load_donor_info, normalize_per_donor

C_COLORS = ['#AA3377', '#4477AA']
ALPHA = 0.55

VARS = [
    ('age', 'Age (years)', 'Age'),
    ('onset_motor', 'Motor onset (years)', 'Motor Onset'),
    ('onset_cognitive', 'Cognitive onset (years)', 'Cognitive Onset'),
    ('vs_grade', 'VS Grade', 'VS Grade'),
    ('cag1', 'Benign CAG (CAG₁)', 'Benign CAG'),
    ('cag2', 'Pathogenic CAG (CAG₂)', 'Pathogenic CAG'),
]


def extract_cag_map(data_dict):
    """Return {donor_id: cag_value} for all valid (non-NaN label) donors."""
    valid = ~np.isnan(data_dict["patient_y"].astype(float))
    ids_v = data_dict["patient_ids"][valid]
    y_v = data_dict["patient_y"][valid].astype(float)
    return {int(ids_v[i]): float(y_v[i]) for i in range(len(ids_v))}


def get_donor_table(data_dict, donor_info, cag1_map, cag2_map, seed=42):
    df = normalize_per_donor(create_donor_df(data_dict))
    pivot = df.pivot_table(index='donor_id', columns='cell_type', values='importance', aggfunc='mean')
    x_scaled = StandardScaler().fit_transform(np.nan_to_num(pivot.values, nan=0.0))
    labs = cluster_k2(x_scaled, seed=seed)
    sil = silhouette_score(x_scaled, labs)

    meta = donor_info.set_index('donor_id')[['age', 'vs_grade', 'onset_motor', 'onset_cognitive']]
    tbl = meta.copy()
    tbl['cag1'] = pd.Series(cag1_map)
    tbl['cag2'] = pd.Series(cag2_map)
    tbl['cluster'] = pd.Series(dict(zip(pivot.index, labs)))
    return tbl.loc[tbl['cluster'].notna()], sil


def sig_label(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def violin_pair(ax, tbl, var, ylabel):
    data = [tbl[tbl['cluster'] == c][var].dropna().values for c in [0, 1]]

    for xi, (vals, col) in enumerate(zip(data, C_COLORS)):
        if len(vals) < 2:
            continue
        vp = ax.violinplot([vals], positions=[xi], showmedians=False, showextrema=False, widths=0.55)
        for body in vp['bodies']:
            body.set_facecolor(col)
            body.set_alpha(ALPHA)
            body.set_edgecolor('none')

        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([xi - 0.12, xi + 0.12], [med, med], color=col, lw=2.2, zorder=4)
        ax.plot([xi, xi], [q1, q3], color=col, lw=1.5, solid_capstyle='round', zorder=3)

        rng = np.random.default_rng(42 + xi)
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(xi + jitter, vals, color=col, s=16, alpha=0.65, zorder=5, edgecolors='none')

    if len(data[0]) > 1 and len(data[1]) > 1:
        _, p = mannwhitneyu(data[0], data[1], alternative='two-sided')
        all_vals = np.concatenate(data)
        top = np.nanmax(all_vals)
        dy = (np.nanmax(all_vals) - np.nanmin(all_vals)) * 0.07
        ax.annotate('', xy=(1, top + dy * 0.8), xytext=(0, top + dy * 0.8),
                    arrowprops=dict(arrowstyle='-', color='#555555', lw=1.1))
        ax.text(0.5, top + dy * 1.2, sig_label(p), ha='center', va='bottom', fontsize=12,
                color='#333333', fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'C0\n(n={len(data[0])})', f'C1\n(n={len(data[1])})'], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=4)
    ax.tick_params(bottom=False)
    ax.set_xlim(-0.6, 1.6)


def main(cag1_npz, cag2_npz, donor_info_path, output_dir, seed=42):
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 13,
        'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
    })

    donor_info = load_donor_info(donor_info_path)
    data_cag1, data_cag2 = np.load(cag1_npz, allow_pickle=True), np.load(cag2_npz, allow_pickle=True)

    print('Computing clusters...')
    cag1_map, cag2_map = extract_cag_map(data_cag1), extract_cag_map(data_cag2)
    tbl1, sil1 = get_donor_table(data_cag1, donor_info, cag1_map, cag2_map, seed=seed)
    tbl2, sil2 = get_donor_table(data_cag2, donor_info, cag1_map, cag2_map, seed=seed)
    print(f'  Benign CAG    k=2  sil={sil1:.3f}')
    print(f'  Pathogenic CAG k=2 sil={sil2:.3f}')

    panels = [(f'Benign CAG\n(silhouette={sil1:.3f})', tbl1), (f'Pathogenic CAG\n(silhouette={sil2:.3f})', tbl2)]
    n_vars = len(VARS)
    fig, axes = plt.subplots(n_vars, 2, figsize=(8, 2.6 * n_vars), dpi=300)
    plt.subplots_adjust(hspace=0.55, wspace=0.38)

    for ci, (title, _) in enumerate(panels):
        axes[0, ci].set_title(title, fontsize=14, fontweight='bold', pad=10)

    for ri, (var, ylabel, varlabel) in enumerate(VARS):
        for ci, (_, tbl) in enumerate(panels):
            violin_pair(axes[ri, ci], tbl, var, ylabel)
        axes[ri, 0].annotate(varlabel, xy=(-0.28, 0.5), xycoords='axes fraction', fontsize=12,
                             fontweight='bold', color='#444444', va='center', ha='right', rotation=90)

    handles = [mpatches.Patch(facecolor=C_COLORS[i], alpha=0.75, label=f'Cluster {i}') for i in range(2)]
    sig_lines = [Line2D([], [], color='#333333', lw=1.1, label='bracket = MWU p-value')]
    fig.legend(handles=handles + sig_lines, loc='lower center', ncol=3, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, -0.015))

    output_dir = Path(output_dir)
    paper_ready_dir = output_dir / 'plots' / 'paper_ready'
    paper_ready_dir.mkdir(parents=True, exist_ok=True)
    for out in (output_dir / 'plots' / 'cluster_clinical_v2.png',
                paper_ready_dir / 'cluster_clinical_v2.png', paper_ready_dir / 'cluster_clinical_v2.pdf'):
        plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.15)
        print('Saved:', out)
    plt.close(fig)
    print('Done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cag1-npz", type=str, required=True)
    parser.add_argument("--cag2-npz", type=str, required=True)
    parser.add_argument("--donor-info", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.cag1_npz, args.cag2_npz, args.donor_info, args.output_dir, seed=args.seed)
