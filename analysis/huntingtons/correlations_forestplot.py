#!/usr/bin/env python3
"""
Forest plot of donor-specific cell-type importance vs. clinical variables
(Figure 5d): Spearman r with 95% bootstrap CI, two panels (Benign / Pathogenic
CAG), clinical variables as colours, cell types on the y-axis.

Usage:
    python -m analysis.huntingtons.correlations_forestplot \
        --cag1-npz HD_models_scale/CAG_1_1107.npz \
        --cag2-npz HD_models_scale/CAG_2_1107.npz \
        --donor-info donors_hh_info.csv --output-dir .
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory
from scipy.stats import spearmanr

from analysis.huntingtons.hd_attention_utils import create_donor_df, load_donor_info, normalize_per_donor

warnings.filterwarnings('ignore')

N_BOOT = 1000

CLIN_VARS = {'age': 'Age', 'vs_grade': 'VS Grade', 'onset_motor': 'Motor Onset', 'onset_cognitive': 'Cognitive Onset'}
CLIN_COLORS = {'age': '#4477AA', 'vs_grade': '#AA3377', 'onset_motor': '#D55E00', 'onset_cognitive': '#009E73'}
TAGS = ['CAG_1', 'CAG_2']
TAG_LABELS = {'CAG_1': 'Benign CAG', 'CAG_2': 'Pathogenic CAG'}


def bootstrap_ci(x, y, rng, n_boot=N_BOOT, ci=95):
    n = len(x)
    boot = [spearmanr(x[rng.integers(0, n, n)], y[rng.integers(0, n, n)])[0] for _ in range(n_boot)]
    boot = np.array(boot)
    return np.percentile(boot, (100 - ci) / 2), np.percentile(boot, 100 - (100 - ci) / 2)


def compute_correlations(data_dict, donor_info, rng):
    df = normalize_per_donor(create_donor_df(data_dict))
    df = df.merge(donor_info[['donor_id'] + list(CLIN_VARS.keys())], on='donor_id', how='left')
    by_donor = df.groupby(['donor_id', 'cell_type'])['importance'].mean().reset_index()
    meta = df.groupby('donor_id')[list(CLIN_VARS.keys())].first().reset_index()
    merged = by_donor.merge(meta, on='donor_id', how='left')

    rows = []
    for ct in sorted(merged['cell_type'].unique()):
        cd = merged[merged['cell_type'] == ct]
        for clin_col, clin_lbl in CLIN_VARS.items():
            mask = cd[clin_col].notna() & cd['importance'].notna()
            if mask.sum() < 5:
                continue
            x, y = cd.loc[mask, 'importance'].values, cd.loc[mask, clin_col].values
            r, p = spearmanr(x, y)
            lo, hi = bootstrap_ci(x, y, rng)
            rows.append({'cell_type': ct, 'clin_var': clin_col, 'clin_lbl': clin_lbl,
                        'r': r, 'p': p, 'ci_lo': lo, 'ci_hi': hi, 'n': mask.sum()})
    return pd.DataFrame(rows)


def main(cag1_npz, cag2_npz, donor_info_path, output_dir, seed=42):
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 15,
        'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })

    rng = np.random.default_rng(seed)
    donor_info = load_donor_info(donor_info_path)
    data = {'CAG_1': np.load(cag1_npz, allow_pickle=True), 'CAG_2': np.load(cag2_npz, allow_pickle=True)}

    print("Computing correlations + bootstrap CIs...")
    corr = {tag: compute_correlations(d, donor_info, rng) for tag, d in data.items()}

    all_cts = sorted(set(corr['CAG_1']['cell_type']) | set(corr['CAG_2']['cell_type']))
    mean_abs_r = {
        ct: np.mean([corr[t][corr[t]['cell_type'] == ct]['r'].abs().mean()
                    for t in TAGS if not corr[t][corr[t]['cell_type'] == ct].empty])
        for ct in all_cts
    }
    all_cts = sorted(all_cts, key=lambda c: mean_abs_r.get(c, 0))

    n_ct, n_clin = len(all_cts), len(CLIN_VARS)
    ct_idx = {ct: i for i, ct in enumerate(all_cts)}
    offsets = np.linspace(-0.28, 0.28, n_clin)
    clin_order = list(CLIN_VARS.keys())
    band_colors = ['#f7f7f7', '#ffffff']

    fig, axes = plt.subplots(1, 2, figsize=(12, max(6, n_ct * 1.05 + 2.2)), dpi=300, sharey=True)
    plt.subplots_adjust(wspace=0.04, left=0.18)

    for ax, tag in zip(axes, TAGS):
        df = corr[tag]
        for i in range(n_ct):
            ax.axhspan(i - 0.5, i + 0.5, color=band_colors[i % 2], zorder=0, lw=0)
        ax.axvline(0, color='#888888', lw=1.0, ls='--', zorder=1)

        for ci, clin_col in enumerate(clin_order):
            color, offset = CLIN_COLORS[clin_col], offsets[ci]
            sub = df[df['clin_var'] == clin_col]
            for _, row in sub.iterrows():
                yi = ct_idx[row['cell_type']] + offset
                ax.plot([row['ci_lo'], row['ci_hi']], [yi, yi], color=color, lw=1.8, alpha=0.75,
                        solid_capstyle='round', zorder=2)
                marker, ms = ('D', 7.5) if row['p'] < 0.05 else ('o', 6)
                ax.plot(row['r'], yi, marker=marker, color=color, ms=ms, zorder=3,
                        markeredgecolor='white', markeredgewidth=0.7)

        ax.set_yticks(range(n_ct))
        ax.set_ylim(-0.6, n_ct - 0.4)
        ax.set_xlim(-1.18, 1.18)
        ax.set_xlabel("Spearman's r  (95% bootstrap CI)", fontsize=15, labelpad=6)
        ax.set_title(TAG_LABELS[tag], fontsize=18, fontweight='bold', pad=8)
        ax.grid(axis='x', alpha=0.2, linewidth=0.5, linestyle='--', zorder=1)
        ax.set_axisbelow(False)
        ax.tick_params(left=False, bottom=True)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        for sp in ax.spines.values():
            sp.set_visible(False)

    axes[0].set_yticklabels([])
    axes[0].set_ylabel('')
    axes[1].set_yticklabels([])

    trans = blended_transform_factory(axes[0].transAxes, axes[0].transData)

    def fmt_ct(ct):
        return ct.upper() if ct.lower() == 'spn' else ct.capitalize()

    for i, ct in enumerate(all_cts):
        axes[0].text(-0.04, i, fmt_ct(ct), ha='right', va='center', fontsize=15, color='#222222', zorder=5, transform=trans)

    clin_patches = [mpatches.Patch(color=CLIN_COLORS[c], label=CLIN_VARS[c]) for c in clin_order]
    sig_dot = plt.Line2D([], [], marker='D', color='#555555', ms=7, markeredgecolor='white', ls='none', label='p < 0.05')
    nosig_dot = plt.Line2D([], [], marker='o', color='#aaaaaa', ms=5.5, markeredgecolor='white', ls='none', label='p ≥ 0.05')
    fig.legend(handles=clin_patches + [sig_dot, nosig_dot], loc='lower center', ncol=6, fontsize=13, frameon=False,
               bbox_to_anchor=(0.5, -0.06), handlelength=1.0, handletextpad=0.5, columnspacing=0.9)

    output_dir = Path(output_dir)
    paper_ready_dir = output_dir / 'plots' / 'paper_ready'
    paper_ready_dir.mkdir(parents=True, exist_ok=True)
    for out in (output_dir / 'plots' / 'correlation_forestplot.png', paper_ready_dir / 'correlation_forestplot.png'):
        plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.18)
        print(f'Saved: {out}')
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cag1-npz", type=str, required=True)
    parser.add_argument("--cag2-npz", type=str, required=True)
    parser.add_argument("--donor-info", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.cag1_npz, args.cag2_npz, args.donor_info, args.output_dir, seed=args.seed)
