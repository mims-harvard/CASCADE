#!/usr/bin/env python3
"""
Publication figure (Figure 5g): clinical differences between donor overlap
groups defined by concordance between the benign- and pathogenic-CAG k=2
attention-profile clusters (see cluster_overlap_analysis.py, which produces
the cluster_overlap_donors.csv this script consumes).

Usage:
    python -m analysis.huntingtons.cluster_overlap_figure \
        --overlap-csv cluster_overlap_donors.csv --output-dir .
"""
import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

GROUPS = ['Both-High', 'LowBenign-HighPath', 'Both-Low']
COLORS = {'Both-High': '#AA3377', 'LowBenign-HighPath': '#EE7733', 'Both-Low': '#4477AA'}
# Labels clarify that "cluster with higher/lower average VS grade" — not individual VS level
LABELS = {
    'Both-High': 'Higher-avg-VS\ncluster in both',
    'LowBenign-HighPath': 'Higher-avg-VS cluster\nin pathogenic only',
    'Both-Low': 'Lower-avg-VS\ncluster in both',
}
VARS = [
    ('vs_grade', 'VS grade', (0, 5.5)),
    ('age', 'Age at death (years)', (0, 115)),
    ('onset_motor', 'Motor onset (years)', (0, 100)),
]


def sig_label(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def all_pairwise(data, groups, var):
    results = {}
    for g1, g2 in itertools.combinations(groups, 2):
        a = data[data['overlap'] == g1][var].dropna().values
        b = data[data['overlap'] == g2][var].dropna().values
        p = mannwhitneyu(a, b, alternative='two-sided')[1] if (len(a) > 1 and len(b) > 1) else 1.0
        results[(g1, g2)] = (p, sig_label(p))
    return results


def main(overlap_csv, output_dir):
    df = pd.read_csv(overlap_csv)
    df = df[df['overlap'].isin(GROUPS)].copy()

    print("All pairwise MWU tests:\n")
    for varname, varlabel, _ in VARS:
        print(f"  {varlabel}:")
        pw = all_pairwise(df, GROUPS, varname)
        for (g1, g2), (p, star) in pw.items():
            a = df[df['overlap'] == g1][varname].dropna()
            b = df[df['overlap'] == g2][varname].dropna()
            print(f"    {g1} vs {g2}: {a.mean():.2f} vs {b.mean():.2f},  p={p:.4f}  {star}")
        print()

    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 11,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })

    fig, axes = plt.subplots(1, 3, figsize=(13, 6.5), dpi=300)
    plt.subplots_adjust(wspace=0.48, bottom=0.38, top=0.84)

    positions = {g: i for i, g in enumerate(GROUPS)}
    bar_w, bracket_lw, bracket_color = 0.55, 1.4, '#333333'

    for ax, (varname, varlabel, ylim) in zip(axes, VARS):
        pw = all_pairwise(df, GROUPS, varname)

        group_tops = {}
        for g in GROUPS:
            xi = positions[g]
            vals = df[df['overlap'] == g][varname].dropna().values
            if len(vals) == 0:
                group_tops[g] = 0
                continue
            m, s = np.mean(vals), np.std(vals)
            ax.bar(xi, m, bar_w, color=COLORS[g], alpha=0.82, edgecolor='white', linewidth=0.4, zorder=3)
            ax.errorbar(xi, m, yerr=s, fmt='none', color='#333333', capsize=4, capthick=1.2, elinewidth=1.1, zorder=4)
            rng = np.random.default_rng(xi + 7)
            jitter = rng.uniform(-0.18, 0.18, len(vals))
            ax.scatter(xi + jitter, vals, color=COLORS[g], s=26, alpha=0.75, zorder=5, edgecolors='none')
            n_all = len(df[df['overlap'] == g])
            ax.text(xi, ylim[1] * 0.02, f'n={n_all}', ha='center', fontsize=9, color='#555555')
            group_tops[g] = m + s

        sig_pairs = [(g1, g2, p, star) for (g1, g2), (p, star) in pw.items() if star != 'ns']
        sig_pairs.sort(key=lambda x: abs(positions[x[0]] - positions[x[1]]))

        tick_h = (ylim[1] - ylim[0]) * 0.03
        v_step = (ylim[1] - ylim[0]) * 0.10
        base_y = max(group_tops.values()) + (ylim[1] - ylim[0]) * 0.05

        for level, (g1, g2, p, star) in enumerate(sig_pairs):
            x0, x1 = positions[g1], positions[g2]
            y = base_y + level * v_step
            ax.plot([x0, x0, x1, x1], [y, y + tick_h, y + tick_h, y],
                    color=bracket_color, lw=bracket_lw, solid_capstyle='round', zorder=6)
            ax.text((x0 + x1) / 2, y + tick_h * 1.1, star, ha='center', va='bottom',
                    fontsize=12, fontweight='bold', color=bracket_color)

        ax.set_xticks(list(positions.values()))
        ax.set_xticklabels([LABELS[g] for g in GROUPS], rotation=35, ha='right', fontsize=10)
        ax.set_ylabel(varlabel, fontsize=12)
        ax.set_ylim(ylim)
        ax.grid(axis='y', alpha=0.25, linewidth=0.5, linestyle='--')
        ax.set_axisbelow(True)

    axes[0].set_title('VS grade', fontweight='bold', pad=18)
    axes[1].set_title('Age at death', fontweight='bold', pad=18)
    axes[2].set_title('Motor onset', fontweight='bold', pad=18)

    handles = [mpatches.Patch(color=COLORS[g], alpha=0.82, label=LABELS[g]) for g in GROUPS]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=11.5, frameon=False, bbox_to_anchor=(0.5, 0.14))
    fig.text(0.5, 0.06, 'Pairwise Mann-Whitney U test (two-sided) - * p<0.05   ** p<0.01   *** p<0.001',
             ha='center', va='bottom', fontsize=9.5, color='#555555', style='italic')

    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    paper_ready_dir = plots_dir / 'paper_ready'
    paper_ready_dir.mkdir(parents=True, exist_ok=True)
    for out in (plots_dir / 'cluster_overlap_clinical.png', plots_dir / 'cluster_overlap_clinical.pdf',
                paper_ready_dir / 'cluster_overlap_clinical.png', paper_ready_dir / 'cluster_overlap_clinical.pdf'):
        fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.2)
        print('Saved:', out)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap-csv", type=str, required=True,
                        help="cluster_overlap_donors.csv from cluster_overlap_analysis.py")
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()
    main(args.overlap_csv, args.output_dir)
