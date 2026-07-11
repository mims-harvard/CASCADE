#!/usr/bin/env python3
"""
Aggregated cell-type attention importance across VS-grade stages (Figure 5c):
dot-plot of per-cell-type importance (colour = rank-normalized importance,
ring = peak HD grade) from Healthy/Control through VS grades 0-4.

Usage:
    python -m analysis.huntingtons.vsgrade_importance_plot \
        --vsgrade-npz HD_models_scale/VSGRADE_1107.npz --output-dir .
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import rankdata

warnings.filterwarnings('ignore')

GRADE_LABELS = {-1: 'Healthy', 0: 'VS Grade 0', 1: 'VS Grade 1', 2: 'VS Grade 2', 3: 'VS Grade 3', 4: 'VS Grade 4'}

# Donor-level Case/Control annotation (VS grade alone doesn't distinguish
# healthy controls from HD cases at grade 0; this labels controls as -1).
DONOR_HEALTHY_MAP = {
    43.0: 'Case', 69.0: 'Control', 88.0: 'Control', 29.0: 'Case', 37.0: 'Case',
    70.0: 'Control', 45.0: 'Case', 101.0: 'Control', 84.0: 'Control', 49.0: 'Case',
    5.0: 'Case', 80.0: 'Control', 58.0: 'Control', 100.0: 'Control', 41.0: 'Case',
    79.0: 'Control', 10.0: 'Case', 53.0: 'Control', 42.0: 'Case', 66.0: 'Control',
    3.0: 'Case', 13.0: 'Case', 16.0: 'Case', 57.0: 'Control', 63.0: 'Control',
    33.0: 'Case', 75.0: 'Control', 81.0: 'Control', 4.0: 'Case', 22.0: 'Case',
    9.0: 'Case', 55.0: 'Control', 90.0: 'Control', 39.0: 'Case', 72.0: 'Control',
    82.0: 'Control', 31.0: 'Case', 96.0: 'Control', 11.0: 'Case', 8.0: 'Case',
    24.0: 'Case', 2.0: 'Case', 102.0: 'Control', 14.0: 'Case', 64.0: 'Control',
    91.0: 'Control', 97.0: 'Control', 44.0: 'Case', 7.0: 'Case', 71.0: 'Control',
    77.0: 'Control', 92.0: 'Control', 98.0: 'Control', 62.0: 'Control', 32.0: 'Case',
    95.0: 'Control', 50.0: 'Control', 78.0: 'Control', 17.0: 'Case', 83.0: 'Control',
    23.0: 'Case', 20.0: 'Case', 76.0: 'Control', 61.0: 'Control', 19.0: 'Case',
    99.0: 'Control', 0.0: 'Case', 94.0: 'Control', 34.0: 'Case', 1.0: 'Case',
    48.0: 'Case', 60.0: 'Control', 86.0: 'Control', 46.0: 'Case', 28.0: 'Case',
    52.0: 'Control', 93.0: 'Control', 18.0: 'Case', 68.0: 'Control', 35.0: 'Case',
    26.0: 'Case', 38.0: 'Case', 51.0: 'Control', 54.0: 'Control', 59.0: 'Control',
    73.0: 'Control', 25.0: 'Case', 85.0: 'Control', 47.0: 'Case', 56.0: 'Control',
    87.0: 'Control', 65.0: 'Control', 89.0: 'Control', 12.0: 'Case', 21.0: 'Case',
    6.0: 'Case', 67.0: 'Control', 40.0: 'Case', 30.0: 'Case', 36.0: 'Case',
    74.0: 'Control', 27.0: 'Case', 15.0: 'Case',
}


def load_labeled_data(vsgrade_npz):
    data_npz = np.load(vsgrade_npz, allow_pickle=True)
    data = {
        'attention_weights': data_npz['attention_weights'],
        'patient_ids': data_npz['patient_ids'],
        'patient_y': data_npz['patient_y'].copy().astype(float),
        'patient_cell_types': data_npz['patient_cell_types'],
    }
    for donor_id, status in DONOR_HEALTHY_MAP.items():
        idx = int(donor_id)
        if status == 'Control' and idx < len(data['patient_y']):
            data['patient_y'][idx] = -1.0
    return data


def fmt_ct(ct):
    return ct.upper() if ct.lower() == 'spn' else ct.capitalize()


def main(vsgrade_npz, output_dir):
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 15, 'axes.titlesize': 17, 'axes.labelsize': 16,
        'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.major.size': 4, 'ytick.major.size': 4,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })

    data = load_labeled_data(vsgrade_npz)

    valid = ~np.isnan(data['patient_y'])
    x = data['attention_weights'][:, valid, :, :]
    y = data['patient_y'][valid].astype(int)
    pct = data['patient_cell_types'][valid]

    ctu = sorted({str(c) for pt in pct for c in pt})
    ct_measures = {yv: {ct: [] for ct in ctu} for yv in np.unique(y)}

    for i in range(x.shape[1]):
        max_z = x[:, i, 0, :].max(axis=0)
        max_z = (max_z - np.nanmean(max_z)) / (np.std(max_z) + 1e-7)
        for ct_j, score in zip([str(c) for c in pct[i]], max_z.tolist()):
            ct_measures[y[i]][ct_j].append(score)

    rows = [{'cell_type': ct, 'score': np.mean(scores), 'grade': grade}
            for grade, cell_scores in ct_measures.items() for ct, scores in cell_scores.items() if scores]
    df_raw = pd.DataFrame(rows)

    all_grades = sorted(df_raw['grade'].unique())
    grade_labels = [GRADE_LABELS.get(g, str(g)) for g in all_grades]
    cell_types = sorted(df_raw['cell_type'].unique())

    pivot_size = df_raw.pivot(index='cell_type', columns='grade', values='score')

    flat = pivot_size.values.flatten()
    mask = ~np.isnan(flat)
    ranks = np.full_like(flat, np.nan)
    ranks[mask] = (rankdata(flat[mask]) - 1) / (mask.sum() - 1)
    pivot_color = pd.DataFrame(ranks.reshape(pivot_size.shape), index=pivot_size.index, columns=pivot_size.columns)

    hd_grades = [g for g in all_grades if g >= 0]
    pivot_hd = pivot_size[[g for g in hd_grades if g in pivot_size.columns]]
    peak_grade = {ct: pivot_hd.loc[ct].idxmax() for ct in cell_types if ct in pivot_hd.index}

    gap, hd_step = 1.2, 0.52
    xpos = {all_grades[0]: 0}
    for k, g in enumerate(all_grades[1:], start=1):
        xpos[g] = gap + (k - 1) * hd_step

    n_ct = len(cell_types)
    x_max = max(xpos.values())

    fig = plt.figure(figsize=(10.5, 4.8), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=(6, 1.6), wspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis('off')

    for g, xi in xpos.items():
        ax.axvline(xi, color='#e8e8e8', lw=0.8, zorder=0)
    for yi in range(n_ct):
        ax.axhline(yi, color='#e8e8e8', lw=0.8, zorder=0)

    ctrl_x = xpos[all_grades[0]]
    if len(all_grades) > 1:
        ax.axvline((ctrl_x + xpos[all_grades[1]]) / 2, color='#bbbbbb', lw=1.2, ls='--', zorder=1)

    cmap = plt.get_cmap('RdYlGn').reversed()
    dot_size = 300

    for yi, ct in enumerate(cell_types):
        for g in all_grades:
            color_val = pivot_color.loc[ct, g] if g in pivot_color.columns else np.nan
            if np.isnan(color_val):
                continue
            ax.scatter(xpos[g], yi, s=dot_size, color=cmap(color_val), edgecolors='white',
                       linewidths=0.6, zorder=3, alpha=0.95)

    for yi, ct in enumerate(cell_types):
        pg = peak_grade.get(ct)
        if pg is not None and pg in xpos:
            ax.scatter(xpos[pg], yi, s=dot_size * 2.4, facecolors='none', edgecolors='#111111', linewidths=2.0, zorder=4)

    ax.set_xticks([xpos[g] for g in all_grades])
    ax.set_xticklabels(grade_labels, fontsize=14, rotation=35, ha='right')
    ax.set_yticks(range(n_ct))
    ax.set_yticklabels([fmt_ct(ct) for ct in cell_types], fontsize=14)
    ax.set_xlim(ctrl_x - 0.55, x_max + 0.45)
    ax.set_ylim(-0.6, n_ct - 0.4)
    ax.set_ylabel('Cell type', labelpad=6, fontsize=16)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cax = inset_axes(ax_leg, width='18%', height='55%', loc='upper left',
                     bbox_to_anchor=(0.0, 0, 1, 1), bbox_transform=ax_leg.transAxes, borderpad=0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Importance\n(rank-normalized)', fontsize=13, labelpad=6)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['Low', 'Mid', 'High'])
    cbar.ax.tick_params(labelsize=12)

    ring_hdl = [Line2D([], [], marker='o', color='w', markerfacecolor='none', markeredgecolor='#111111',
                       markeredgewidth=1.8, markersize=12, label='Peak HD grade\n(per cell type)')]
    ax_leg.legend(handles=ring_hdl, loc='lower left', bbox_to_anchor=(0.0, 0.0), frameon=True,
                 framealpha=0.95, edgecolor='#cccccc', fontsize=12, handletextpad=0.5)

    output_dir = Path(output_dir)
    paper_ready_dir = output_dir / 'plots' / 'paper_ready'
    paper_ready_dir.mkdir(parents=True, exist_ok=True)
    for out in (output_dir / 'plots' / 'cell_type_importance_vsgrade_combined_sidebyside.png',
                paper_ready_dir / 'cell_type_importance_vsgrade_combined_sidebyside.png'):
        plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.15)
        print(f'Saved: {out}')
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vsgrade-npz", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()
    main(args.vsgrade_npz, args.output_dir)
