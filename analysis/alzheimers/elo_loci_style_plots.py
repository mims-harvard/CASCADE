#!/usr/bin/env python3
"""
Correlation between LLM-arena-derived and CASCADE-derived cell-type
importance (Figure 3b-c; Methods 2.3), at granular (11-group dumbbell) and
coarse (7-group scatter) cell-type resolution.

Reads the Elo CSVs produced by elo_score.py (results_{intermediate,major}_
{ad,control}_elo_elo.csv) and correlates |AD - Control| Elo against CASCADE's
own AD-vs-Control absolute-change cell-type importance.

  elo_scatter_individual_absolute_change.png/pdf
      Dumbbell: 11 intermediate groups on x-axis, percentile rank on y-axis.
      Circle  = CASCADE absolute change percentile rank
      Diamond = |AD - Control| Elo percentile rank

  elo_scatter_major_absolute_change.png/pdf
      Percentile-rank scatter: 7 major groups.
      x = CASCADE absolute change - percentile rank
      y = |AD - Control| Elo - percentile rank

Note: the CASCADE cell-type importance values below (cl_xai, CASC_ABS_MAJOR)
are the already-computed CASCADE-Explainer output values for this analysis;
this script does not recompute them from raw attention weights.

Usage:
    python -m analysis.alzheimers.elo_loci_style_plots \
        --elo-dir $CASCADE_DATA_ROOT/SEATTLE/gene_explainer_validation \
        --output-dir $CASCADE_DATA_ROOT/SEATTLE/gene_explainer_validation
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import rankdata, spearmanr

from cascade.explainer.config import CASCADE_DATA_ROOT

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 17, 'axes.titlesize': 18, 'axes.labelsize': 17,
    'xtick.labelsize': 16, 'ytick.labelsize': 16,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

# ── CASCADE-Explainer cell-type importance (AD vs. Control), granular ──────────
CL_XAI = {
    'normal': {
        'L2/3-6 intratelencephalic projecting glutamatergic neuron': 3.056481110336459e-05,
        'L5 extratelencephalic projecting glutamatergic cortical neuron': 2.494518746613235e-05,
        'L6b glutamatergic cortical neuron': 2.0655870887747518e-05,
        'VIP GABAergic cortical interneuron': 5.058877837674863e-05,
        'astrocyte of the cerebral cortex': 2.203111061233098e-05,
        'caudal ganglionic eminence derived interneuron': 3.2378874418186486e-05,
        'cerebral cortex endothelial cell': 2.6958171808241565e-05,
        'chandelier pvalb GABAergic cortical interneuron': 2.890697296781795e-05,
        'corticothalamic-projecting glutamatergic cortical neuron': 2.1238196934509294e-05,
        'lamp5 GABAergic cortical interneuron': 3.9493278898224634e-05,
        'microglial cell': 4.236469856917454e-05,
        'near-projecting glutamatergic cortical neuron': 2.1634302288399166e-05,
        'oligodendrocyte': 1.6661245259988576e-05,
        'oligodendrocyte precursor cell': 1.8172856383800037e-05,
        'pvalb GABAergic cortical interneuron': 1.915442777330497e-05,
        'sncg GABAergic cortical interneuron': 5.103827731695919e-05,
        'sst GABAergic cortical interneuron': 2.0506218951523697e-05,
        'vascular leptomeningeal cell': 2.6101062985255774e-05,
    },
    'dementia': {
        'L2/3-6 intratelencephalic projecting glutamatergic neuron': 3.569007017091766e-05,
        'L5 extratelencephalic projecting glutamatergic cortical neuron': 3.864478042309867e-05,
        'L6b glutamatergic cortical neuron': 2.255325537505328e-05,
        'VIP GABAergic cortical interneuron': 2.8237634844235484e-05,
        'astrocyte of the cerebral cortex': 1.9768162310224534e-05,
        'caudal ganglionic eminence derived interneuron': 2.7270667395194703e-05,
        'cerebral cortex endothelial cell': 3.102333066700878e-05,
        'chandelier pvalb GABAergic cortical interneuron': 2.2844742901881207e-05,
        'corticothalamic-projecting glutamatergic cortical neuron': 2.0061280509065554e-05,
        'lamp5 GABAergic cortical interneuron': 3.253126666334407e-05,
        'microglial cell': 2.8027125494176734e-05,
        'near-projecting glutamatergic cortical neuron': 3.381629791868076e-05,
        'oligodendrocyte': 2.0432116635441367e-05,
        'oligodendrocyte precursor cell': 1.9062329444776067e-05,
        'pvalb GABAergic cortical interneuron': 1.7234400921161922e-05,
        'sncg GABAergic cortical interneuron': 2.6424232025305686e-05,
        'sst GABAergic cortical interneuron': 2.8473551101269566e-05,
        'vascular leptomeningeal cell': 5.0773862737278566e-05,
    },
}

# ── Intermediate group -> granular members ──────────────────────────────────────
INTERMEDIATE_MAP = {
    'microglial cell': ['microglial cell'],
    'astrocyte of the cerebral cortex': ['astrocyte of the cerebral cortex'],
    'oligodendrocyte': ['oligodendrocyte'],
    'oligodendrocyte precursor cell': ['oligodendrocyte precursor cell'],
    'cerebral cortex endothelial cell': ['cerebral cortex endothelial cell'],
    'vascular leptomeningeal cell': ['vascular leptomeningeal cell'],
    'glutamatergic excitatory cortical neuron': [
        'L2/3-6 intratelencephalic projecting glutamatergic neuron',
        'L5 extratelencephalic projecting glutamatergic cortical neuron',
        'L6b glutamatergic cortical neuron',
        'corticothalamic-projecting glutamatergic cortical neuron',
        'near-projecting glutamatergic cortical neuron',
    ],
    'parvalbumin GABAergic cortical interneuron': [
        'chandelier pvalb GABAergic cortical interneuron',
        'pvalb GABAergic cortical interneuron',
    ],
    'somatostatin GABAergic cortical interneuron': ['sst GABAergic cortical interneuron'],
    'VIP GABAergic cortical interneuron': ['VIP GABAergic cortical interneuron'],
    'other GABAergic cortical interneuron': [
        'lamp5 GABAergic cortical interneuron',
        'caudal ganglionic eminence derived interneuron',
        'sncg GABAergic cortical interneuron',
    ],
}

GROUP_COLORS = {
    'Microglia': '#4477AA', 'Inhibitory': '#EE6677', 'Excitatory': '#CCBB44',
    'Astrocytes': '#228833', 'Oligodendrocytes': '#AA3377', 'OPC': '#66CCEE', 'Endothelial': '#BBBBBB',
}
GROUP_BG = {
    'Microglia': '#EEF3FA', 'Inhibitory': '#FDF0F1', 'Excitatory': '#FAFAEE',
    'Astrocytes': '#EDF7EE', 'Oligodendrocytes': '#F7EEF5', 'OPC': '#EEF9FC', 'Endothelial': '#F7F7F7',
}
GROUP_ORDER = ['Microglia', 'Inhibitory', 'Excitatory', 'Astrocytes', 'Oligodendrocytes', 'OPC', 'Endothelial']

INT_MAJOR = {
    'microglial cell': 'Microglia',
    'parvalbumin GABAergic cortical interneuron': 'Inhibitory',
    'somatostatin GABAergic cortical interneuron': 'Inhibitory',
    'VIP GABAergic cortical interneuron': 'Inhibitory',
    'other GABAergic cortical interneuron': 'Inhibitory',
    'glutamatergic excitatory cortical neuron': 'Excitatory',
    'astrocyte of the cerebral cortex': 'Astrocytes',
    'oligodendrocyte': 'Oligodendrocytes',
    'oligodendrocyte precursor cell': 'OPC',
    'cerebral cortex endothelial cell': 'Endothelial',
    'vascular leptomeningeal cell': 'Endothelial',
}

SHORT_LABEL = {
    'microglial cell': 'Microglia',
    'parvalbumin GABAergic cortical interneuron': 'Pvalb',
    'somatostatin GABAergic cortical interneuron': 'SST',
    'VIP GABAergic cortical interneuron': 'VIP',
    'other GABAergic cortical interneuron': 'Other IN',
    'glutamatergic excitatory cortical neuron': 'Excitatory',
    'astrocyte of the cerebral cortex': 'Astrocyte',
    'oligodendrocyte': 'Oligodendrocyte',
    'oligodendrocyte precursor cell': 'OPC',
    'cerebral cortex endothelial cell': 'Endothelial',
    'vascular leptomeningeal cell': 'VLMC',
}

# ── CASCADE-Explainer cell-type importance (AD vs. Control), major groups ──────
CASC_ABS_MAJOR = {
    'microglial cell': 1.433757e-05,
    'astrocyte of the cerebral cortex': 2.262948e-06,
    'oligodendrocyte': 3.770871e-06,
    'oligodendrocyte precursor cell': 8.894731e-07,
    'cerebral cortex endothelial cell': 4.065159e-06,
    'glutamatergic excitatory cortical neuron': 6.816230e-06,
    'GABAergic inhibitory cortical interneuron': 1.071214e-05,
}
MAJ_LABEL = {
    'microglial cell': 'Microglia',
    'astrocyte of the cerebral cortex': 'Astrocytes',
    'oligodendrocyte': 'Oligodendrocytes',
    'oligodendrocyte precursor cell': 'Oligodendrocyte\nprogenitor cells',
    'cerebral cortex endothelial cell': 'Endothelial cells',
    'glutamatergic excitatory cortical neuron': 'Excitatory neurons',
    'GABAergic inhibitory cortical interneuron': 'Inhibitory neurons',
}
MAJ_COLOR = {
    'microglial cell': GROUP_COLORS['Microglia'],
    'astrocyte of the cerebral cortex': GROUP_COLORS['Astrocytes'],
    'oligodendrocyte': GROUP_COLORS['Oligodendrocytes'],
    'oligodendrocyte precursor cell': GROUP_COLORS['OPC'],
    'cerebral cortex endothelial cell': GROUP_COLORS['Endothelial'],
    'glutamatergic excitatory cortical neuron': GROUP_COLORS['Excitatory'],
    'GABAergic inhibitory cortical interneuron': GROUP_COLORS['Inhibitory'],
}
LABEL_OFFSETS = {
    'microglial cell': (-0.05, 0.03, 'right'),
    'astrocyte of the cerebral cortex': (0.04, -0.03, 'left'),
    'oligodendrocyte': (-0.04, 0.03, 'right'),
    'oligodendrocyte precursor cell': (0.04, -0.04, 'left'),
    'cerebral cortex endothelial cell': (0.04, 0.03, 'left'),
    'glutamatergic excitatory cortical neuron': (0.04, 0.03, 'left'),
    'GABAergic inhibitory cortical interneuron': (-0.04, -0.03, 'right'),
}


def casc_abs_int(xai_df, g):
    return float(xai_df.loc[INTERMEDIATE_MAP[g], 'abs_change'].mean())


def save_both(fig, stem, output_dir):
    for ext in ['png', 'pdf']:
        fig.savefig(f'{output_dir}/{stem}.{ext}', dpi=300, bbox_inches='tight', pad_inches=0.15)


def plot_individual_dumbbell(xai_df, elo_diff_int, output_dir):
    all_groups = list(INTERMEDIATE_MAP.keys())
    casc_vals = {g: casc_abs_int(xai_df, g) for g in all_groups}
    elo_vals = {g: float(elo_diff_int[g]) for g in all_groups}

    rows = []
    for major in GROUP_ORDER:
        members = [(g, casc_vals[g], elo_vals[g]) for g in all_groups if INT_MAJOR[g] == major]
        members.sort(key=lambda t: -t[1])
        rows.extend(members)

    groups = [r[0] for r in rows]
    casc_v = np.array([r[1] for r in rows])
    elo_v = np.array([r[2] for r in rows])
    labels = [SHORT_LABEL[g] for g in groups]
    colors = [GROUP_COLORS[INT_MAJOR[g]] for g in groups]
    n = len(rows)

    casc_pct = rankdata(casc_v) / n * 100
    elo_pct = rankdata(elo_v) / n * 100
    rho_i, p_i = spearmanr(casc_v, elo_v)

    fig1, ax1 = plt.subplots(figsize=(20, 8), dpi=300)

    group_bounds = {}
    for major in GROUP_ORDER:
        idxs = [i for i, g in enumerate(groups) if INT_MAJOR[g] == major]
        if idxs:
            group_bounds[major] = (min(idxs), max(idxs))
    for major, (lo, hi) in group_bounds.items():
        ax1.axvspan(lo - 0.5, hi + 0.5, color=GROUP_BG[major], zorder=0)

    for i, (cp, ep, col) in enumerate(zip(casc_pct, elo_pct, colors)):
        lo_y, hi_y = min(cp, ep), max(cp, ep)
        ax1.plot([i, i], [lo_y, hi_y], color=col, lw=1.6, alpha=0.5, zorder=1)
        ax1.scatter(i, ep, color=col, s=120, zorder=3, edgecolor='white', linewidth=0.7, marker='o')
        ax1.scatter(i, cp, color=col, s=120, zorder=3, edgecolor='white', linewidth=0.7, marker='D')

    prev = None
    for i, g in enumerate(groups):
        major = INT_MAJOR[g]
        if major != prev and i > 0:
            ax1.axvline(i - 0.5, color='#cccccc', lw=0.8, zorder=2)
        prev = major

    ax1.set_xlim(-0.5, n - 0.5)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(labels, fontsize=16, rotation=45, ha='right', rotation_mode='anchor')
    for tick, col in zip(ax1.get_xticklabels(), colors):
        tick.set_color(col)

    ax1.set_ylim(-5, 115)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=16)
    ax1.set_ylabel('Percentile rank (%)', fontsize=18, labelpad=6)
    ax1.grid(axis='y', alpha=0.2, linewidth=0.5, linestyle='--')

    for major, (lo, hi) in group_bounds.items():
        mid = (lo + hi) / 2
        ax1.text(mid, 112, major, ha='center', va='bottom', fontsize=16,
                 color=GROUP_COLORS[major], fontweight='bold', clip_on=False)

    legend_items = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#555555', markersize=11, label='|AD - Control| Elo'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#555555', markersize=10, label='CASCADE score'),
    ]
    ax1.legend(handles=legend_items, loc='lower left', fontsize=15, frameon=True, framealpha=0.85, edgecolor='#cccccc')

    plt.tight_layout(pad=1.0)
    save_both(fig1, 'elo_scatter_individual_absolute_change', output_dir)
    plt.close(fig1)
    print(f'Saved: dumbbell (n={n}, rho={rho_i:.3f}, p={p_i:.4f})')


def plot_major_scatter(elo_diff_maj, output_dir):
    maj_groups = list(CASC_ABS_MAJOR.keys())
    xm_raw = np.array([CASC_ABS_MAJOR[g] for g in maj_groups])
    ym_raw = np.array([float(elo_diff_maj[g]) for g in maj_groups])

    rho_m, p_m = spearmanr(xm_raw, ym_raw)
    p_str_m = f'{p_m:.4f}' if p_m >= 0.0001 else f'{p_m:.2e}'

    n_m = len(maj_groups)
    xm = pd.Series(xm_raw, index=maj_groups).rank() / n_m
    ym = pd.Series(ym_raw, index=maj_groups).rank() / n_m

    fig2, ax2 = plt.subplots(figsize=(6.5, 6.5), dpi=300)
    ax2.plot([0, 1], [0, 1], ls='--', color='#aaaaaa', lw=1.2, zorder=1)

    for g in maj_groups:
        col = MAJ_COLOR[g]
        ax2.scatter(xm[g], ym[g], s=140, color=col, edgecolor='white', linewidth=0.8, zorder=3)
        dx, dy, ha = LABEL_OFFSETS.get(g, (0.04, 0.02, 'left'))
        ax2.text(xm[g] + dx, ym[g] + dy, MAJ_LABEL[g], fontsize=14, ha=ha, va='center', color=col, fontweight='bold')

    ax2.set_xlim(-0.05, 1.25)
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_xlabel('CASCADE absolute change - percentile rank', fontsize=16, labelpad=6)
    ax2.set_ylabel('|AD - Control| Elo - percentile rank', fontsize=16, labelpad=6)
    ax2.set_title(f'Spearman rho = {rho_m:.3f},  p = {p_str_m}', fontsize=16, pad=10)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=15)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=15)
    ax2.grid(alpha=0.18, linewidth=0.5, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.set_aspect('equal')

    plt.tight_layout(pad=1.2)
    save_both(fig2, 'elo_scatter_major_absolute_change', output_dir)
    plt.close(fig2)
    print(f'Saved: major percentile scatter (n={n_m}, rho={rho_m:.3f}, p={p_m:.4f})')


def main(elo_dir, output_dir):
    xai_df = pd.DataFrame(CL_XAI)
    xai_df['abs_change'] = (xai_df['dementia'] - xai_df['normal']).abs()

    ad_int = pd.read_csv(f'{elo_dir}/results_intermediate_ad_elo_elo.csv').set_index('name')['elo']
    ctl_int = pd.read_csv(f'{elo_dir}/results_intermediate_control_elo_elo.csv').set_index('name')['elo']
    elo_diff_int = (ad_int - ctl_int).abs()

    ad_maj = pd.read_csv(f'{elo_dir}/results_major_ad_elo_elo.csv').set_index('name')['elo']
    ctl_maj = pd.read_csv(f'{elo_dir}/results_major_control_elo_elo.csv').set_index('name')['elo']
    elo_diff_maj = (ad_maj - ctl_maj).abs()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_individual_dumbbell(xai_df, elo_diff_int, output_dir)
    plot_major_scatter(elo_diff_maj, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--elo-dir", type=str, default=str(CASCADE_DATA_ROOT / "SEATTLE" / "gene_explainer_validation"),
                        help="Directory containing the elo_score.py output CSVs.")
    parser.add_argument("--output-dir", type=str, default=None, help="Defaults to --elo-dir.")
    args = parser.parse_args()
    main(args.elo_dir, args.output_dir or args.elo_dir)
