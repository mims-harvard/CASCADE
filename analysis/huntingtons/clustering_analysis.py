#!/usr/bin/env python3
"""
Primary donor clustering from CASCADE-Explainer cell-type attention profiles
(Figure 5f; Methods 2.7). For each CAG-repeat-length model (benign CAG_1,
pathogenic CAG_2), selects the best k (by silhouette score, k in 2..8) for
KMeans clustering on donor x cell-type attention-importance profiles, then
produces:
  1. A clinical bar chart comparing age/onset/VS-grade across clusters.
  2. Per-model heatmaps of mean cell-type importance per cluster, with a
     delta panel and top-40%-by-|delta| cell types highlighted.

Usage:
    python -m analysis.huntingtons.clustering_analysis \
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
import seaborn as sns
from matplotlib.transforms import blended_transform_factory
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler

from analysis.huntingtons.hd_attention_utils import (
    cluster_best_k, create_donor_df, load_donor_info, normalize_per_donor, prepare_features,
)

warnings.filterwarnings('ignore')

CLIN_COLS = {'age': 'Age', 'onset_motor': 'Motor onset', 'onset_cognitive': 'Cognitive onset'}
CLUSTER_NAMES = {0: 'Cluster A', 1: 'Cluster B'}
CLIN_COLORS = ['#4477AA', '#D55E00', '#009E73', '#AA3377']
BORDER_COLOR = '#E65100'


def sig_label(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def bh_correct(pvals):
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    sorted_idx = np.argsort(pvals)
    adjusted = np.zeros(n)
    for rank, idx in enumerate(sorted_idx):
        adjusted[idx] = min(1.0, pvals[idx] * n / (rank + 1))
    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(adjusted[sorted_idx[i]], adjusted[sorted_idx[i + 1]])
    return adjusted


def cluster_and_analyse(raw_data, tag, donor_info, seed=42):
    df = normalize_per_donor(create_donor_df(raw_data, tag).merge(donor_info, on='donor_id', how='left'))
    feats = prepare_features(df)
    feat_cols = [c for c in feats.columns if c.endswith('_importance')]
    cell_types = [c.replace('_importance', '') for c in feat_cols]
    x_scaled = StandardScaler().fit_transform(np.nan_to_num(feats[feat_cols].values, nan=0.0))

    labels, best_k, best_sil = cluster_best_k(x_scaled, seed=seed)
    print(f'  {tag}: best k={best_k} (silhouette={best_sil:.3f})')
    feats = feats.copy()
    feats['cluster'] = labels
    return feats, best_k, cell_types


def main(cag1_npz, cag2_npz, donor_info_path, output_dir, seed=42):
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14, 'axes.titlesize': 15, 'axes.labelsize': 14,
        'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.major.size': 4, 'ytick.major.size': 4,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })

    print("CLUSTERING ANALYSIS: CELL TYPE IMPORTANCE -> CLINICAL DIFFERENCES")
    donor_info = load_donor_info(donor_info_path)
    data_cag1, data_cag2 = np.load(cag1_npz, allow_pickle=True), np.load(cag2_npz, allow_pickle=True)

    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    paper_ready_dir = plots_dir / 'paper_ready'
    paper_ready_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing data...")
    results = {}
    for raw_data, tag in [(data_cag1, 'CAG_1'), (data_cag2, 'CAG_2')]:
        results[tag] = cluster_and_analyse(raw_data, tag, donor_info, seed=seed)

    tags = ['CAG_1', 'CAG_2']
    n_cag1 = results['CAG_1'][1]

    # ── Build heatmap data per tag ──────────────────────────────────────────
    hm_data, col_keys, col_display = {}, [], []
    for tag in tags:
        feats, best_k, _ = results[tag]
        feat_cols = [c for c in feats.columns if c.endswith('_importance')]
        cts = [c.replace('_importance', '') for c in feat_cols]
        for i in range(best_k):
            n_i = (feats['cluster'] == i).sum()
            key = f'{tag}_C{i}'
            col_keys.append(key)
            col_display.append(f'{CLUSTER_NAMES.get(i, f"Cluster {i}")}\n(n={n_i})')
            sub = feats[feats['cluster'] == i]
            hm_data[key] = {ct: sub[f'{ct}_importance'].mean() for ct in cts}

    all_cts = sorted({ct for d in hm_data.values() for ct in d})
    hm_df_full = pd.DataFrame({k: [hm_data[k].get(ct, np.nan) for ct in all_cts] for k in col_keys}, index=all_cts)

    hm_per_tag, delta_per_tag = {}, {}
    for tag in tags:
        keys = [k for k in col_keys if k.startswith(tag)]
        disp = [col_display[col_keys.index(k)] for k in keys]
        hm = hm_df_full[keys].copy()
        hm.columns = disp
        hm_per_tag[tag] = hm
        delta_per_tag[tag] = hm.iloc[:, 1] - hm.iloc[:, 0] if hm.shape[1] > 1 else pd.Series(0, index=hm.index)

    # ── BH FDR-corrected MWU significance (cell type x CAG group) ──────────
    ct_tag_list, raw_pvals = [], []
    for tag in tags:
        feats, _, _ = results[tag]
        for ct in all_cts:
            col = f'{ct}_importance'
            ct_tag_list.append((ct, tag))
            if col not in feats.columns:
                raw_pvals.append(1.0)
                continue
            c0 = feats[feats['cluster'] == 0][col].dropna().values
            c1 = feats[feats['cluster'] == 1][col].dropna().values if 1 in feats['cluster'].values else np.array([])
            if len(c0) > 3 and len(c1) > 3:
                _, p = mannwhitneyu(c0, c1, alternative='two-sided')
            else:
                p = 1.0
            raw_pvals.append(p)
    adj_pvals = bh_correct(raw_pvals)
    ct_sig = {pair: sig_label(p) for pair, p in zip(ct_tag_list, adj_pvals)}

    print("\n  Cell-type significance (BH FDR-corrected MWU):")
    for (ct, tag), p_raw, p_adj in zip(ct_tag_list, raw_pvals, adj_pvals):
        star = ct_sig[(ct, tag)]
        if star or p_raw < 0.1:
            print(f'    {tag} {ct}: raw p={p_raw:.4f}, adj p={p_adj:.4f}  {star}')

    annot_per_tag = {}
    for tag in tags:
        hm = hm_per_tag[tag]
        mat = np.full(hm.shape, '', dtype=object)
        for ci in range(hm.shape[1]):
            for ri, ct in enumerate(all_cts):
                mat[ri, ci] = f'{hm.iloc[ri, ci]:.2f}'
        annot_per_tag[tag] = mat

    # ── Clinical pairwise tests (MWU) ───────────────────────────────────────
    all_clin_cols = list(CLIN_COLS.keys()) + ['vs_grade']
    clin_pairwise_p = {}
    for tag in tags:
        feats, _, _ = results[tag]
        for col in all_clin_cols:
            c0 = feats[feats['cluster'] == 0][col].dropna().values
            c1 = feats[feats['cluster'] == 1][col].dropna().values if 1 in feats['cluster'].values else np.array([])
            if len(c0) > 3 and len(c1) > 3:
                _, p = mannwhitneyu(c0, c1, alternative='two-sided')
            else:
                p = 1.0
            clin_pairwise_p[(tag, col)] = p

    clin_stats = {}
    for key in col_keys:
        tag, ci_str = key.rsplit('_C', 1)
        sub = results[tag][0]
        sub = sub[sub['cluster'] == int(ci_str)]
        for clin_col in all_clin_cols:
            vals = sub[clin_col].dropna()
            clin_stats[(key, clin_col)] = (vals.mean(), vals.std())

    _plot_clinical_bar_chart(col_keys, tags, n_cag1, results, clin_stats, clin_pairwise_p, paper_ready_dir, plots_dir)
    _draw_heatmap_panel('CAG_1', hm_per_tag, delta_per_tag, annot_per_tag, ct_sig, all_cts,
                        plots_dir / 'clustering_analysis_celltypes_benign.png')
    _draw_heatmap_panel('CAG_2', hm_per_tag, delta_per_tag, annot_per_tag, ct_sig, all_cts,
                        plots_dir / 'clustering_analysis_celltypes_pathogenic.png')

    print('\nDone.')


def _plot_clinical_bar_chart(col_keys, tags, n_cag1, results, clin_stats, clin_pairwise_p, paper_ready_dir, plots_dir):
    clin_list = list(CLIN_COLS.items())
    n_clin = len(clin_list)
    n_bars = n_clin + 1
    bar_w, group_sep, cag_gap = 0.24, 1.55, 1.3

    x_positions = []
    for gi in range(len(col_keys)):
        cag_idx = 0 if gi < n_cag1 else 1
        cl_idx = gi if cag_idx == 0 else gi - n_cag1
        x_positions.append(cag_idx * (n_cag1 * group_sep + cag_gap) + cl_idx * group_sep)

    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    ax1_vs = ax1.twinx()
    bar_tops, bar_tops_vs = {}, {}

    for gi, (key, xc) in enumerate(zip(col_keys, x_positions)):
        for bi, (clin_col, _) in enumerate(clin_list):
            offset = (bi - (n_bars - 1) / 2) * bar_w
            mean_val, std_val = clin_stats[(key, clin_col)]
            x_bar = xc + offset
            ax1.bar(x_bar, mean_val, bar_w * 0.88, color=CLIN_COLORS[bi], alpha=0.82,
                    edgecolor='white', linewidth=0.3, zorder=3)
            ax1.errorbar(x_bar, mean_val, yerr=std_val, fmt='none', color='#333333',
                        capsize=3.5, capthick=1.2, elinewidth=1.0, zorder=4)
            bar_tops[(key, clin_col)] = (x_bar, mean_val + std_val)

        bi_vs = n_clin
        offset_vs = (bi_vs - (n_bars - 1) / 2) * bar_w
        mean_vs, std_vs = clin_stats[(key, 'vs_grade')]
        x_bar_vs = xc + offset_vs
        ax1_vs.bar(x_bar_vs, mean_vs, bar_w * 0.88, color=CLIN_COLORS[3], alpha=0.82,
                   edgecolor='white', linewidth=0.3, zorder=3)
        ax1_vs.errorbar(x_bar_vs, mean_vs, yerr=std_vs, fmt='none', color='#333333',
                        capsize=3.5, capthick=1.2, elinewidth=1.0, zorder=4)
        bar_tops_vs[(key, 'vs_grade')] = (x_bar_vs, mean_vs + std_vs)

    tick_h, v_step = 1.5, 7.0
    max_bracket_y = {tag: 0 for tag in tags}
    for tag in tags:
        c0_key, c1_key = f'{tag}_C0', f'{tag}_C1'
        if c0_key not in col_keys or c1_key not in col_keys:
            continue
        sig_vars = [(bi, col) for bi, (col, _) in enumerate(clin_list) if sig_label(clin_pairwise_p.get((tag, col), 1.0))]
        sig_vars.sort(key=lambda x: 0 if x[1] != 'age' else 1)
        for level, (bi, clin_col) in enumerate(sig_vars):
            star = sig_label(clin_pairwise_p[(tag, clin_col)])
            x0, top0 = bar_tops[(c0_key, clin_col)]
            x1, top1 = bar_tops[(c1_key, clin_col)]
            y_b = max(top0, top1) + 1.5 + level * v_step
            ax1.plot([x0, x0, x1, x1], [y_b, y_b + tick_h, y_b + tick_h, y_b],
                     color=CLIN_COLORS[bi], lw=1.6, zorder=5, solid_capstyle='round')
            ax1.text((x0 + x1) / 2, y_b + tick_h + 0.2, star, ha='center', va='bottom',
                     fontsize=13, fontweight='bold', color=CLIN_COLORS[bi])
            max_bracket_y[tag] = max(max_bracket_y[tag], y_b + tick_h + 2)

    for tag in tags:
        c0_key, c1_key = f'{tag}_C0', f'{tag}_C1'
        if c0_key not in col_keys or c1_key not in col_keys:
            continue
        star = sig_label(clin_pairwise_p.get((tag, 'vs_grade'), 1.0))
        if not star:
            continue
        x0 = bar_tops_vs[(c0_key, 'vs_grade')][0]
        x1 = bar_tops_vs[(c1_key, 'vs_grade')][0]
        y_b = max_bracket_y[tag] + v_step
        ax1.plot([x0, x0, x1, x1], [y_b, y_b + tick_h, y_b + tick_h, y_b],
                 color=CLIN_COLORS[3], lw=1.6, zorder=5, solid_capstyle='round')
        ax1.text((x0 + x1) / 2, y_b + tick_h + 0.2, star, ha='center', va='bottom',
                 fontsize=13, fontweight='bold', color=CLIN_COLORS[3])
        max_bracket_y[tag] = max(max_bracket_y[tag], y_b + tick_h + 2)

    xtick_labels = []
    for key in col_keys:
        t, ci_str = key.rsplit('_C', 1)
        n_k = (results[t][0]['cluster'] == int(ci_str)).sum()
        name = CLUSTER_NAMES.get(int(ci_str), f'Cluster {ci_str}')
        xtick_labels.append(f'{name}\n(n={n_k})')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(xtick_labels, fontsize=13)

    trans = blended_transform_factory(ax1.transData, ax1.transAxes)
    for xi, lbl in [(np.mean(x_positions[:n_cag1]), 'Benign CAG'), (np.mean(x_positions[n_cag1:]), 'Pathogenic CAG')]:
        ax1.text(xi, 1.01, lbl, ha='center', va='bottom', fontsize=15, fontweight='bold', transform=trans)

    leg = [mpatches.Patch(color=CLIN_COLORS[i], alpha=0.82, label=lbl) for i, (_, lbl) in enumerate(clin_list)]
    leg += [mpatches.Patch(color=CLIN_COLORS[3], alpha=0.82, label='VS Grade')]
    from matplotlib.lines import Line2D
    sig_handle = Line2D([], [], color='none', label='  * p<0.05   ** p<0.01   *** p<0.001 (MWU)')
    ax1.legend(handles=leg + [sig_handle], fontsize=12, frameon=True, framealpha=0.9, edgecolor='#cccccc',
               loc='lower center', bbox_to_anchor=(0.5, -0.20), ncol=5,
               handlelength=1.2, handletextpad=0.5, columnspacing=0.8)

    ax1.set_ylabel('Mean value (years)', fontsize=14, labelpad=5)
    ax1_vs.set_ylabel('VS Grade', fontsize=14, labelpad=8, color=CLIN_COLORS[3])
    ax1_vs.tick_params(axis='y', labelcolor=CLIN_COLORS[3])
    ax1_vs.set_ylim(bottom=0)
    ax1.set_ylim(bottom=0, top=max(max_bracket_y.values()) + v_step)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.5, linestyle='--')
    ax1.set_axisbelow(True)

    plt.tight_layout()
    for out1 in (plots_dir / 'clustering_analysis_clinical.png', plots_dir / 'clustering_analysis_clinical.pdf',
                 paper_ready_dir / 'clustering_analysis_clinical.png', paper_ready_dir / 'clustering_analysis_clinical.pdf'):
        plt.savefig(out1, dpi=300, bbox_inches='tight', pad_inches=0.2)
        print(f'Saved: {out1}')
    plt.close(fig1)


def _draw_heatmap_panel(tag, hm_per_tag, delta_per_tag, annot_per_tag, ct_sig, all_cts, out_path):
    hm, delta, annot = hm_per_tag[tag], delta_per_tag[tag], annot_per_tag[tag]
    n_ct = len(all_cts)
    n_highlight = max(2, round(n_ct * 0.40))
    top_cts = set(delta.abs().nlargest(n_highlight).index)

    delta_col_name = 'B − A'
    delta_annot = np.array(
        [f'{delta.iloc[ri]:.2f}{ct_sig.get((ct, tag), "")}' for ri, ct in enumerate(all_cts)]
    ).reshape(-1, 1)
    delta_df_single = pd.DataFrame({delta_col_name: delta.values}, index=all_cts)
    delta_abs_max = float(delta.abs().max()) * 1.1 if delta.abs().max() > 0 else 1.0

    cell_sz = 0.72
    fig_w = 2 * cell_sz + 0.75 * cell_sz + 2.8
    fig_h = n_ct * cell_sz + 2.2
    fig, (ax_main, ax_delta) = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=300,
                                             gridspec_kw={'width_ratios': [2, 0.75], 'wspace': 0.55})

    hm = hm.copy()
    hm.columns = [c.split('\n')[0] for c in hm.columns]

    sns.heatmap(hm, ax=ax_main, cmap='Blues', vmin=0, linewidths=0.4, linecolor='#e0e0e0',
                cbar_kws={'label': 'Mean importance', 'shrink': 0.78, 'aspect': 18, 'pad': 0.02},
                annot=annot, fmt='', annot_kws={'size': 11}, xticklabels=True)
    ax_main.collections[0].colorbar.ax.tick_params(labelsize=10)
    ax_main.collections[0].colorbar.set_label('Mean importance', fontsize=11)
    ax_main.set_xticklabels(hm.columns, rotation=0, fontsize=12)
    ax_main.set_yticklabels(all_cts, rotation=0, fontsize=12)
    ax_main.set_xlabel('')
    ax_main.set_ylabel('Cell type', labelpad=6, fontsize=12)
    ax_main.tick_params(bottom=False)

    sns.heatmap(delta_df_single, ax=ax_delta, cmap='BrBG', vmin=-delta_abs_max, vmax=delta_abs_max, center=0,
                linewidths=0.4, linecolor='#e0e0e0',
                cbar_kws={'label': 'Δ (B−A)', 'shrink': 0.78, 'aspect': 18, 'pad': 0.14},
                annot=delta_annot, fmt='', annot_kws={'size': 8.5}, xticklabels=True)
    ax_delta.collections[0].colorbar.ax.tick_params(labelsize=10)
    ax_delta.collections[0].colorbar.set_label('Δ (B−A)', fontsize=11)
    ax_delta.set_xticklabels([delta_col_name], rotation=0, fontsize=12)
    ax_delta.set_yticks([])
    ax_delta.set_ylabel('')
    ax_delta.set_xlabel('')
    ax_delta.tick_params(bottom=False)

    n_main_cols = hm.shape[1]
    for ri, ct in enumerate(all_cts):
        if ct not in top_cts:
            continue
        ax_main.add_patch(mpatches.FancyBboxPatch((0, ri), n_main_cols, 1, boxstyle="square,pad=0",
                                                    fill=False, edgecolor=BORDER_COLOR, linewidth=2.4, zorder=6))
        ax_delta.add_patch(mpatches.FancyBboxPatch((0, ri), 1, 1, boxstyle="square,pad=0",
                                                     fill=False, edgecolor=BORDER_COLOR, linewidth=2.4, zorder=6))

    fig.text(0.5, -0.02, f'BH FDR: * p<0.05, ** p<0.01, *** p<0.001 (MWU)     '
             f'□ orange border = top {n_highlight} cell types by |Δ|',
             ha='center', fontsize=8.5, color='#555555', style='italic')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cag1-npz", type=str, required=True)
    parser.add_argument("--cag2-npz", type=str, required=True)
    parser.add_argument("--donor-info", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.cag1_npz, args.cag2_npz, args.donor_info, args.output_dir, seed=args.seed)
