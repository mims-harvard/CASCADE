#!/usr/bin/env python3
"""
Headline Section 2.4 comparison figure: CASCADE-Explainer vs. an
attention-derived baseline vs. cell-type-specific DEG, scored by AUROC
against two ground-truth marker sources (in-house/experimental and
literature), for healthy-only SEATTLE cell types (Methods 2.4,
"CASCADE-Explainer recovers canonical cell-type gene programmes").

DEG rankings are built on the fly from the healthy-only grouped cell-type
marker CSVs (see analysis/alzheimers/healthy_celltype_markers.py), restricted
to the same gene universe as CASCADE for a fair comparison.

Usage:
    python -m analysis.alzheimers.create_combined_gt_viz \
        --data-dir $CASCADE_DATA_ROOT/SEATTLE/gene_explainer_validation \
        --cascade-ranks $CASCADE_DATA_ROOT/SEATTLE/median_gene_ranks_corrected.csv \
        --output-dir $CASCADE_DATA_ROOT/SEATTLE/gene_explainer_validation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cascade.explainer.config import CASCADE_DATA_ROOT
from analysis.alzheimers.gt_analysis import (
    CELL_TYPE_MAP, analyze_ground_truth_auroc, load_biomart_mapping, load_cell_type_mapping,
    load_ground_truth_from_excel, load_ground_truth_markers, load_ranking_file,
)

GT_SOURCES = [
    ('txt', 'ground_truth', 'Reference: in-house', '#4878CF'),
    ('paper', 'GT_paper', 'Reference: literature', '#6ACC65'),
]
METHOD_COLORS = {'CASCADE': '#333333', 'Attention': '#CC6677', 'DEG': '#44AA99'}
CELL_TYPES_ORDER = ['Astrocytes', 'Endothelial', 'IO', 'Microglia', 'OPC', 'Oligodendrocytes']
CT_DISPLAY = {'IO': 'Neurons'}  # rename at display time only

METHOD_BG = {'CASCADE': '#E8EEF7', 'Attention': '#FAE8EC', 'DEG': '#E4F4F1'}
METHOD_BG_EDGE = {'CASCADE': '#9BAFD0', 'Attention': '#D4889A', 'DEG': '#6BBFB3'}


def build_deg_ranking(marker_dir: Path, cascade_genes: dict[str, set] | None = None):
    """Load pre-grouped healthy-only DEG marker files (see
    healthy_celltype_markers.py) and build up/down/updown ranking DataFrames.
    All genes are ranked by log2FC (no p-value filter). If cascade_genes is
    given, each cell type's DEG pool is restricted to genes present in the
    CASCADE set for that cell type, so the pools are directly comparable.
    """
    frames = []
    for csv_path in sorted(marker_dir.glob('markers_healthy_only_*.csv')):
        if 'all_grouped_cell_types' in csv_path.name:
            continue
        df = pd.read_csv(csv_path, usecols=['grouped_cell_type', 'gene', 'log2FC'])
        frames.append(df.rename(columns={'grouped_cell_type': 'display_ct'}))

    combined = pd.concat(frames, ignore_index=True)
    combined['gene'] = combined['gene'].astype(str).str.strip()

    rows_up, rows_down, rows_updown = [], [], []
    for ct, grp in combined.groupby('display_ct'):
        all_genes = grp.copy()
        if cascade_genes is not None and ct in cascade_genes:
            all_genes = all_genes[all_genes['gene'].isin(cascade_genes[ct])].copy()

        up = all_genes.sort_values('log2FC', ascending=False).copy()
        up['median_rank'], up['n_rows'], up['cell_type'] = range(1, len(up) + 1), 1, ct
        rows_up.append(up[['cell_type', 'gene', 'median_rank', 'n_rows']])

        down = all_genes.sort_values('log2FC', ascending=True).copy()
        down['median_rank'], down['n_rows'], down['cell_type'] = range(1, len(down) + 1), 1, ct
        rows_down.append(down[['cell_type', 'gene', 'median_rank', 'n_rows']])

        updown = (all_genes.assign(abs_fc=all_genes['log2FC'].abs())
                  .sort_values('abs_fc', ascending=False).drop(columns='abs_fc').copy())
        updown['median_rank'], updown['n_rows'], updown['cell_type'] = range(1, len(updown) + 1), 1, ct
        rows_updown.append(updown[['cell_type', 'gene', 'median_rank', 'n_rows']])

    deg_up, deg_down, deg_updown = (pd.concat(r, ignore_index=True) for r in (rows_up, rows_down, rows_updown))
    for df in (deg_up, deg_down, deg_updown):
        df['gene'] = df['gene'].astype(str).str.strip()
    return deg_up, deg_down, deg_updown


def _gene_sets(df: pd.DataFrame) -> dict:
    return {ct: set(g['gene']) for ct, g in df.groupby('cell_type')}


def _filter_to_universe(df: pd.DataFrame, universe: dict) -> pd.DataFrame:
    parts = [df[df['cell_type'] == ct][df[df['cell_type'] == ct]['gene'].isin(genes)]
             for ct, genes in universe.items() if ct in df['cell_type'].values]
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[:0]


def p_to_stars(p):
    if pd.isna(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def main(data_dir, cascade_ranks_path, output_dir):
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 15, 'axes.titlesize': 16, 'axes.labelsize': 15,
        'xtick.labelsize': 15, 'ytick.labelsize': 16,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    deg_marker_dir = data_dir / 'healthy_grouped_cell_type_markers'

    rank_files = [
        (Path(cascade_ranks_path), 'CASCADE'),
        (data_dir / 'attention_baseline_ranks.csv', 'Attention'),
        (None, 'DEG'),
    ]

    print("Loading resources...")
    biomart_mapping = load_biomart_mapping(data_dir / 'biomart_cleaned.csv')
    cell_type_mapping = load_cell_type_mapping(data_dir / 'mapping-final-seattle.csv')
    paper_path = data_dir / 'GT_paper.xlsx'
    gt_folder = data_dir / 'ground_truth'

    print("Computing per-direction shared gene universes...")
    attn_raw = load_ranking_file(data_dir / 'attention_baseline_ranks.csv')
    if cell_type_mapping:
        attn_raw['cell_type'] = attn_raw['cell_type'].map(lambda v: cell_type_mapping.get(str(v).strip(), str(v).strip()))
    attn_aggregated = attn_raw.groupby(['cell_type', 'gene'], as_index=False)['median_rank'].mean().assign(n_rows=1)
    attn_genes = _gene_sets(attn_aggregated)

    casc_both_raw = load_ranking_file(Path(cascade_ranks_path))
    if cell_type_mapping:
        casc_both_raw['cell_type'] = casc_both_raw['cell_type'].map(lambda v: cell_type_mapping.get(str(v).strip(), str(v).strip()))
    cascade_gene_sets = _gene_sets(casc_both_raw)

    deg_up, _, _ = build_deg_ranking(deg_marker_dir, cascade_genes=cascade_gene_sets)

    # Universe: CASCADE bidirectional genes ∩ Attention genes ∩ DEG-up genes
    # (same universe for both GT sources since both use upregulation markers)
    universes = {'up': {ct: cascade_gene_sets.get(ct, set()) & attn_genes.get(ct, set()) & _gene_sets(deg_up).get(ct, set())
                        for ct in CELL_TYPES_ORDER}}
    for direction, u in universes.items():
        print(f"  Universe [{direction}]:", {ct: len(s) for ct, s in u.items()})

    cascade_both_up = _filter_to_universe(casc_both_raw, universes['up'])

    print("\nRunning analysis...")
    results: dict[str, dict[str, dict[str, dict]]] = {}
    src_universe = {'txt': 'up', 'paper': 'up'}

    for rank_path, method_label in rank_files:
        print(f"\n-- {method_label} --------------------")
        results[method_label] = {}

        for src_key, src_label, display_label, _ in GT_SOURCES:
            gt_markers = load_ground_truth_markers(gt_folder) if src_key == 'txt' else load_ground_truth_from_excel(paper_path)
            universe = universes[src_universe[src_key]]

            if method_label == 'DEG':
                eval_df = _filter_to_universe(deg_up, universe)
            elif method_label == 'CASCADE':
                eval_df = cascade_both_up
            else:
                eval_df = _filter_to_universe(attn_aggregated, universe)

            plot_df, rand_means, rand_ses, p_values = analyze_ground_truth_auroc(
                gt_markers=gt_markers, gt_source=src_label, biomart_mapping=biomart_mapping,
                median_df=eval_df, rng_seed=42, bootstrap_iters=1000,
            )

            ct_data: dict[str, dict] = {}
            for idx, (_, row) in enumerate(plot_df.iterrows()):
                ct_data[row['cell_type']] = {
                    'mean': row['auroc'], 'se': row.get('rand_se', np.nan),
                    'rand_mean': rand_means[idx] if idx < len(rand_means) else np.nan,
                    'rand_se': rand_ses[idx] if idx < len(rand_ses) else np.nan,
                    'p': p_values[idx] if idx < len(p_values) else np.nan,
                }
            results[method_label][display_label] = ct_data

    print("\nAll analyses done. Generating plot...")
    _plot_grouped_bars(results, rank_files, output_dir)


def _plot_grouped_bars(results, rank_files, output_dir):
    methods = [label for _, label in rank_files]
    gt_labels = [d for _, _, d, _ in GT_SOURCES]
    gt_colors = {d: c for _, _, d, c in GT_SOURCES}
    n_ct, n_method, n_gt = len(CELL_TYPES_ORDER), len(methods), len(gt_labels)

    bar_w, gt_pad, m_pad, ct_gap = 0.42, 0.18, 0.30, 5.2
    cluster_w = n_gt * bar_w + (n_gt - 1) * gt_pad
    method_offsets = np.array([(i - (n_method - 1) / 2) * (cluster_w + m_pad) for i in range(n_method)])
    gt_offsets = np.array([(j - (n_gt - 1) / 2) * (bar_w + gt_pad) for j in range(n_gt)])
    ct_centers = np.arange(n_ct) * ct_gap

    fig, ax = plt.subplots(figsize=(28, 9))

    bg_half = cluster_w / 2 + 0.10
    for ci in range(n_ct):
        for mi, method in enumerate(methods):
            cx = ct_centers[ci] + method_offsets[mi]
            ax.axvspan(cx - bg_half, cx + bg_half, color=METHOD_BG[method], alpha=1.0, zorder=0, lw=0)
            for xv in [cx - bg_half, cx + bg_half]:
                ax.axvline(xv, color=METHOD_BG_EDGE[method], lw=0.6, zorder=1)

    for ci in range(n_ct - 1):
        ax.axvline((ct_centers[ci] + ct_centers[ci + 1]) / 2, color='#999999', lw=1.2, ls='-', zorder=2)
    ax.axhline(0.5, color='#666666', lw=1.0, ls='--', zorder=3)

    null_band_color = '#444444'
    gt_legend_handles = []
    for gi, gt_lbl in enumerate(gt_labels):
        col = gt_colors[gt_lbl]
        for mi, method in enumerate(methods):
            for ci, ct in enumerate(CELL_TYPES_ORDER):
                xc = ct_centers[ci] + method_offsets[mi] + gt_offsets[gi]
                d = results[method].get(gt_lbl, {}).get(ct, {})
                auroc, rand_mean, rand_se, p = (d.get(k, np.nan) for k in ('mean', 'rand_mean', 'rand_se', 'p'))

                ax.bar(xc, max(auroc - 0.3, 0) if not np.isnan(auroc) else 0, bottom=0.3, width=bar_w,
                      color=col, alpha=0.88, edgecolor='white', linewidth=0.5, zorder=4)

                if not np.isnan(rand_mean) and not np.isnan(rand_se):
                    half = bar_w / 2
                    ax.fill_between([xc - half, xc + half], rand_mean - rand_se, rand_mean + rand_se,
                                    color=null_band_color, alpha=0.18, zorder=5, lw=0)
                    ax.plot([xc - half, xc + half], [rand_mean, rand_mean], color=null_band_color,
                           lw=1.4, zorder=6, solid_capstyle='butt')

                star = p_to_stars(p)
                if star and not np.isnan(auroc):
                    ax.text(xc, auroc + 0.018, star, ha='center', va='bottom', fontsize=9,
                           fontweight='bold', color='#111111', zorder=7)

        gt_legend_handles.append(mpatches.Patch(facecolor=col, alpha=0.88, edgecolor='white', linewidth=0.5, label=gt_lbl))

    method_legend_handles = [mpatches.Patch(facecolor=METHOD_BG[m], edgecolor=METHOD_BG_EDGE[m], linewidth=1.2, label=m)
                             for m in methods]
    null_handle = mlines.Line2D([], [], color=null_band_color, lw=1.4, label='Null mean +/- SE (permutation)')

    ax.set_xticks(ct_centers)
    ax.set_xticklabels([CT_DISPLAY.get(ct, ct) for ct in CELL_TYPES_ORDER], fontsize=20, fontweight='bold', color='#222222')
    ax.tick_params(axis='x', length=0, pad=8)
    ax.set_ylim(0.28, 1.02)
    ax.set_xlim(ct_centers[0] - ct_gap / 2, ct_centers[-1] + ct_gap / 2)
    ax.set_ylabel('AUROC', fontsize=20)
    ax.tick_params(axis='y', labelsize=17)
    ax.spines['bottom'].set_visible(False)

    all_handles = gt_legend_handles + method_legend_handles + [null_handle]
    ax.legend(handles=all_handles, loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=len(all_handles),
             fontsize=16, frameon=True, edgecolor='#cccccc', handlelength=1.8, handleheight=1.4, columnspacing=1.6)

    for ext in ('png', 'pdf', 'svg'):
        kwargs = {'dpi': 200} if ext != 'pdf' else {}
        fig.savefig(output_dir / f'viz_option2_grouped.{ext}', bbox_inches='tight', pad_inches=0.2, **kwargs)
    plt.close(fig)
    print(f"\nSaved: {output_dir / 'viz_option2_grouped.png'} (+ .pdf, .svg)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=CASCADE_DATA_ROOT / "SEATTLE" / "gene_explainer_validation",
                        help="Directory with ground_truth/, GT_paper.xlsx, biomart_cleaned.csv, attention_baseline_ranks.csv, "
                             "mapping-final-seattle.csv, healthy_grouped_cell_type_markers/.")
    parser.add_argument("--cascade-ranks", type=Path, required=True,
                        help="CASCADE gene-explainer median-rank CSV (bidirectional, e.g. median_gene_ranks_corrected.csv).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to --data-dir.")
    args = parser.parse_args()
    main(args.data_dir, args.cascade_ranks, args.output_dir or args.data_dir)
