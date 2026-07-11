#!/usr/bin/env python3
"""
Donor stratification from CASCADE-Explainer cell-type attention profiles
(Figure 5f-g; Methods 2.7, "CASCADE enables molecular and clinical
stratification of Huntington's disease patients").

Re-runs k=2 KMeans on donor-specific, per-cell-type max attention importance
(one clustering per CAG-repeat-length model: benign CAG_1, pathogenic CAG_2),
labels each model's two clusters as 'High-VS'/'Low-VS' by mean Vonsattel (VS)
grade, then cross-tabulates donor cluster membership between the two models
to test whether benign-CAG clustering is confounded by pathogenic-CAG effects.

Expects donor-level attention-weight caches (one .npz per CAG model) with
keys: attention_weights (n_layers, n_donors, n_query, n_cells), patient_y
(donor label, e.g. VS grade), patient_ids, patient_cell_types — produced by
the attention-extraction step of cascade.explainer.attention_analysis.

Usage:
    python -m analysis.huntingtons.cluster_overlap_analysis \
        --cag1-npz HD_models_scale/CAG_1_1107.npz \
        --cag2-npz HD_models_scale/CAG_2_1107.npz \
        --donor-info donors_hh_info.csv --output-dir .
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
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.preprocessing import StandardScaler

from analysis.huntingtons.hd_attention_utils import (
    CLINICAL_COLUMNS, cluster_k2, create_donor_df, load_donor_info, normalize_per_donor, prepare_features,
)

CAT_COLORS = {
    'Both-High': '#AA3377',
    'Both-Low': '#4477AA',
    'HighBenign-LowPath': '#EE7733',
    'LowBenign-HighPath': '#009988',
}
CAT_LABELS = {
    'Both-High': 'High-VS in both\n(concordant)',
    'Both-Low': 'Low-VS in both\n(concordant)',
    'HighBenign-LowPath': 'High-VS benign only\n(discordant)',
    'LowBenign-HighPath': 'High-VS pathogenic only\n(discordant)',
}


def cluster_and_orient_by_vs_grade(raw_data, tag, donor_info, seed=42):
    """Cluster a donor x cell-type feature matrix into k=2 groups, then
    orient the (arbitrary) cluster labels as 'High-VS'/'Low-VS' by mean
    Vonsattel grade so the two CAG models' clusters are comparable."""
    df = create_donor_df(raw_data, tag)
    df = df.merge(donor_info[CLINICAL_COLUMNS], on='donor_id', how='left')
    df = normalize_per_donor(df)
    feats = prepare_features(df)
    feat_cols = [c for c in feats.columns if c.endswith('_importance')]
    x = np.nan_to_num(feats[feat_cols].values, nan=0.0)
    x_scaled = StandardScaler().fit_transform(x)
    feats = feats.copy()
    feats['cluster_raw'] = cluster_k2(x_scaled, seed=seed)

    mean_vs = feats.groupby('cluster_raw')['vs_grade'].mean()
    high_vs_raw = mean_vs.idxmax()
    feats['cluster_vs'] = feats['cluster_raw'].map({high_vs_raw: 'High-VS', 1 - high_vs_raw: 'Low-VS'})
    return feats[['donor_id', 'cluster_raw', 'cluster_vs', 'age', 'vs_grade',
                  'onset_motor', 'onset_cognitive', 'disease_status']]


def overlap_category(row):
    benign, pathogenic = row['cag1_vs'], row['cag2_vs']
    if benign == 'High-VS' and pathogenic == 'High-VS':
        return 'Both-High'
    if benign == 'Low-VS' and pathogenic == 'Low-VS':
        return 'Both-Low'
    if benign == 'High-VS' and pathogenic == 'Low-VS':
        return 'HighBenign-LowPath'
    return 'LowBenign-HighPath'


def plot_overlap_summary(merged, ct, oddsratio, pval_fisher, output_dir):
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=300)
    plt.subplots_adjust(wspace=0.42)

    ax = axes[0]
    im = ax.imshow(ct.values, cmap='Blues', aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(ct.columns.tolist(), fontsize=11)
    ax.set_yticklabels(ct.index.tolist(), fontsize=11)
    ax.set_xlabel('Pathogenic CAG cluster', fontsize=11)
    ax.set_ylabel('Benign CAG cluster', fontsize=11)
    ax.set_title('Donor cluster overlap\n(cross-tabulation)', fontsize=12, fontweight='bold')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(ct.values[i, j]), ha='center', va='center', fontsize=16, fontweight='bold',
                    color='white' if ct.values[i, j] > ct.values.max() * 0.5 else '#333333')
    fig.colorbar(im, ax=ax, shrink=0.75, label='N donors')
    ax.text(0.5, -0.18, f"Fisher's exact: OR={oddsratio:.2f}, p={pval_fisher:.4f}",
            transform=ax.transAxes, ha='center', fontsize=9, color='#555555', style='italic')

    cats = ['Both-High', 'HighBenign-LowPath', 'LowBenign-HighPath', 'Both-Low']
    for panel_idx, var in enumerate(['vs_grade', 'age']):
        ax = axes[panel_idx + 1]
        for xi, cat in enumerate(cats):
            vals = merged[merged['overlap'] == cat][var].dropna().values
            if len(vals) == 0:
                continue
            ax.bar(xi, np.mean(vals), color=CAT_COLORS[cat], alpha=0.82, edgecolor='white', width=0.65)
            ax.errorbar(xi, np.mean(vals), yerr=np.std(vals), fmt='none', color='#333333', capsize=4, elinewidth=1.2)
            rng = np.random.default_rng(xi + 10 * (panel_idx + 1))
            jitter = rng.uniform(-0.18, 0.18, len(vals))
            ax.scatter(xi + jitter, vals, color=CAT_COLORS[cat], s=22, alpha=0.7, zorder=5, edgecolors='none')
            if var == 'vs_grade':
                ax.text(xi, -0.25, f'n={len(vals)}', ha='center', fontsize=9, color='#555555')
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([CAT_LABELS[c] for c in cats], fontsize=8.5)
        ax.set_ylabel('VS grade' if var == 'vs_grade' else 'Age (years)', fontsize=11)
        ax.set_title(('VS grade' if var == 'vs_grade' else 'Age') + ' by concordance\ncategory',
                     fontsize=12, fontweight='bold')
        if var == 'vs_grade':
            ax.set_ylim(bottom=-0.4)

    handles = [mpatches.Patch(color=CAT_COLORS[c], alpha=0.82, label=CAT_LABELS[c]) for c in cats]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.12))

    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(plots_dir / f'cluster_overlap_analysis.{ext}', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


def main(cag1_npz, cag2_npz, donor_info_path, output_dir, seed=42):
    donor_info = load_donor_info(donor_info_path)
    data_cag1 = np.load(cag1_npz, allow_pickle=True)
    data_cag2 = np.load(cag2_npz, allow_pickle=True)

    print("Running k=2 clustering per CAG model...")
    res1 = cluster_and_orient_by_vs_grade(data_cag1, 'CAG_1', donor_info, seed=seed).rename(
        columns={'cluster_raw': 'cag1_raw', 'cluster_vs': 'cag1_vs'})
    res2 = cluster_and_orient_by_vs_grade(data_cag2, 'CAG_2', donor_info, seed=seed).rename(
        columns={'cluster_raw': 'cag2_raw', 'cluster_vs': 'cag2_vs'})

    merged = res1.merge(res2[['donor_id', 'cag2_raw', 'cag2_vs']], on='donor_id', how='inner')
    merged = merged[merged['disease_status'] == 'Case'].copy()

    print(f"\nTotal donors with both CAG clusterings: {len(merged)}")
    print(f"Benign CAG:     High-VS={(merged['cag1_vs'] == 'High-VS').sum()},  "
          f"Low-VS={(merged['cag1_vs'] == 'Low-VS').sum()}")
    print(f"Pathogenic CAG: High-VS={(merged['cag2_vs'] == 'High-VS').sum()},  "
          f"Low-VS={(merged['cag2_vs'] == 'Low-VS').sum()}")

    ct = pd.crosstab(merged['cag1_vs'], merged['cag2_vs'],
                      rownames=['Benign CAG cluster'], colnames=['Pathogenic CAG cluster'])
    print("\nCross-tabulation (donor counts):")
    print(ct)

    oddsratio, pval_fisher = fisher_exact(ct.values)
    print(f"\nFisher's exact test: OR={oddsratio:.2f}, p={pval_fisher:.4f}")

    merged['overlap'] = merged.apply(overlap_category, axis=1)
    print("\nOverlap category counts:")
    print(merged['overlap'].value_counts())

    merged['concordant'] = merged['cag1_vs'] == merged['cag2_vs']
    print("\n--- Clinical comparison: concordant vs discordant donors ---")
    for var in ['age', 'vs_grade', 'onset_motor', 'onset_cognitive']:
        conc = merged[merged['concordant']][var].dropna()
        disc = merged[~merged['concordant']][var].dropna()
        if len(conc) > 1 and len(disc) > 1:
            _, p = mannwhitneyu(conc, disc, alternative='two-sided')
            print(f"  {var}: concordant={conc.mean():.1f}+-{conc.std():.1f} (n={len(conc)}), "
                  f"discordant={disc.mean():.1f}+-{disc.std():.1f} (n={len(disc)}), MWU p={p:.4f}")

    print("\n--- Clinical stats by overlap category ---")
    for cat, grp in merged.groupby('overlap'):
        vs, age, motor = grp['vs_grade'].dropna(), grp['age'].dropna(), grp['onset_motor'].dropna()
        print(f"\n  {cat} (n={len(grp)}):")
        print(f"    VS grade:    {vs.mean():.2f} +- {vs.std():.2f}  (n={len(vs)})")
        print(f"    Age:         {age.mean():.1f} +- {age.std():.1f}")
        print(f"    Motor onset: {motor.mean():.1f} +- {motor.std():.1f}  (n={len(motor)})")

    print("\n--- Donor IDs per overlap category ---")
    for cat in ['Both-High', 'Both-Low', 'HighBenign-LowPath', 'LowBenign-HighPath']:
        sub = merged[merged['overlap'] == cat]
        print(f"\n  {cat} (n={len(sub)}): donors {sorted(sub['donor_id'].tolist())}")

    print("\n--- MWU p-values between key category pairs ---")
    for cat_a, cat_b in itertools.combinations(
            ['Both-High', 'HighBenign-LowPath', 'LowBenign-HighPath'], 2):
        a = merged[merged['overlap'] == cat_a]['vs_grade'].dropna()
        b = merged[merged['overlap'] == cat_b]['vs_grade'].dropna()
        if len(a) > 1 and len(b) > 1:
            _, p = mannwhitneyu(a, b, alternative='two-sided')
            print(f"  {cat_a} vs {cat_b} [vs_grade]: {a.mean():.2f} vs {b.mean():.2f}, p={p:.4f}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / 'cluster_overlap_donors.csv'
    merged.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    plot_overlap_summary(merged, ct, oddsratio, pval_fisher, output_dir)
    print(f"Saved: {output_dir}/plots/cluster_overlap_analysis.png / .pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cag1-npz", type=str, required=True, help="Benign CAG (CAG_1) attention-weight cache")
    parser.add_argument("--cag2-npz", type=str, required=True, help="Pathogenic CAG (CAG_2) attention-weight cache")
    parser.add_argument("--donor-info", type=str, required=True, help="donors_hh_info.csv-style clinical metadata")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.cag1_npz, args.cag2_npz, args.donor_info, args.output_dir, seed=args.seed)
