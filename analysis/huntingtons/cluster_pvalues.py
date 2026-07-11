#!/usr/bin/env python3
"""
Quick console report of cluster-vs-clinical-variable Mann-Whitney U p-values
for the k=2 attention-profile clusters (supplementary sanity check for
Figure 5f; see clustering_analysis.py for the full figure).

Usage:
    python -m analysis.huntingtons.cluster_pvalues \
        --cag1-npz HD_models_scale/CAG_1_1107.npz \
        --cag2-npz HD_models_scale/CAG_2_1107.npz \
        --donor-info donors_hh_info.csv
"""
import argparse

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler

from analysis.huntingtons.hd_attention_utils import cluster_k2, create_donor_df, load_donor_info, normalize_per_donor

VARS = [('age', 'Age'), ('vs_grade', 'VS Grade'), ('onset_motor', 'Motor Onset'),
        ('onset_cognitive', 'Cog Onset'), ('cag1', 'Benign CAG'), ('cag2', 'Pathogenic CAG')]


def run(data_dict, seed=42):
    df = normalize_per_donor(create_donor_df(data_dict))
    pivot = df.pivot_table(index='donor_id', columns='cell_type', values='importance', aggfunc='mean')
    x_scaled = StandardScaler().fit_transform(np.nan_to_num(pivot.values, nan=0.0))
    labels = cluster_k2(x_scaled, seed=seed)
    return dict(zip(pivot.index, labels)), df.groupby('donor_id')['y_value'].first()


def main(cag1_npz, cag2_npz, donor_info_path, seed=42):
    donor_info = load_donor_info(donor_info_path)
    data_cag1, data_cag2 = np.load(cag1_npz, allow_pickle=True), np.load(cag2_npz, allow_pickle=True)

    cl1, cag1_vals = run(data_cag1, seed=seed)
    cl2, cag2_vals = run(data_cag2, seed=seed)
    meta = donor_info.set_index('donor_id')[['age', 'vs_grade', 'onset_motor', 'onset_cognitive']]

    for tag, clusters in [('BENIGN CAG', cl1), ('PATHOGENIC CAG', cl2)]:
        tbl = meta.copy()
        tbl['cag1'], tbl['cag2'] = cag1_vals, cag2_vals
        tbl['cl'] = tbl.index.to_series().map(clusters)
        tbl = tbl[tbl['cl'].notna()]
        print(tag)
        for col, label in VARS:
            c0 = tbl[tbl['cl'] == 0][col].dropna().values
            c1 = tbl[tbl['cl'] == 1][col].dropna().values
            if len(c0) > 1 and len(c1) > 1:
                _, p = mannwhitneyu(c0, c1, alternative='two-sided')
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                print(f'  {label:<20s} C0={c0.mean():.2f}(n={len(c0)})  C1={c1.mean():.2f}(n={len(c1)})  p={p:.4f}  {sig}')
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cag1-npz", type=str, required=True)
    parser.add_argument("--cag2-npz", type=str, required=True)
    parser.add_argument("--donor-info", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.cag1_npz, args.cag2_npz, args.donor_info, seed=args.seed)
