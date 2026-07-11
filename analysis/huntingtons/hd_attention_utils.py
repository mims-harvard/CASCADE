"""
Shared helpers for donor-level, cell-type attention analyses of the
Huntington's disease CASCADE-Explainer outputs (Figure 5c-g).

Expects donor-level attention-weight caches (.npz) with keys:
attention_weights (n_layers, n_donors, n_query, n_cells), patient_y (donor
label, e.g. CAG length or VS grade), patient_ids, patient_cell_types.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

CLINICAL_RENAME = {
    'Donor': 'donor_id', 'Age': 'age', 'Sex': 'sex',
    'disease': 'disease_status', 'VS_Grade': 'vs_grade',
    'Onset/Motor': 'onset_motor', 'Onset/Cog': 'onset_cognitive',
}
CLINICAL_COLUMNS = ['donor_id', 'age', 'sex', 'disease_status', 'vs_grade', 'onset_motor', 'onset_cognitive']


def load_donor_info(donor_info_path):
    donor_info = pd.read_csv(donor_info_path)
    donor_info.columns = donor_info.columns.str.strip()
    return donor_info.rename(columns=CLINICAL_RENAME)


def create_donor_df(data_dict, metric_name=None):
    """Per-donor, per-cell-type max attention importance (query position 0
    = the [PAT] token), restricted to donors with a non-missing label.

    The originally uploaded version of this helper filtered
    attention_weights by a `valid` (non-NaN patient_y) boolean mask but then
    indexed patient_ids/patient_y/patient_cell_types with the same loop
    variable *unfiltered* -- i.e. it mixed a compacted index space with the
    raw one. Whenever an invalid donor appeared before the last valid donor
    in the raw array, this silently dropped or mislabeled donors. Verified
    against the actual CAG_1/CAG_2 caches: this dropped exactly 1 of 52
    valid donors (the raw position landed on a different, invalid donor,
    tripping the `if not valid[i]: continue` guard). Fixed here by indexing
    all four arrays with the same `valid` mask before iterating.
    """
    valid = ~np.isnan(data_dict["patient_y"].astype(float))
    x = data_dict["attention_weights"][:, valid, :, :]
    ids_v = data_dict["patient_ids"][valid]
    y_v = data_dict["patient_y"][valid].astype(float)
    ct_v = data_dict["patient_cell_types"][valid]

    rows = []
    for i in range(x.shape[1]):
        did = int(ids_v[i])
        yval = float(y_v[i])
        ct_i = [str(c) for c in ct_v[i]]
        max_z = x[:, i, 0, :].max(axis=0)
        for ct_j, score in zip(ct_i, max_z):
            row = {'donor_id': did, 'cell_type': ct_j, 'importance': float(score), 'y_value': yval}
            if metric_name is not None:
                row['metric'] = metric_name
            rows.append(row)
    return pd.DataFrame(rows)


def normalize_per_donor(df):
    """Min-max normalize importance to [0, 1] within each donor."""
    out = df.copy()
    for did in df['donor_id'].unique():
        m = df['donor_id'] == did
        v = df.loc[m, 'importance']
        mn, mx = v.min(), v.max()
        out.loc[m, 'importance'] = (v - mn) / (mx - mn) if mx > mn else 0.5
    return out


def prepare_features(df_norm, meta_cols=CLINICAL_COLUMNS):
    """Pivot per-donor, per-cell-type importance into a donor x cell-type
    feature matrix, with clinical metadata columns attached."""
    pivot = df_norm.pivot_table(index='donor_id', columns='cell_type', values='importance', aggfunc='mean')
    pivot.columns = [f'{c}_importance' for c in pivot.columns]
    pivot = pivot.reset_index()
    meta = df_norm[[c for c in meta_cols if c in df_norm.columns]].drop_duplicates('donor_id')
    return pivot.merge(meta, on='donor_id', how='left')


def cluster_best_k(x_scaled, k_range=range(2, 9), seed=42):
    """KMeans over a range of k, selected by max silhouette score."""
    best_k, best_sil, best_labels = 2, -1, None
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(x_scaled)
        sil = silhouette_score(x_scaled, labels)
        if sil > best_sil:
            best_sil, best_k, best_labels = sil, k, labels
    return best_labels, best_k, best_sil


def cluster_k2(x_scaled, seed=42):
    """Fixed k=2 KMeans (used where k=2 is assumed rather than selected)."""
    return KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(x_scaled)
