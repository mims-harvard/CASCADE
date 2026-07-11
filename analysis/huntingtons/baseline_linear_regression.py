#!/usr/bin/env python3
"""
Raw-expression linear-regression baseline for Huntington's disease donor-level
features (VS_Grade, CAG_1, CAG_2), predicting from donor-aggregated (or cell-level)
pseudobulk gene expression rather than CASCADE embeddings - a classical-ML baseline
for comparison against the CASCADE-embedding-based predictions (Methods 9.11).
"""
import argparse
import json
import os
import re

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.linear_model import ElasticNetCV, LassoCV, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler, StandardScaler

from cascade.data.splits import SPLITS_BY_DATASET


def sanitize_feature_filename(feature_name):
    """Safe basename for artifact files."""
    s = re.sub(r"[^\w.\-]+", "_", str(feature_name))
    return s.strip("_") or "feature"


def save_regression_artifacts(results, artifact_dir, regularization_method=None):
    """Write .npz (PCA coords, explained variance, predictions) and a metadata JSON."""
    os.makedirs(artifact_dir, exist_ok=True)
    fname = sanitize_feature_filename(results["feature"])
    path = os.path.join(artifact_dir, f"{fname}_artifacts.npz")

    payload = {
        "pca_explained_variance_ratio": results["pca_explained_variance_ratio"],
        "pca_cumulative_variance_ratio": results["pca_cumulative_variance_ratio"],
        "n_components": np.array([results["n_components"]], dtype=np.int32),
        "X_train_pca": results["X_train_pca"],
        "X_test_pca": results["X_test_pca"],
        "y_train": results["y_train"],
        "y_test": results["y_test"],
        "y_train_pred": results["y_train_pred"],
        "y_test_pred": results["y_test_pred"],
        "donor_ids_train": results["donor_ids_train"],
        "donor_ids_test": results["donor_ids_test"],
    }
    comp = results.get("pca_components_")
    if comp is not None and comp.size <= 2_000_000:
        # Avoid multi-GB files when PCA is fit on the full gene space (cell-level).
        payload["pca_components_"] = comp
    np.savez_compressed(path, **payload)
    print(f"Saved {path}")

    meta = {
        "feature": results["feature"],
        "n_components": int(results["n_components"]),
        "total_explained_variance_ratio": float(np.sum(results["pca_explained_variance_ratio"])),
        "best_alpha": float(results["best_alpha"]),
        "artifacts_npz": os.path.abspath(path),
        "pca_loadings_in_npz": bool(comp is not None and comp.size <= 2_000_000),
    }
    if regularization_method is not None:
        meta["regularization_method"] = regularization_method
    if results.get("best_l1_ratio") is not None:
        meta["best_l1_ratio"] = float(results["best_l1_ratio"])
    meta_path = os.path.join(artifact_dir, f"{fname}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path, meta_path


def load_anndata(file_path):
    print(f"Loading AnnData from {file_path}...")
    adata = ad.read_h5ad(file_path)
    print(f"Loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")
    return adata


def aggregate_by_donor(adata, aggregation_method='mean', log_transform=False):
    """Aggregate gene expression by donor_id (mapping legacy 'Donor' -> donor_id = Donor + 1)."""
    print(f"\nAggregating expression data by donor using {aggregation_method}...")

    if 'Donor' in adata.obs.columns:
        print("Found 'Donor' column, mapping to donor_id (Donor + 1)...")
        adata.obs['donor_id'] = (adata.obs['Donor'].astype(float) + 1.0).astype(float)
    elif 'donor_id' not in adata.obs.columns:
        raise ValueError("Neither 'Donor' nor 'donor_id' found in adata.obs.columns")
    else:
        adata.obs['donor_id'] = adata.obs['donor_id'].astype(float)

    unique_donors = adata.obs['donor_id'].unique().astype(float)
    print(f"Found {len(unique_donors)} unique donors")

    if adata.X is None:
        raise ValueError("No expression data found in adata.X")
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    if log_transform:
        X = np.log1p(X)

    donor_expressions, donor_metadata_list, valid_donor_ids = [], [], []
    for donor_id in unique_donors:
        donor_mask = (adata.obs['donor_id'].astype(float) == float(donor_id))
        n_cells = donor_mask.sum()
        if n_cells == 0:
            continue
        donor_cells = X[donor_mask]
        donor_expression = donor_cells.mean(axis=0) if aggregation_method == 'mean' else donor_cells.sum(axis=0)
        donor_expressions.append(donor_expression)
        valid_donor_ids.append(donor_id)
        donor_metadata_list.append(adata.obs[donor_mask].iloc[0].to_dict())

    donor_expression_matrix = np.array(donor_expressions)
    donor_metadata = pd.DataFrame(donor_metadata_list)
    unique_donors = np.array(valid_donor_ids, dtype=float)

    print(f"Aggregated expression shape: {donor_expression_matrix.shape}")
    return donor_expression_matrix, donor_metadata, unique_donors


def run_linear_regression_cell_level(X_cells, y_cells, donor_ids_cells, feature_name, train_donors, test_donors):
    """Ridge regression on cell-level PCA features (GroupKFold CV by donor), predictions
    averaged back to donor level for evaluation - avoids donor-level leakage in the CV split."""
    print(f"\n{'='*60}\nPredicting: {feature_name} (CELL-LEVEL)\n{'='*60}")

    donor_ids_cells = np.array(donor_ids_cells, dtype=float)
    train_donors_set = set(float(d) for d in train_donors)
    test_donors_set = set(float(d) for d in test_donors)

    valid_mask = ~np.isnan(y_cells)
    X_clean, y_clean, donor_ids_clean = X_cells[valid_mask], y_cells[valid_mask], donor_ids_cells[valid_mask]
    if len(y_clean) < 10 or np.std(y_clean) < 1e-6:
        print(f"Warning: insufficient data/variation for {feature_name}. Skipping.")
        return None

    train_mask = np.array([d in train_donors_set for d in donor_ids_clean])
    test_mask = np.array([d in test_donors_set for d in donor_ids_clean])
    X_train, y_train = X_clean[train_mask], y_clean[train_mask]
    X_test, y_test = X_clean[test_mask], y_clean[test_mask]
    donor_ids_train, donor_ids_test = donor_ids_clean[train_mask], donor_ids_clean[test_mask]
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Warning: empty train or test split for {feature_name}. Skipping.")
        return None

    n_train_donors, n_test_donors = len(np.unique(donor_ids_train)), len(np.unique(donor_ids_test))
    print(f"Train: {len(X_train)} cells / {n_train_donors} donors; Test: {len(X_test)} cells / {n_test_donors} donors")

    max_components = min(20, len(X_train) - 2, n_train_donors - 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=max_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    evr = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    cumulative_evr = np.cumsum(evr)
    pca_components_ = np.asarray(pca.components_, dtype=np.float64)
    print(f"  PCA: {X_train_pca.shape[1]} components, {evr.sum():.4f} explained variance")

    alphas = np.logspace(-3, 5, 40)
    groups = donor_ids_train
    gkf = GroupKFold(n_splits=min(5, n_train_donors))
    best_alpha, best_cv_score = None, -np.inf
    for alpha in alphas:
        cv_scores = []
        for train_idx, val_idx in gkf.split(X_train_pca, y_train, groups=groups):
            m = Ridge(alpha=alpha).fit(X_train_pca[train_idx], y_train[train_idx])
            cv_scores.append(r2_score(y_train[val_idx], m.predict(X_train_pca[val_idx])))
        mean_cv_score = np.mean(cv_scores)
        if mean_cv_score > best_cv_score:
            best_cv_score, best_alpha = mean_cv_score, alpha
    print(f"  Best alpha: {best_alpha:.4f} (CV R2: {best_cv_score:.4f})")

    model = Ridge(alpha=best_alpha).fit(X_train_pca, y_train)
    y_train_pred, y_test_pred = model.predict(X_train_pca), model.predict(X_test_pca)

    # Predictions are per-cell; the target is donor-level, so average predictions per donor for evaluation.
    def _to_donor_level(y, y_pred, donor_ids):
        unique_donors = np.unique(donor_ids)
        y_d = np.array([y[donor_ids == d][0] for d in unique_donors])
        y_pred_d = np.array([np.mean(y_pred[donor_ids == d]) for d in unique_donors])
        return y_d, y_pred_d

    y_train_donor, y_train_pred_donor = _to_donor_level(y_train, y_train_pred, donor_ids_train)
    y_test_donor, y_test_pred_donor = _to_donor_level(y_test, y_test_pred, donor_ids_test)

    train_pearson, _ = pearsonr(y_train_donor, y_train_pred_donor)
    test_pearson, _ = pearsonr(y_test_donor, y_test_pred_donor)
    train_spearman, _ = spearmanr(y_train_donor, y_train_pred_donor)
    test_spearman, _ = spearmanr(y_test_donor, y_test_pred_donor)
    test_r2 = r2_score(y_test_donor, y_test_pred_donor)
    print(f"  Test (donor-level): R2={test_r2:.4f} Pearson={test_pearson:.4f}")

    return {
        'feature': feature_name, 'n_train_cells': len(X_train), 'n_test_cells': len(X_test),
        'n_train_donors': n_train_donors, 'n_test_donors': n_test_donors,
        'train_r2': r2_score(y_train_donor, y_train_pred_donor), 'test_r2': test_r2,
        'train_mse': mean_squared_error(y_train_donor, y_train_pred_donor), 'test_mse': mean_squared_error(y_test_donor, y_test_pred_donor),
        'train_mae': mean_absolute_error(y_train_donor, y_train_pred_donor), 'test_mae': mean_absolute_error(y_test_donor, y_test_pred_donor),
        'train_pearson': train_pearson, 'test_pearson': test_pearson,
        'train_spearman': train_spearman, 'test_spearman': test_spearman,
        'best_alpha': best_alpha, 'n_components': X_train_pca.shape[1],
        'pca_explained_variance_ratio': evr, 'pca_cumulative_variance_ratio': cumulative_evr,
        'pca_components_': pca_components_, 'X_train_pca': X_train_pca, 'X_test_pca': X_test_pca,
        'y_train': np.asarray(y_train, dtype=np.float64), 'y_test': np.asarray(y_test, dtype=np.float64),
        'y_train_pred': np.asarray(y_train_pred, dtype=np.float64), 'y_test_pred': np.asarray(y_test_pred, dtype=np.float64),
        'donor_ids_train': np.asarray(donor_ids_train, dtype=np.float64), 'donor_ids_test': np.asarray(donor_ids_test, dtype=np.float64),
        'best_l1_ratio': None, 'regularization_method': 'ridge',
    }


def run_linear_regression(X, y, donor_ids, feature_name, train_donors, test_donors):
    """ElasticNetCV regression on donor-aggregated pseudobulk PCA features."""
    print(f"\n{'='*60}\nPredicting: {feature_name}\n{'='*60}")

    donor_ids = np.array(donor_ids, dtype=float)
    train_donors_set = set(float(d) for d in train_donors)
    test_donors_set = set(float(d) for d in test_donors)

    valid_mask = ~np.isnan(y)
    X_clean, y_clean, donor_ids_clean = X[valid_mask], y[valid_mask], donor_ids[valid_mask]
    if len(y_clean) < 10 or np.std(y_clean) < 1e-6:
        print(f"Warning: insufficient data/variation for {feature_name}. Skipping.")
        return None

    train_mask = np.array([d in train_donors_set for d in donor_ids_clean])
    test_mask = np.array([d in test_donors_set for d in donor_ids_clean])
    X_train, y_train = X_clean[train_mask], y_clean[train_mask]
    X_test, y_test = X_clean[test_mask], y_clean[test_mask]
    donor_ids_train, donor_ids_test = donor_ids_clean[train_mask], donor_ids_clean[test_mask]
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Warning: empty train or test split for {feature_name}. Skipping.")
        return None

    print(f"Train: {len(X_train)} donors, Test: {len(X_test)} donors, {X_train.shape[1]} genes")

    variance_selector = VarianceThreshold(threshold=0.01)
    X_train_var = variance_selector.fit_transform(X_train)
    X_test_var = variance_selector.transform(X_test)

    if X_train_var.shape[1] > 100:
        k_features = min(500, X_train_var.shape[1], len(X_train) * 10)
        univariate_selector = SelectKBest(score_func=f_regression, k=k_features)
        X_train_selected = univariate_selector.fit_transform(X_train_var, y_train)
        X_test_selected = univariate_selector.transform(X_test_var)
    else:
        X_train_selected, X_test_selected = X_train_var, X_test_var

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    max_components = min(20, len(X_train) - 2, X_train_scaled.shape[1])
    if max_components < X_train_scaled.shape[1]:
        pca = PCA(n_components=0.95)
        pca.fit(X_train_scaled)
        n_components_final = min(pca.n_components_, max_components)
        pca = PCA(n_components=n_components_final)
    else:
        pca = PCA(n_components=max_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    evr = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    cumulative_evr = np.cumsum(evr)
    pca_components_ = np.asarray(pca.components_, dtype=np.float64)
    print(f"  PCA: {X_train_pca.shape[1]} components, {evr.sum():.4f} explained variance")

    alphas = np.logspace(-3, 3, 20)
    l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
    elasticnet_cv = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=min(5, len(X_train)), max_iter=2000)
    elasticnet_cv.fit(X_train_pca, y_train)
    model = elasticnet_cv
    best_alpha, best_l1_ratio = elasticnet_cv.alpha_, elasticnet_cv.l1_ratio_
    print(f"  Best ElasticNet alpha={best_alpha:.4f}, l1_ratio={best_l1_ratio:.4f}")

    y_train_pred, y_test_pred = model.predict(X_train_pca), model.predict(X_test_pca)

    train_r2, test_r2 = r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)
    train_pearson, _ = pearsonr(y_train, y_train_pred)
    test_pearson, _ = pearsonr(y_test, y_test_pred)
    train_spearman, _ = spearmanr(y_train, y_train_pred)
    test_spearman, _ = spearmanr(y_test, y_test_pred)
    print(f"  Test: R2={test_r2:.4f} Pearson={test_pearson:.4f}")

    return {
        'feature': feature_name, 'n_train': len(X_train), 'n_test': len(X_test),
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_mse': mean_squared_error(y_train, y_train_pred), 'test_mse': mean_squared_error(y_test, y_test_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred), 'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_pearson': train_pearson, 'test_pearson': test_pearson,
        'train_spearman': train_spearman, 'test_spearman': test_spearman,
        'best_alpha': best_alpha, 'n_components': X_train_pca.shape[1], 'regularization_method': 'elasticnet',
        'pca_explained_variance_ratio': evr, 'pca_cumulative_variance_ratio': cumulative_evr,
        'pca_components_': pca_components_, 'X_train_pca': X_train_pca, 'X_test_pca': X_test_pca,
        'y_train': np.asarray(y_train, dtype=np.float64), 'y_test': np.asarray(y_test, dtype=np.float64),
        'y_train_pred': np.asarray(y_train_pred, dtype=np.float64), 'y_test_pred': np.asarray(y_test_pred, dtype=np.float64),
        'donor_ids_train': np.asarray(donor_ids_train, dtype=np.float64), 'donor_ids_test': np.asarray(donor_ids_test, dtype=np.float64),
        'best_l1_ratio': best_l1_ratio,
    }


def main(adata_path, use_cell_level=False, output_dir='.', artifact_dir=None):
    split = SPLITS_BY_DATASET['HH']
    target_features = ['VS_Grade', 'CAG_1', 'CAG_2']

    print("=" * 80)
    print("DONOR-LEVEL FEATURE PREDICTION USING LINEAR REGRESSION (raw expression baseline)")
    print("=" * 80)

    adata = load_anndata(adata_path)
    print("\nAvailable columns in adata.obs:", adata.obs.columns.tolist())

    if use_cell_level:
        if 'Donor' in adata.obs.columns:
            adata.obs['donor_id'] = (adata.obs['Donor'].astype(float) + 1.0).astype(float)
        elif 'donor_id' not in adata.obs.columns:
            raise ValueError("Neither 'Donor' nor 'donor_id' found in adata.obs.columns")
        X_cells = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        donor_ids_cells = adata.obs['donor_id'].values.astype(float)
        donor_metadata = adata.obs.copy()
    else:
        donor_expression, donor_metadata, donor_ids = aggregate_by_donor(adata, aggregation_method='mean')

    available_features = [f for f in target_features if f in donor_metadata.columns]
    missing_features = [f for f in target_features if f not in donor_metadata.columns]
    print(f"\nAvailable target features: {available_features}")
    if missing_features:
        print(f"Missing target features: {missing_features}")
    if not available_features:
        print("\nError: none of the target features are available in the data!")
        return

    all_results = []
    for feature in available_features:
        if pd.isna(donor_metadata[feature]).all():
            print(f"\nSkipping {feature}: all values are missing")
            continue

        if use_cell_level:
            y_cells = donor_metadata[feature].values
            results = run_linear_regression_cell_level(
                X_cells, y_cells, donor_ids_cells=donor_ids_cells, feature_name=feature,
                train_donors=split['train_donors'], test_donors=split['test_donors'],
            )
        else:
            y = donor_metadata[feature].values
            results = run_linear_regression(
                donor_expression, y, donor_ids=donor_ids, feature_name=feature,
                train_donors=split['train_donors'], test_donors=split['test_donors'],
            )
        if results is not None:
            all_results.append(results)

    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    if not all_results:
        print("No successful predictions completed.")
        return

    summary_df = pd.DataFrame([{
        'Feature': r['feature'], 'Best Alpha': r['best_alpha'], 'Train R2': r['train_r2'], 'Test R2': r['test_r2'],
        'Test MSE': r['test_mse'], 'Test MAE': r['test_mae'], 'Test Pearson r': r['test_pearson'], 'Test Spearman': r['test_spearman'],
    } for r in all_results])
    print("\nResults Summary:")
    print(summary_df.to_string(index=False))

    output_file = os.path.join(output_dir, "donor_feature_prediction_results.csv")
    summary_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    artifact_dir = artifact_dir or os.environ.get("HH_BASELINE_ARTIFACT_DIR", os.path.join(output_dir, "baseline_hh_pca_artifacts"))
    pca_summary_rows = []
    for r in all_results:
        reg = r.get("regularization_method") or ("ridge" if use_cell_level else "elasticnet")
        npz_path, meta_path = save_regression_artifacts(r, artifact_dir, regularization_method=reg)
        pca_summary_rows.append({
            "Feature": r["feature"], "n_components": r["n_components"],
            "total_explained_variance_ratio": float(np.sum(r["pca_explained_variance_ratio"])),
            "artifacts_npz": npz_path, "meta_json": meta_path,
        })
    pd.DataFrame(pca_summary_rows).to_csv(os.path.join(artifact_dir, "pca_artifacts_summary.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata-path", type=str, required=True, help="Path to the HH AnnData (.h5ad) file")
    parser.add_argument("--cell-level", action="store_true", help="Use cell-level features instead of donor-aggregated pseudobulk")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--artifact-dir", type=str, default=None)
    args = parser.parse_args()
    main(args.adata_path, use_cell_level=args.cell_level, output_dir=args.output_dir, artifact_dir=args.artifact_dir)
