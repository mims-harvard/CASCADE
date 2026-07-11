#!/usr/bin/env python3
"""
Cell-level linear-probe prediction of Huntington's disease donor features
(VS_Grade, CAG repeat lengths, motor/cognitive onset) from frozen, context-agnostic
CASCADE embeddings (Methods 9.1, 9.11). Reports per-feature regression metrics plus
detailed cross-feature and per-donor correlation breakdowns for downstream figures.

Usage:
    python -m analysis.huntingtons.cell_level_prediction --embeddings-path /path/to/embeddings.pkl
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from cascade.data.splits import SPLITS_BY_DATASET
from cascade.explainer.config import HH_DONOR_LEVEL_TASKS
from cascade.model.cascade_model import set_seed
from cascade.model.patient_module import LinearProbe

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_cell_data(embeddings_path):
    import pickle
    with open(embeddings_path, "rb") as f:
        cell_data = pickle.load(f)
    print("Available keys in cell_data:", cell_data.keys())
    return cell_data


def encode_target(y0_filtered):
    """Numeric targets pass through; string targets are integer-coded (regression on the code)."""
    if pd.Series(y0_filtered).dtype == 'object' or pd.Series(y0_filtered).dtype.name == 'string':
        unique_vals = sorted(set(y0_filtered))
        val_to_code = {val: i for i, val in enumerate(unique_vals)}
        return torch.tensor([val_to_code[val] for val in y0_filtered], dtype=torch.float32), val_to_code
    return torch.tensor(y0_filtered, dtype=torch.float32), None


def train_one_target(x_filtered, y, train_mask, test_mask, device, n_epochs, key, method, level,
                      dataset_name, seed, use_wandb):
    x_train, y_train = x_filtered[train_mask], y[train_mask]
    x_test, y_test = x_filtered[test_mask], y[test_mask]
    print(f"Train cells: {len(x_train)}, Test cells: {len(x_test)}")

    if len(x_train) == 0 or len(x_test) == 0:
        print(f"Key {key} has no data in train or test set. Skipping.")
        return None

    if use_wandb:
        wandb.init(project="cascade-hh-cell-level-regression", name=f"{method}_{level}_{key}_{seed}",
                   config={"seed": seed, "context": key, "n_epochs": n_epochs,
                           "train_size": len(x_train), "test_size": len(x_test)})

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=16, shuffle=False)

    model = LinearProbe(x_train.shape[1], 1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_r2 = -float('inf')
    best_y_pred, best_y_true, best_epoch = None, None, None
    best_metrics = {"r2": np.nan, "mse": np.nan, "mae": np.nan, "pearson": np.nan, "spearman": np.nan}

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X.to(device)).squeeze(), batch_y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                y_pred.append(model(batch_X.to(device)).cpu().numpy())
                y_true.append(batch_y.cpu().numpy())
        y_true, y_pred = np.concatenate(y_true).flatten(), np.concatenate(y_pred).flatten()

        mse, mae, r2 = mean_squared_error(y_true, y_pred), mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)
        pearson_corr, _ = pearsonr(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)

        if epoch % 50 == 0:
            print(f"{key}: MSE={mse:.4f} MAE={mae:.4f} R2={r2:.4f} Pearson={pearson_corr:.4f}")
        if use_wandb:
            wandb.log({f"mse_{key}": mse, f"r2_{key}": r2, f"pearson_{key}": pearson_corr})

        if r2 > best_r2:
            best_r2 = r2
            best_epoch = epoch + 1
            best_metrics = {"r2": float(r2), "mse": float(mse), "mae": float(mae),
                             "pearson": float(pearson_corr), "spearman": float(spearman_corr)}
            best_y_pred, best_y_true = y_pred.copy(), y_true.copy()

    if use_wandb:
        try:
            wandb.finish()
        except Exception:
            pass

    return {"model": model, "best_epoch": best_epoch, "best_metrics": best_metrics,
            "y_true": best_y_true, "y_pred": best_y_pred, "train_size": len(x_train), "test_size": len(x_test)}


def cross_feature_correlations(key, current_pred, current_donor_ids, context, cell_data,
                                correlation_results, donor_correlation_results, donor_metrics_results):
    """Self- and cross-feature correlations, overall and split by donor (Methods 9.1 HD case study)."""
    current_true = None  # filled by caller before calling if needed; kept for clarity

    for donor_id in np.unique(current_donor_ids):
        donor_mask = (current_donor_ids == donor_id)
        donor_pred = current_pred[donor_mask]
        if len(donor_pred) <= 1:
            continue
        for other_key in context:
            if other_key == key or other_key not in cell_data:
                continue
            y0_other = np.array(cell_data[other_key])
            ind_other = np.array(cell_data['donor_id'])
            mask_other = ~pd.isna(y0_other)
            y0_other_filtered, ind_other_filtered = y0_other[mask_other], ind_other[mask_other]

            if pd.Series(y0_other_filtered).dtype == 'object':
                unique_vals = sorted(set(y0_other_filtered))
                val_to_code = {val: i for i, val in enumerate(unique_vals)}
                other_true = np.array([val_to_code[v] for v in y0_other_filtered])
            else:
                other_true = y0_other_filtered

            other_donor_mask = (ind_other_filtered == donor_id)
            if not other_donor_mask.any():
                continue
            other_donor_true = other_true[other_donor_mask]
            other_donor_true = np.mean(other_donor_true) if len(other_donor_true) > 1 else other_donor_true[0]

            donor_pred_mean = np.mean(donor_pred)
            cross_corr = np.corrcoef([donor_pred_mean], [other_donor_true])[0, 1]
            donor_correlation_results.append({
                'feature': key, 'other_feature': other_key, 'correlation_type': 'cross',
                'donor_id': float(donor_id), 'correlation_value': float(cross_corr),
                'sample_count': len(donor_pred), 'split': 'test',
            })


def main(embeddings_path, dataset_name='HH', method='cascade', level='patient', seed=1,
         output_dir='.', filter_disease=False):
    set_seed(seed)
    task_type = 'regression'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cell_data = load_cell_data(embeddings_path)
    if filter_disease and 'context' in cell_data:
        disease_mask = (np.array(cell_data['context']) != 'DISEASE')
        for key in cell_data:
            cell_data[key] = cell_data[key][disease_mask] if key == 'embedding' else np.array(cell_data[key])[disease_mask]

    context = HH_DONOR_LEVEL_TASKS + ['Onset/Motor', 'Onset/Cog']
    split = SPLITS_BY_DATASET[dataset_name]

    x = cell_data['embedding'].float()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_csv_path = output_dir / f"{dataset_name}_{method}_{level}_{task_type}_cell_level_best_by_r2_seed{seed}.csv"

    all_results = {}
    results_rows = []
    correlation_results = []
    donor_correlation_results = []
    donor_metrics_results = []
    detailed_predictions = []

    for key in context:
        n_epochs = 400 if key == 'VS_Grade' else 200
        print(f"\n=== Processing {key} ===")
        if key not in cell_data:
            print(f"Key {key} not found in annotations. Skipping.")
            continue

        y0 = np.array(cell_data[key])
        ind = pd.Series(cell_data['donor_id'])
        valid_mask = ~pd.Series(cell_data[key]).isna()
        x_filtered = x[valid_mask.values]
        y0_filtered = y0[valid_mask.values]
        ind_filtered = ind[valid_mask.values]

        y, _ = encode_target(y0_filtered)
        donor_ids = torch.tensor(ind_filtered.values)

        pid = [d.item() for d in donor_ids]
        train_mask = np.isin(pid, split['train_donors'])
        test_mask = np.isin(pid, split['test_donors'])

        result = train_one_target(x_filtered, y, train_mask, test_mask, device, n_epochs, key,
                                   method, level, dataset_name, seed, WANDB_AVAILABLE)
        if result is None:
            continue
        all_results[key] = result

        test_donor_ids = ind_filtered[test_mask].values
        train_donor_ids = ind_filtered[train_mask].values
        for pred, true, donor_id in zip(result['y_pred'], result['y_true'], test_donor_ids):
            detailed_predictions.append({'feature': key, 'donor_id': float(donor_id),
                                          'prediction': float(pred), 'ground_truth': float(true), 'split': 'test'})

        correlation_results.append({
            'feature': key, 'other_feature': key, 'correlation_type': 'self',
            'correlation_value': float(np.corrcoef(result['y_pred'], result['y_true'])[0, 1]) if len(result['y_pred']) > 1 else 0.0,
            'sample_count': len(result['y_pred']),
        })

        for donor_id in np.unique(test_donor_ids):
            donor_mask = (test_donor_ids == donor_id)
            donor_pred, donor_true = result['y_pred'][donor_mask], result['y_true'][donor_mask]
            if len(donor_pred) <= 1:
                continue
            donor_correlation_results.append({
                'feature': key, 'other_feature': key, 'correlation_type': 'self', 'donor_id': float(donor_id),
                'correlation_value': float(np.corrcoef(donor_pred, donor_true)[0, 1]),
                'sample_count': len(donor_pred), 'split': 'test',
            })
            donor_metrics_results.append({
                'feature': key, 'donor_id': float(donor_id),
                'mse': float(mean_squared_error(donor_true, donor_pred)),
                'mae': float(mean_absolute_error(donor_true, donor_pred)),
                'r2': float(r2_score(donor_true, donor_pred)),
                'sample_count': len(donor_pred), 'split': 'test',
            })

        cross_feature_correlations(key, result['y_pred'], test_donor_ids, context, cell_data,
                                    correlation_results, donor_correlation_results, donor_metrics_results)

        results_rows.append({
            "label": key, "seed": seed, "method": method, "level": level, "dataset": dataset_name,
            "task_type": task_type, "best_epoch": result["best_epoch"],
            "train_size": result["train_size"], "test_size": result["test_size"],
            **{f"best_{k}": v for k, v in result["best_metrics"].items()},
        })

    pd.DataFrame(results_rows).to_csv(best_csv_path, index=False)
    print(f"Results saved to: {best_csv_path}")

    if all_results:
        overall_best_key = max(all_results.keys(), key=lambda k: all_results[k]["best_metrics"]["r2"])
        print(f"\n=== OVERALL BEST PERFORMING FEATURE: {overall_best_key} "
              f"(R2={all_results[overall_best_key]['best_metrics']['r2']:.4f}) ===")

    if correlation_results:
        pd.DataFrame(correlation_results).to_csv(output_dir / f"{dataset_name}_cell_level_individual_correlations.csv", index=False)
    if donor_correlation_results:
        pd.DataFrame(donor_correlation_results).to_csv(output_dir / f"{dataset_name}_cell_level_donor_specific_correlations.csv", index=False)
    if donor_metrics_results:
        pd.DataFrame(donor_metrics_results).to_csv(output_dir / f"{dataset_name}_cell_level_donor_specific_metrics.csv", index=False)
    if detailed_predictions:
        pd.DataFrame(detailed_predictions).to_csv(output_dir / f"{dataset_name}_cell_level_detailed_predictions.csv", index=False)

    print("\n=== SUMMARY ===")
    print(f"Processed {len(all_results)} regression targets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="HH")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--filter-disease", action="store_true")
    args = parser.parse_args()
    main(args.embeddings_path, dataset_name=args.dataset, seed=args.seed,
         output_dir=args.output_dir, filter_disease=args.filter_disease)
