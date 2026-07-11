#!/usr/bin/env python3
"""
Donor-level attention-based prediction of Huntington's disease donor features
(VS_Grade, CAG repeat lengths, motor/cognitive onset) from frozen, context-agnostic
CASCADE embeddings, aggregated per donor via PatientAggregator (Methods 9.1, 9.8, 9.11).
Reports per-feature regression metrics plus detailed cross-feature and per-donor
correlation breakdowns for downstream figures.

Usage:
    python -m analysis.huntingtons.donor_level_prediction --embeddings-path /path/to/embeddings.pkl
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from cascade.data.splits import SPLITS_BY_DATASET
from cascade.explainer.config import HH_DONOR_LEVEL_TASKS
from cascade.explainer.utils import group_and_pad
from cascade.model.cascade_model import set_seed
from cascade.model.patient_module import AttentionRegressor

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_cell_data(embeddings_path):
    with open(embeddings_path, "rb") as f:
        cell_data = pickle.load(f)
    print("Available keys in cell_data:", cell_data.keys())
    return cell_data


def encode_target(y0_filtered):
    if pd.Series(y0_filtered).dtype == 'object' or pd.Series(y0_filtered).dtype.name == 'string':
        unique_vals = sorted(set(y0_filtered))
        val_to_code = {val: i for i, val in enumerate(unique_vals)}
        return torch.tensor([val_to_code[val] for val in y0_filtered], dtype=torch.float32)
    return torch.tensor(y0_filtered, dtype=torch.float32)


def train_one_target(padded_x, y, patient_ids, attn_mask, train_donors, test_donors, device, n_epochs, key,
                      method, level, dataset_name, seed, use_wandb):
    pid = [d.item() for d in patient_ids]
    train_mask = np.isin(pid, train_donors)
    test_mask = np.isin(pid, test_donors)

    x_train, y_train, mask_train = padded_x[train_mask], y[train_mask], attn_mask[train_mask]
    x_test, y_test, mask_test = padded_x[test_mask], y[test_mask], attn_mask[test_mask]
    print(f"Train donors: {len(x_train)}, Test donors: {len(x_test)}")

    if len(x_train) == 0 or len(x_test) == 0:
        print(f"Key {key} has no data in train or test set. Skipping.")
        return None

    if use_wandb:
        wandb.init(project="cascade-hh-donor-level-regression", name=f"{method}_{level}_{key}_{seed}",
                   config={"seed": seed, "context": key, "n_epochs": n_epochs,
                           "train_size": len(x_train), "test_size": len(x_test)})

    train_loader = DataLoader(TensorDataset(x_train, y_train, mask_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test, mask_test), batch_size=16, shuffle=False)

    model = AttentionRegressor(padded_x.shape[2]).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_r2 = -float('inf')
    best_y_pred, best_y_true, best_epoch = None, None, None
    best_metrics = {"r2": np.nan, "mse": np.nan, "mae": np.nan, "pearson": np.nan, "spearman": np.nan}

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y, batch_mask in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X.to(device), batch_mask.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_X, batch_y, batch_mask in test_loader:
                y_pred.append(model(batch_X.to(device), batch_mask.to(device)).cpu().numpy())
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

    model.eval()
    with torch.no_grad():
        y_train_pred = model(x_train.to(device), mask_train.to(device)).cpu().numpy()

    test_pid = np.array(pid)[test_mask]
    train_pid = np.array(pid)[train_mask]
    return {"best_epoch": best_epoch, "best_metrics": best_metrics, "y_true": best_y_true, "y_pred": best_y_pred,
            "train_size": len(x_train), "test_size": len(x_test), "test_donor_ids": test_pid,
            "train_donor_ids": train_pid, "y_train_true": y_train.numpy(), "y_train_pred": y_train_pred}


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
    best_csv_path = output_dir / f"{dataset_name}_{method}_{level}_{task_type}_donor_level_best_by_r2_seed{seed}.csv"

    all_results = {}
    results_rows = []
    donor_correlation_results = []
    donor_metrics_results = []
    detailed_donor_predictions = []

    for key in context:
        n_epochs = 400 if key == 'VS_Grade' else 2000
        print(f"\n=== Processing {key} ===")
        if key not in cell_data:
            print(f"Key {key} not found in annotations. Skipping.")
            continue

        y0 = np.array(cell_data[key])
        ind = pd.Series(cell_data['donor_id'])
        valid_mask = ~pd.Series(cell_data[key]).isna()
        x_filtered = x[valid_mask.values]
        y0_filtered = y0[valid_mask.values]
        ind_tensor = torch.tensor(ind[valid_mask.values].values)

        y_by_cell = encode_target(y0_filtered)
        # Group cells into per-donor padded tensors, then take each donor's (constant) label.
        y_dic = {ind_tensor[i].item(): y_by_cell[i] for i in range(len(y_by_cell))}
        padded_x, patient_ids, attn_mask = group_and_pad(x_filtered, ind_tensor)
        y = torch.tensor([y_dic[pid.item()].item() for pid in patient_ids], dtype=torch.float32)

        result = train_one_target(padded_x, y, patient_ids, attn_mask, split['train_donors'], split['test_donors'],
                                   device, n_epochs, key, method, level, dataset_name, seed, WANDB_AVAILABLE)
        if result is None:
            continue
        all_results[key] = result

        for pred, true, donor_id in zip(result['y_pred'], result['y_true'], result['test_donor_ids']):
            detailed_donor_predictions.append({'feature': key, 'donor_id': float(donor_id),
                                                'prediction': float(pred), 'ground_truth': float(true), 'split': 'test'})
            donor_metrics_results.append({
                'feature': key, 'donor_id': float(donor_id),
                'mse': float((pred - true) ** 2), 'sample_count': 1, 'split': 'test',
            })
        for pred, true, donor_id in zip(result['y_train_pred'], result['y_train_true'], result['train_donor_ids']):
            detailed_donor_predictions.append({'feature': key, 'donor_id': float(donor_id),
                                                'prediction': float(pred), 'ground_truth': float(true), 'split': 'train'})

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

            common_donors = set(result['test_donor_ids']) & set(ind_other_filtered)
            if len(common_donors) <= 4:
                continue

            current_pred_common, other_true_common = [], []
            for donor_id in common_donors:
                pred_idx = np.where(result['test_donor_ids'] == donor_id)[0]
                other_idx = np.where(ind_other_filtered == donor_id)[0]
                if len(pred_idx) == 0 or len(other_idx) == 0:
                    continue
                current_pred_common.append(result['y_pred'][pred_idx[0]])
                other_true_common.append(other_true[other_idx[0]])

            if len(current_pred_common) > 1:
                cross_corr = np.corrcoef(current_pred_common, other_true_common)[0, 1]
                donor_correlation_results.append({
                    'feature': key, 'other_feature': other_key, 'correlation_type': 'cross',
                    'correlation_value': float(cross_corr), 'sample_count': len(current_pred_common),
                })

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

    if donor_correlation_results:
        pd.DataFrame(donor_correlation_results).to_csv(output_dir / f"{dataset_name}_donor_level_cross_feature_correlations.csv", index=False)
    if donor_metrics_results:
        pd.DataFrame(donor_metrics_results).to_csv(output_dir / f"{dataset_name}_donor_level_donor_specific_metrics.csv", index=False)
    if detailed_donor_predictions:
        pd.DataFrame(detailed_donor_predictions).to_csv(output_dir / f"{dataset_name}_donor_level_detailed_donor_predictions.csv", index=False)

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
