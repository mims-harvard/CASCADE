#!/usr/bin/env python3
"""
Context ablation: prediction analysis for LUCA embeddings extracted from three
separately-pretrained single-context models (cell_type-only, tissue-only,
disease-only), compared against the full multi-context model, to test how much
each individual context contributes (Methods 9.4/9.7 context-aware tokenization).
"""
import argparse
import gc
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from torch.utils.data import DataLoader, TensorDataset

from cascade.data.splits import SPLITS_BY_DATASET
from cascade.model.patient_module import AttentionClassifier, AttentionRegressor

ENABLE_DOWNSAMPLING = True
MAX_CELLS_PER_DONOR = 250_000

LUCA_TARGET_VARS = ['disease', 'tissue']
LUCA_CELL_LEVEL_VARS = [
    'cell_type_major', 'cell_type_neutro', 'cell_type_neutro_coarse',
    'cell_type', 'ann_fine', 'ann_coarse', 'cell_type_tumor',
]


def build_dataset_configs(embeddings_base):
    split = SPLITS_BY_DATASET['LUCA']
    return {
        'LUCA-tissue-ctx': {
            'embeddings_file': f'{embeddings_base}/embeddings_LUCA_step_tissue.pkl',
            'target_vars': LUCA_TARGET_VARS, 'cell_level_vars': LUCA_CELL_LEVEL_VARS,
            'train_donors': split['train_donors'], 'test_donors': split['test_donors'],
        },
        'LUCA-disease-ctx': {
            'embeddings_file': f'{embeddings_base}/embeddings_LUCA_step_disease.pkl',
            'target_vars': LUCA_TARGET_VARS, 'cell_level_vars': LUCA_CELL_LEVEL_VARS,
            'train_donors': split['train_donors'], 'test_donors': split['test_donors'],
        },
        'LUCA-cell_type-ctx': {
            'embeddings_file': f'{embeddings_base}/embeddings_LUCA_step_cell_type.pkl',
            'target_vars': ['cell_type'], 'cell_level_vars': LUCA_CELL_LEVEL_VARS,
            'train_donors': split['train_donors'], 'test_donors': split['test_donors'],
        },
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def downsample_donor_cells(x, donor_ids, y, max_cells=MAX_CELLS_PER_DONOR, seed=None):
    if seed is not None:
        np.random.seed(seed)
    keep = []
    for d in np.unique(donor_ids):
        idx = np.where(donor_ids == d)[0]
        keep.extend(idx.tolist() if len(idx) <= max_cells else np.random.choice(idx, max_cells, replace=False).tolist())
    keep = np.array(keep, dtype=np.int64)
    x_out = x[keep] if not isinstance(x, torch.Tensor) else x[torch.from_numpy(keep).long()]
    y_out = y[keep] if not isinstance(y, torch.Tensor) else y[torch.from_numpy(keep).long()]
    return x_out, donor_ids[keep], y_out


def train_classify(model, train_loader, test_loader, device, n_epochs=400, is_donor_level=True, patience=30, is_binary=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5)
    best_f1, best_metrics, wait = 0.0, {}, 0

    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            if is_donor_level:
                bx, by, bm = batch
                out = model(bx.to(device), bm.to(device))
            else:
                bx, by = batch
                out = model(bx.to(device))
            loss = criterion(out, by.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        yt, yp, ypr = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                if is_donor_level:
                    bx, by, bm = batch
                    out = model(bx.to(device), bm.to(device))
                else:
                    bx, by = batch
                    out = model(bx.to(device))
                ypr.append(torch.softmax(out, 1).cpu().numpy())
                yp.append(torch.argmax(out, 1).cpu().numpy())
                yt.append(by.cpu().numpy())

        yt, yp, ypr = np.concatenate(yt), np.concatenate(yp), np.concatenate(ypr)
        avg = 'binary' if is_binary else 'weighted'
        f1 = f1_score(yt, yp, average=avg)
        prec = precision_score(yt, yp, average=avg, zero_division=0)
        rec = recall_score(yt, yp, average=avg, zero_division=0)
        bacc = balanced_accuracy_score(yt, yp)
        present = np.unique(yt)
        try:
            if len(present) == 2:
                auroc = roc_auc_score(yt, ypr[:, 1])
                auprc = average_precision_score(label_binarize(yt, classes=present), ypr[:, 1])
            elif len(present) > 2:
                auroc = roc_auc_score(yt, ypr, multi_class='ovr', average='macro')
                auprc = average_precision_score(label_binarize(yt, classes=present), ypr[:, present], average='macro')
            else:
                auroc = auprc = np.nan
        except Exception:
            auroc = auprc = np.nan

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {'f1': f1, 'precision': prec, 'recall': rec, 'balanced_accuracy': bacc, 'auroc': auroc, 'auprc': auprc}
            wait = 0
        else:
            wait += 1

        scheduler.step(f1)
        if (epoch + 1) % 10 == 0 or wait >= patience:
            print(f"    Epoch {epoch+1}: F1={f1:.4f} AUROC={auroc:.4f}", flush=True)
        if wait >= patience:
            print(f"    Early stopping at epoch {epoch+1}", flush=True)
            break

    return best_metrics


def train_regress(model, train_loader, test_loader, device, n_epochs=400, is_donor_level=True, patience=30):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    best_r2, best_metrics, wait = -np.inf, {}, 0

    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            if is_donor_level:
                bx, by, bm = batch
                out = model(bx.to(device), bm.to(device))
            else:
                bx, by = batch
                out = model(bx.to(device))
            loss = criterion(out, by.to(device).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for batch in test_loader:
                if is_donor_level:
                    bx, by, bm = batch
                    out = model(bx.to(device), bm.to(device))
                else:
                    bx, by = batch
                    out = model(bx.to(device))
                yp.append(out.cpu().numpy())
                yt.append(by.cpu().numpy())

        yt, yp = np.concatenate(yt), np.concatenate(yp)
        r2, mse, mae = r2_score(yt, yp), mean_squared_error(yt, yp), mean_absolute_error(yt, yp)
        try:
            pearson, _ = pearsonr(yt, yp)
            spearman, _ = spearmanr(yt, yp)
        except Exception:
            pearson = spearman = np.nan

        if r2 > best_r2:
            best_r2 = r2
            best_metrics = {'r2': r2, 'mse': mse, 'mae': mae, 'pearson': pearson, 'spearman': spearman}
            wait = 0
        else:
            wait += 1

        scheduler.step(mse)
        if (epoch + 1) % 10 == 0 or wait >= patience:
            print(f"    Epoch {epoch+1}: R2={r2:.4f} MSE={mse:.4f}", flush=True)
        if wait >= patience:
            print(f"    Early stopping at epoch {epoch+1}", flush=True)
            break

    return best_metrics


def run_analysis(dataset_name, config, output_file, seed=1):
    print(f"\n{'='*80}\nLUCA SINGLE-CONTEXT PREDICTION: {dataset_name}\n{'='*80}", flush=True)

    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(config['embeddings_file'], 'rb') as f:
        cell_data = pickle.load(f)

    lat = cell_data['embedding']
    x = lat.cpu().numpy() if isinstance(lat, torch.Tensor) else np.asarray(lat)
    print(f"Embeddings shape: {x.shape}", flush=True)

    train_donors, test_donors = config['train_donors'], config['test_donors']
    cell_level_vars = config.get('cell_level_vars', [])
    all_results = []
    file_exists = Path(output_file).exists()

    for var_name in config['target_vars']:
        print(f"\n--- {var_name} ---", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if var_name not in cell_data:
            print("  Not found in cell_data, skipping.", flush=True)
            continue

        y0 = np.array(cell_data[var_name])
        donor_ids = np.array(cell_data['donor_id'])
        valid_mask = ~pd.Series(y0).isna()
        x_v, y0_v, did_v = x[valid_mask], y0[valid_mask], donor_ids[valid_mask]
        if len(x_v) == 0:
            print("  No valid data, skipping.", flush=True)
            continue

        regression_vars = ['S.Score', 'G2M.Score', 'pseudotime', 'pseudotime_ranks', 'Continuous Pseudo-progression Score', 'age']
        if var_name in regression_vars:
            task_type = 'regression'
            yf = pd.to_numeric(y0_v, errors='coerce')
            nm = ~pd.isna(yf)
            if nm.sum() == 0:
                print("  No numeric data, skipping.", flush=True)
                continue
            y = torch.tensor(yf[nm], dtype=torch.float32)
            x_v, did_v = x_v[nm], did_v[nm]
        else:
            task_type = 'classification'
            if pd.Series(y0_v).dtype == 'object':
                y = torch.tensor(LabelEncoder().fit_transform(y0_v), dtype=torch.long)
            else:
                y = torch.tensor(y0_v.astype(int), dtype=torch.long)

        is_donor_level = var_name not in cell_level_vars
        print(f"  Task: {task_type}, donor-level: {is_donor_level}, cells: {len(x_v)}", flush=True)

        if is_donor_level:
            train_mask, test_mask = np.isin(did_v, train_donors), np.isin(did_v, test_donors)
            x_tr_c, x_te_c = x_v[train_mask], x_v[test_mask]
            d_tr, d_te = did_v[train_mask], did_v[test_mask]
            y_tr_c, y_te_c = y[train_mask], y[test_mask]

            if ENABLE_DOWNSAMPLING:
                x_tr_c, d_tr, y_tr_c = downsample_donor_cells(x_tr_c, d_tr, y_tr_c, seed=seed)
                x_te_c, d_te, y_te_c = downsample_donor_cells(x_te_c, d_te, y_te_c, seed=seed)
            if len(np.unique(d_tr)) == 0 or len(np.unique(d_te)) == 0:
                print("  No train or test donors after filtering, skipping.", flush=True)
                continue

            def pad_batch(x_list, y_list):
                lengths = [s.shape[0] for s in x_list]
                max_len = max(lengths)
                dim = x_list[0].shape[1]
                padded = torch.zeros(len(x_list), max_len, dim)
                mask = torch.zeros(len(x_list), max_len, dtype=torch.bool)
                for i, (s, l) in enumerate(zip(x_list, lengths)):
                    padded[i, :l] = s
                    mask[i, :l] = True
                yt = torch.tensor(y_list, dtype=torch.long if task_type == 'classification' else torch.float32)
                return padded, yt, mask

            x_tr_list, y_tr_list, x_te_list, y_te_list = [], [], [], []
            for donors_arr, x_c, y_c, dst_x, dst_y in ((np.unique(d_tr), x_tr_c, y_tr_c, x_tr_list, y_tr_list),
                                                        (np.unique(d_te), x_te_c, y_te_c, x_te_list, y_te_list)):
                for d in donors_arr:
                    m = (d_tr == d) if dst_x is x_tr_list else (d_te == d)
                    dst_x.append(torch.tensor(x_c[m], dtype=torch.float32))
                    yvals = y_c[m].numpy() if isinstance(y_c, torch.Tensor) else y_c[m]
                    lbl = int(np.bincount(yvals).argmax()) if task_type == 'classification' else float(yvals.mean())
                    dst_y.append(lbl)

            if not x_tr_list or not x_te_list:
                print("  Empty train or test after grouping, skipping.", flush=True)
                continue

            X_tr, y_tr, mask_tr = pad_batch(x_tr_list, y_tr_list)
            X_te, y_te, mask_te = pad_batch(x_te_list, y_te_list)
            train_loader = DataLoader(TensorDataset(X_tr, y_tr, mask_tr), batch_size=16, shuffle=True)
            test_loader = DataLoader(TensorDataset(X_te, y_te, mask_te), batch_size=16, shuffle=False)

            dim = x_v.shape[1]
            if task_type == 'classification':
                n_cls = len(torch.unique(y_tr))
                model = AttentionClassifier(dim, n_cls).to(device)
                metrics = train_classify(model, train_loader, test_loader, device, is_donor_level=True, is_binary=(n_cls == 2))
                print(f"  F1={metrics['f1']:.4f} AUROC={metrics['auroc']:.4f}", flush=True)
            else:
                model = AttentionRegressor(dim).to(device)
                metrics = train_regress(model, train_loader, test_loader, device, is_donor_level=True)
                print(f"  R2={metrics['r2']:.4f} Pearson={metrics['pearson']:.4f}", flush=True)

            level, agg = 'donor', 'attention'
            n_cls_val = int(torch.unique(y_tr).numel()) if task_type == 'classification' else None
        else:
            train_mask, test_mask = np.isin(did_v, train_donors), np.isin(did_v, test_donors)
            x_tr, x_te = torch.tensor(x_v[train_mask], dtype=torch.float32), torch.tensor(x_v[test_mask], dtype=torch.float32)
            if task_type == 'classification':
                y_tr, y_te = y[train_mask].long(), y[test_mask].long()
            else:
                y_tr, y_te = y[train_mask].float(), y[test_mask].float()
            print(f"  Cells train: {len(x_tr)}, test: {len(x_te)}", flush=True)

            dim = x_v.shape[1]
            train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=2048, shuffle=True)
            test_loader = DataLoader(TensorDataset(x_te, y_te), batch_size=2048, shuffle=False)

            if task_type == 'classification':
                n_cls = int(y_tr.unique().numel())
                model = nn.Linear(dim, n_cls).to(device)
                metrics = train_classify(model, train_loader, test_loader, device, is_donor_level=False, is_binary=(n_cls == 2))
                print(f"  F1={metrics['f1']:.4f} AUROC={metrics['auroc']:.4f}", flush=True)
            else:
                model = nn.Linear(dim, 1).to(device)
                metrics = train_regress(model, train_loader, test_loader, device, is_donor_level=False)
                print(f"  R2={metrics['r2']:.4f} Pearson={metrics['pearson']:.4f}", flush=True)

            level, agg = 'cell', 'linear'
            n_cls_val = int(y_tr.unique().numel()) if task_type == 'classification' else None

        row = {
            'dataset': dataset_name, 'variable': var_name, 'task_type': task_type, 'level': level,
            'aggregation_method': agg, 'model_type': 'linear' if level == 'cell' else 'attention', 'n_classes': n_cls_val,
        }
        row.update(metrics)
        all_results.append(row)

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(output_file, mode='a', header=not file_exists, index=False)
        file_exists = True
        print(f"  Saved to {output_file}", flush=True)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


def main(embeddings_base, output_dir, seed=1):
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    datasets_cfg = build_dataset_configs(embeddings_base)
    all_results = []
    for dataset_name, config in datasets_cfg.items():
        results = run_analysis(dataset_name, config, f'{output_dir}/results_{dataset_name}.csv', seed=seed)
        all_results.extend(results)

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(f'{output_dir}/results_all.csv', index=False)
        print(f"\nAll results saved to {output_dir}/results_all.csv", flush=True)
        print(df.to_string(), flush=True)
    else:
        print("No results produced.", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-base", type=str, required=True, help="Directory containing the per-context LUCA embedding pickles")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    main(args.embeddings_base, args.output_dir, seed=args.seed)
