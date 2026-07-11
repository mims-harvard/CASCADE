#!/usr/bin/env python3
"""
Broad donor-level and cell-level prediction benchmarking across SEATTLE, AUTISM,
HLCA and LUCA (Methods 9.11): for every available clinical/demographic/pathology
variable, predicts it from frozen, context-agnostic CASCADE embeddings using a
PatientAggregator-based attention model (donor-level) or a linear probe
(cell-level), with early stopping and optional Weights & Biases logging.

Usage:
    python -m analysis.benchmarking.multi_task_prediction --dataset SEATTLE
    python -m analysis.benchmarking.multi_task_prediction --dataset all --output-dir results/
"""
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset

from cascade.data.splits import SPLITS_BY_DATASET
from cascade.explainer.chunked_io import load_chunked_embeddings
from cascade.explainer.config import CASCADE_CKPT_ROOT
from cascade.explainer.utils import group_and_pad
from cascade.model.cascade_model import set_seed
from cascade.model.patient_module import AttentionClassifier, AttentionRegressor

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

# Cells beyond this count are randomly subsampled per donor to bound memory on the
# largest datasets (e.g. SEATTLE); applied independently to train and test splits.
MAX_CELLS_PER_DONOR = 250_000

# Continuous variables that don't already have a per-dataset 'regression_vars' entry.
DEFAULT_REGRESSION_VARS = {'S.Score', 'G2M.Score', 'pseudotime', 'pseudotime_ranks',
                            'Continuous Pseudo-progression Score'}

DATASET_CONFIGS = {
    'SEATTLE': {
        'chunk_dir': CASCADE_CKPT_ROOT / 'SEATTLE_embeds_2',
        'chunk_prefix': 'embeddings_all_labels_complete_contextagnostic_epoch40000_parallel_rank',
        'target_vars': ['disease', 'tissue', 'ADNC', 'APOE4 status', 'Age at death', 'Braak stage',
                         'CERAD score', 'Class', 'Cognitive status', 'Continuous Pseudo-progression Score',
                         'LATE-NC stage', 'Lewy body disease pathology', 'Microinfarct pathology', 'Subclass',
                         'Supertype', 'Thal phase', 'Years of education', 'self_reported_ethnicity', 'sex',
                         'development_stage', 'cell_type'],
        'regression_vars': ['Age at death', 'Years of education'],
        'cell_level_vars': ['cell_type'],
    },
    'AUTISM': {
        'chunk_dir': CASCADE_CKPT_ROOT / 'AUTISM_embeds_20',
        'chunk_prefix': 'embeddings_all_labels_complete_contextagnostic_epoch20_parallel_rank',
        'target_vars': ['Cause_of_death', 'Other_diagnosis', 'ADI-R-A', 'ADI-R-C', 'ADI-R-Bnonverbal',
                         'ADI-R-D', 'ADI-R-Bverbal', 'disease', 'tissue', 'Capbatch', 'Seqbatch', 'Epilepsy',
                         'Attention-deficit/hyperactivity_disorder', 'Cardiac_malformation', 'Depression',
                         'Pneumonia', 'Lead_poisoning', 'Cerebellar_Heterotopia', 'Developmental_Delay',
                         'binned', 'post-mortem-binned', 'cell_type'],
        'regression_vars': ['ADI-R-A', 'ADI-R-C', 'ADI-R-Bnonverbal', 'ADI-R-D', 'ADI-R-Bverbal'],
        'cell_level_vars': ['cell_type'],
    },
    'HLCA': {
        'chunk_dir': CASCADE_CKPT_ROOT / 'HLCA_embeds_3',
        'chunk_prefix': 'embeddings_all_labels_complete_contextagnostic_epoch2_3682000_parallel_rank',
        'target_vars': ['cause_of_death', 'development_stage', 'age_group', 'lung_condition', 'smoking_status',
                         'self_reported_ethnicity', 'BMI_Groups', 'tissue', 'disease', 'sex', 'cell_type',
                         'ann_finest_level', 'ann_level_1', 'ann_level_2', 'ann_level_3', 'ann_level_4',
                         'ann_level_5', 'sequencing_platform', 'anatomical_region_ccf_score'],
        'regression_vars': ['anatomical_region_ccf_score'],
        'cell_level_vars': ['ann_finest_level', 'ann_level_1', 'ann_level_2', 'ann_level_3', 'ann_level_4',
                             'ann_level_5', 'cell_type', 'sequencing_platform'],
    },
    'LUCA': {
        'chunk_dir': CASCADE_CKPT_ROOT / 'LUCA_embeds_3',
        'chunk_prefix': 'embeddings_all_labels_complete_contextagnostic_epoch1_400000_parallel_rank',
        'target_vars': ['ever_smoker', 'tumor_stage', 'EGFR_mutation', 'TP53_mutation', 'ALK_mutation',
                         'BRAF_mutation', 'ERBB2_mutation', 'KRAS_mutation', 'ROS_mutation', 'origin_fine',
                         'origin', 'uicc_stage_combined', 'sex', 'tissue', 'dataset', 'study', 'platform',
                         'age-binned', 'self_reported_ethnicity', 'uicc_stage', 'disease', 'assay', 'organism',
                         'cell_type', 'age', 'ann_fine', 'ann_coarse', 'cell_type_tumor', 'cell_type_major',
                         'cell_type_neutro', 'cell_type_neutro_coarse', 'suspension_type', 'tissue_type',
                         'development_stage'],
        'cell_level_vars': ['cell_type_major', 'cell_type_neutro', 'cell_type_neutro_coarse',
                             'suspension_type', 'cell_type', 'ann_fine', 'ann_coarse', 'cell_type_tumor'],
    },
}


def downsample_donor_cells(x, donor_ids, y, max_cells_per_donor=MAX_CELLS_PER_DONOR, seed=None):
    """Randomly subsample cells so no donor contributes more than max_cells_per_donor."""
    rng = np.random.RandomState(seed)
    keep = []
    for donor_id in np.unique(donor_ids):
        donor_indices = np.where(donor_ids == donor_id)[0]
        if len(donor_indices) > max_cells_per_donor:
            donor_indices = rng.choice(donor_indices, size=max_cells_per_donor, replace=False)
        keep.append(donor_indices)
    keep = np.concatenate(keep)
    return x[keep], donor_ids[keep], y[keep]


def train_and_evaluate_classification(model, train_loader, test_loader, device, n_epochs=400,
                                       is_donor_level=True, early_stopping_patience=30, is_binary=False,
                                       wandb_run=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_f1 = 0.0
    best_metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "balanced_accuracy": 0.0,
                     "auroc": np.nan, "auprc": np.nan}
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        for batch_data in train_loader:
            if is_donor_level:
                batch_X, batch_y, batch_mask = batch_data
                outputs = model(batch_X.to(device), batch_mask.to(device))
            else:
                batch_X, batch_y = batch_data
                outputs = model(batch_X.to(device))
            loss = criterion(outputs, batch_y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for batch_data in test_loader:
                if is_donor_level:
                    batch_X, batch_y, batch_mask = batch_data
                    outputs = model(batch_X.to(device), batch_mask.to(device))
                else:
                    batch_X, batch_y = batch_data
                    outputs = model(batch_X.to(device))
                probs = torch.softmax(outputs, dim=1)
                y_pred.append(torch.argmax(outputs, dim=1).cpu().numpy())
                y_true.append(batch_y.cpu().numpy())
                y_prob.append(probs.cpu().numpy())
        y_true, y_pred, y_prob = np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_prob)

        average = 'binary' if is_binary else 'weighted'
        f1 = f1_score(y_true, y_pred, average=average)
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        present_classes = np.unique(y_true)
        auroc = auprc = np.nan
        try:
            if len(present_classes) == 2:
                auroc = roc_auc_score(y_true, y_prob[:, 1])
                y_true_onehot = label_binarize(y_true, classes=present_classes)
                auprc = average_precision_score(y_true_onehot, y_prob[:, 1])
            elif len(present_classes) > 2:
                auroc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                y_true_onehot = label_binarize(y_true, classes=present_classes)
                auprc = average_precision_score(y_true_onehot, y_prob[:, present_classes], average='macro')
        except ValueError:
            pass

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {"f1": f1, "precision": precision, "recall": recall,
                             "balanced_accuracy": balanced_acc, "auroc": auroc, "auprc": auprc}
            patience_counter = 0
        else:
            patience_counter += 1

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch + 1, "validation_f1": f1, "validation_auroc": auroc})

        scheduler.step(f1)
        if patience_counter >= early_stopping_patience:
            break

    return best_metrics


def train_and_evaluate_regression(model, train_loader, test_loader, device, n_epochs=400,
                                   is_donor_level=True, early_stopping_patience=30, wandb_run=None):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_r2 = -np.inf
    best_metrics = {"r2": -np.inf, "mse": np.inf, "mae": np.inf, "pearson": np.nan, "spearman": np.nan}
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        for batch_data in train_loader:
            if is_donor_level:
                batch_X, batch_y, batch_mask = batch_data
                outputs = model(batch_X.to(device), batch_mask.to(device))
            else:
                batch_X, batch_y = batch_data
                outputs = model(batch_X.to(device))
            loss = criterion(outputs, batch_y.to(device).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_data in test_loader:
                if is_donor_level:
                    batch_X, batch_y, batch_mask = batch_data
                    outputs = model(batch_X.to(device), batch_mask.to(device))
                else:
                    batch_X, batch_y = batch_data
                    outputs = model(batch_X.to(device))
                y_pred.append(outputs.cpu().numpy())
                y_true.append(batch_y.cpu().numpy())
        y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)

        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        try:
            pearson, _ = pearsonr(y_true, y_pred)
            spearman, _ = spearmanr(y_true, y_pred)
        except ValueError:
            pearson = spearman = np.nan

        if r2 > best_r2:
            best_r2 = r2
            best_metrics = {"r2": r2, "mse": mse, "mae": mae, "pearson": pearson, "spearman": spearman}
            patience_counter = 0
        else:
            patience_counter += 1

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch + 1, "validation_r2": r2, "validation_mse": mse})

        scheduler.step(mse)
        if patience_counter >= early_stopping_patience:
            break

    return best_metrics


def _encode_labels(y0_valid, task_type):
    if task_type == 'regression':
        y0_float = pd.to_numeric(y0_valid, errors='coerce')
        valid_mask = ~pd.isna(y0_float)
        return torch.tensor(y0_float[valid_mask], dtype=torch.float32), valid_mask
    valid_mask = np.ones(len(y0_valid), dtype=bool)
    if pd.Series(y0_valid).dtype == 'object':
        y = torch.tensor(LabelEncoder().fit_transform(y0_valid), dtype=torch.long)
    else:
        y = torch.tensor(y0_valid, dtype=torch.long)
    return y, valid_mask


def run_prediction_analysis(dataset_name, config, train_donors, test_donors, seed=1, output_file=None,
                             wandb_project=None, wandb_entity=None):
    """Sweep every target variable for one dataset: donor-level variables are predicted
    via a PatientAggregator attention model over each donor's cells; cell-level
    variables (e.g. cell_type) via a plain linear probe on individual cell embeddings."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading chunked embeddings for {dataset_name}...")
    cell_data = load_chunked_embeddings(config['chunk_dir'], config['chunk_prefix'])
    x = cell_data['embedding']
    x = x.cpu().numpy() if hasattr(x, 'cpu') else np.asarray(x)
    donor_ids_all = np.array(cell_data['donor_id'])

    regression_vars = set(config.get('regression_vars', [])) | DEFAULT_REGRESSION_VARS
    cell_level_vars = set(config.get('cell_level_vars', []))

    all_results = []
    file_exists = False
    if output_file is not None:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        file_exists = Path(output_file).exists()

    for var_name in config['target_vars']:
        if var_name not in cell_data:
            print(f"  [{dataset_name}] '{var_name}' not found in embeddings, skipping")
            continue
        print(f"\n--- {dataset_name}: {var_name} ---")

        y0 = np.array(cell_data[var_name])
        valid_mask = ~pd.Series(y0).isna()
        if valid_mask.sum() == 0:
            continue
        x_valid, donor_ids_valid = x[valid_mask.values], donor_ids_all[valid_mask.values]

        task_type = 'regression' if var_name in regression_vars else 'classification'
        y, numeric_mask = _encode_labels(y0[valid_mask.values], task_type)
        if numeric_mask.sum() == 0:
            continue
        x_valid, donor_ids_valid = x_valid[numeric_mask], donor_ids_valid[numeric_mask]

        is_donor_level = var_name not in cell_level_vars
        var_results = []

        if is_donor_level:
            train_mask = np.isin(donor_ids_valid, train_donors)
            test_mask = np.isin(donor_ids_valid, test_donors)
            x_train_c, x_test_c = x_valid[train_mask], x_valid[test_mask]
            donors_train_c, donors_test_c = donor_ids_valid[train_mask], donor_ids_valid[test_mask]
            y_train_c, y_test_c = y[train_mask], y[test_mask]

            x_train_c, donors_train_c, y_train_c = downsample_donor_cells(x_train_c, donors_train_c, y_train_c, seed=seed)
            x_test_c, donors_test_c, y_test_c = downsample_donor_cells(x_test_c, donors_test_c, y_test_c, seed=seed)
            if len(x_train_c) == 0 or len(x_test_c) == 0:
                continue

            donor_id_to_int = {d: i for i, d in enumerate(sorted(set(donors_train_c) | set(donors_test_c)))}
            train_ids_int = torch.tensor([donor_id_to_int[d] for d in donors_train_c], dtype=torch.long)
            test_ids_int = torch.tensor([donor_id_to_int[d] for d in donors_test_c], dtype=torch.long)

            padded_x_train, unique_train_ids, mask_train = group_and_pad(torch.from_numpy(x_train_c).float(), train_ids_int)
            padded_x_test, unique_test_ids, mask_test = group_and_pad(torch.from_numpy(x_test_c).float(), test_ids_int)

            y_by_donor_train = {}
            for i, did in enumerate(donors_train_c):
                y_by_donor_train[did] = y_train_c[i]
            y_by_donor_test = {}
            for i, did in enumerate(donors_test_c):
                y_by_donor_test[did] = y_test_c[i]
            int_to_donor = {v: k for k, v in donor_id_to_int.items()}
            y_train_donor = torch.stack([y_by_donor_train[int_to_donor[i.item()]] for i in unique_train_ids])
            y_test_donor = torch.stack([y_by_donor_test[int_to_donor[i.item()]] for i in unique_test_ids])

            train_loader = DataLoader(TensorDataset(padded_x_train, y_train_donor, mask_train), batch_size=16, shuffle=True)
            test_loader = DataLoader(TensorDataset(padded_x_test, y_test_donor, mask_test), batch_size=16, shuffle=False)

            wandb_run = None
            if WANDB_AVAILABLE and wandb_project:
                wandb_run = wandb.init(project=wandb_project, entity=wandb_entity, reinit=True,
                                        name=f"{dataset_name}-{var_name}-donor")
            try:
                if task_type == 'classification':
                    n_classes = len(np.unique(y_train_donor.numpy()))
                    model = AttentionClassifier(padded_x_train.shape[2], n_classes).to(device)
                    metrics = train_and_evaluate_classification(model, train_loader, test_loader, device,
                                                                  is_binary=(n_classes == 2), wandb_run=wandb_run)
                else:
                    model = AttentionRegressor(padded_x_train.shape[2]).to(device)
                    metrics = train_and_evaluate_regression(model, train_loader, test_loader, device, wandb_run=wandb_run)
            finally:
                if wandb_run is not None:
                    wandb_run.finish()

            result = {'dataset': dataset_name, 'variable': var_name, 'task_type': task_type, 'level': 'donor',
                      'train_size': len(unique_train_ids), 'test_size': len(unique_test_ids)}
            result.update(metrics)
            all_results.append(result)
            var_results.append(result)

        else:
            train_mask = np.isin(donor_ids_valid, train_donors)
            test_mask = np.isin(donor_ids_valid, test_donors)
            x_train = torch.from_numpy(x_valid[train_mask]).float()
            x_test = torch.from_numpy(x_valid[test_mask]).float()
            y_train, y_test = y[train_mask], y[test_mask]
            if len(x_train) == 0 or len(x_test) == 0:
                continue

            train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=2048, shuffle=True)
            test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=2048, shuffle=False)

            if task_type == 'classification':
                n_classes = len(np.unique(y_train.numpy()))
                model = nn.Linear(x_train.shape[1], n_classes).to(device)
                metrics = train_and_evaluate_classification(model, train_loader, test_loader, device,
                                                              is_donor_level=False, is_binary=(n_classes == 2))
            else:
                model = nn.Linear(x_train.shape[1], 1).to(device)
                metrics = train_and_evaluate_regression(model, train_loader, test_loader, device, is_donor_level=False)

            result = {'dataset': dataset_name, 'variable': var_name, 'task_type': task_type, 'level': 'cell',
                       'train_size': len(x_train), 'test_size': len(x_test)}
            result.update(metrics)
            all_results.append(result)
            var_results.append(result)

        if output_file is not None and var_results:
            pd.DataFrame(var_results).to_csv(output_file, mode='a', header=not file_exists, index=False)
            file_exists = True

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="all",
                        choices=list(DATASET_CONFIGS.keys()) + ["all"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]
    all_results = []
    for dataset_name in dataset_names:
        config = DATASET_CONFIGS[dataset_name]
        split = SPLITS_BY_DATASET[dataset_name]
        results = run_prediction_analysis(
            dataset_name, config, split['train_donors'], split['test_donors'], seed=args.seed,
            output_file=output_dir / f"multi_task_results_{dataset_name}.csv",
            wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
        )
        all_results.extend(results)

    if all_results:
        pd.DataFrame(all_results).to_csv(output_dir / "multi_task_results_all.csv", index=False)
        print(f"\nSaved {len(all_results)} results to {output_dir / 'multi_task_results_all.csv'}")


if __name__ == "__main__":
    main()
