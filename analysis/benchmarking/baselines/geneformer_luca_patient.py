#!/usr/bin/env python3
"""
Patient-level attention-probe baseline for LUCA using pre-computed Geneformer
embeddings (Methods 9.11: "GF" baseline in Supplementary Tables S9-S12). Cell
embeddings are projected to a lower dimension, aggregated per donor via a
PatientAggregator cross-attention head, and trained with early stopping on a
held-out validation split carved from the training donors.

Usage:
    python -m analysis.benchmarking.baselines.geneformer_luca_patient --seed 1 \
        --geneformer-h5ad $CASCADE_DATA_ROOT/LUCA/BASELINE/geneformer/adata_geneformer.h5ad
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

from cascade.data.splits import SPLITS_BY_DATASET
from cascade.explainer.config import CASCADE_DATA_ROOT
from cascade.explainer.utils import group_and_pad, str_list_to_unique_index
from cascade.model.cascade_model import set_seed
from cascade.model.patient_module import PatientAggregator

from analysis.benchmarking.baselines.geneformer_luca import evaluate_classifier, load_common_cells

DONOR_LEVEL_TASKS = ["age-binned", 'cell_type', 'ann_fine', 'ann_coarse', 'cell_type_tumor',
                     'cell_type_major', 'cell_type_neutro', 'cell_type_neutro_coarse']


class EarlyStopping:
    """Stops training once validation loss hasn't improved for `patience` epochs."""

    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, loss):
        if np.isnan(loss):
            self.early_stop = True
            return
        score = -loss
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class ProjectedPatientClassifier(nn.Module):
    """PatientAggregator classifier with a linear projection before aggregation,
    matching the Geneformer baseline's reduced-dimensionality attention head."""

    def __init__(self, embed_dim, num_classes, num_heads=2, num_layers=2, dropout=0.1, proj_dim=128):
        super().__init__()
        self.project = nn.Linear(embed_dim, proj_dim) if proj_dim != embed_dim else nn.Identity()
        self.aggregator = PatientAggregator(proj_dim, num_heads, num_layers, dropout)
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, cell_embeddings, attention_mask=None):
        cell_embeddings = self.project(cell_embeddings)
        patient_embeddings = self.aggregator(cell_embeddings, attention_mask=attention_mask)
        return self.classifier(patient_embeddings)


def main(seed, anno_h5ad, geneformer_h5ad, output_dir, method='geneformer', level='patient', n_epochs=100):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    anno, x_all = load_common_cells(anno_h5ad, geneformer_h5ad)
    split = dict(SPLITS_BY_DATASET['LUCA'])
    split['vali_donors'] = random.sample(split['train_donors'], int(0.1 * len(split['train_donors'])))
    split['train_donors'] = [d for d in split['train_donors'] if d not in split['vali_donors']]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"LUCA_{method}_{level}_linear_probe_results_ES_{seed}.txt"
    done = set()
    if out_file.exists():
        done = set(pd.read_csv(out_file, header=None)[0].values)

    for key in DONOR_LEVEL_TASKS:
        key_ = key.replace(",", "")
        if key_ in done:
            print(f"{key_} is done, skip!")
            continue
        if key not in anno.obs.columns:
            print(f"Key {key} not found in annotations.")
            continue

        y0 = anno.obs[key].values
        donor_ids = anno.obs['donor_id']
        valid_mask = ~anno.obs[key].isna()
        x = x_all[valid_mask.values]
        y0 = y0[valid_mask.values]
        donor_ids = donor_ids[valid_mask.values]

        if anno.obs[key].dtype not in [float, int]:
            y = torch.tensor(str_list_to_unique_index(y0), dtype=torch.long)
        else:
            y = torch.tensor(y0, dtype=torch.long)

        y_dic = {donor_ids.values[i]: y[i] for i in range(len(y))}
        donor_ids_tensor = torch.tensor(donor_ids.values)
        padded_x, patient_ids, attn_mask = group_and_pad(x, donor_ids_tensor)

        y_by_donor = [y_dic[pid.item()].item() for pid in patient_ids]
        unique_vals = sorted(set(y_by_donor))
        val_to_index = {val: i for i, val in enumerate(unique_vals)}
        y = torch.tensor([val_to_index[v] for v in y_by_donor], dtype=torch.long)

        pid = [i.item() for i in patient_ids]
        train_mask, vali_mask, test_mask = (np.isin(pid, split['train_donors']),
                                             np.isin(pid, split['vali_donors']),
                                             np.isin(pid, split['test_donors']))
        x_train, y_train, mask_train = padded_x[train_mask], y[train_mask], attn_mask[train_mask]
        x_vali, y_vali, mask_vali = padded_x[vali_mask], y[vali_mask], attn_mask[vali_mask]
        x_test, y_test, mask_test = padded_x[test_mask], y[test_mask], attn_mask[test_mask]
        n_classes_train, n_classes_test = len(np.unique(y_train)), len(np.unique(y_test))
        print(f"{key}: train={len(x_train)} test={len(x_test)} classes(train={n_classes_train}, test={n_classes_test})")

        if n_classes_train < 2:
            print(f"  Skipping {key}: fewer than 2 classes in training set.")
            continue

        train_loader = DataLoader(TensorDataset(x_train, y_train, mask_train), batch_size=4, shuffle=True)
        vali_loader = DataLoader(TensorDataset(x_vali, y_vali, mask_vali), batch_size=4, shuffle=False)
        test_loader = DataLoader(TensorDataset(x_test, y_test, mask_test), batch_size=4, shuffle=False)

        model = ProjectedPatientClassifier(padded_x.shape[2], len(np.unique(y))).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        early_stopping = EarlyStopping(patience=10)

        for epoch in range(n_epochs):
            model.train()
            total_loss = 0.0
            for batch_X, batch_y, batch_mask in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_X.to(device), batch_mask.to(device)), batch_y.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            total_loss /= max(len(train_loader), 1)

            model.eval()
            vali_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y, batch_mask in vali_loader:
                    outputs = model(batch_X.to(device), batch_mask.to(device))
                    vali_loss += criterion(outputs, batch_y.to(device)).item()
            vali_loss /= max(len(vali_loader), 1)

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}/{n_epochs}, train_loss={total_loss:.4f}, vali_loss={vali_loss:.4f}")
            early_stopping(vali_loss)
            if early_stopping.early_stop:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        model.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for batch_X, batch_y, batch_mask in test_loader:
                outputs = model(batch_X.to(device), batch_mask.to(device))
                y_pred.append(torch.argmax(outputs, dim=1).cpu().numpy())
                y_true.append(batch_y.cpu().numpy())
                y_prob.append(torch.softmax(outputs, dim=1).cpu().numpy())
        y_true, y_pred, y_prob = np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_prob)

        f1, auroc, auprc = evaluate_classifier(y_true, y_pred, y_prob, n_classes_test)
        print(f"{key}: F1={f1:.4f} AUROC={auroc:.4f} AUPRC={auprc:.4f}")
        with open(out_file, 'a') as f:
            f.write(f"{key_}, {len(x_train)}, {len(x_test)}, {n_classes_train}, {n_classes_test}, "
                    f"{f1:.4f}, {auroc:.4f}, {auprc:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--anno-h5ad", type=str, default=str(CASCADE_DATA_ROOT / "LUCA/adata_annotated_protein_coding_clinical.h5ad"))
    parser.add_argument("--geneformer-h5ad", type=str, default=str(CASCADE_DATA_ROOT / "LUCA/BASELINE/geneformer/adata_geneformer.h5ad"))
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()
    main(args.seed, args.anno_h5ad, args.geneformer_h5ad, args.output_dir)
