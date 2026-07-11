#!/usr/bin/env python3
"""
Cell-level linear-probe baseline for LUCA using pre-computed Geneformer embeddings
(Methods 9.11: "GF" baseline in Supplementary Tables S9-S12). Trains one linear
classifier per cell-type task directly on Geneformer's frozen cell embeddings,
using the same fixed donor split as CASCADE, and appends F1/AUROC/AUPRC per task
to a resumable results file.

Usage:
    python -m analysis.benchmarking.baselines.geneformer_luca --seed 1 \
        --geneformer-h5ad $CASCADE_DATA_ROOT/LUCA/BASELINE/geneformer/adata_geneformer.h5ad
"""
import argparse
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
from cascade.explainer.utils import str_list_to_unique_index
from cascade.model.cascade_model import set_seed
from cascade.model.patient_module import LinearProbe

CELL_LEVEL_TASKS = ['cell_type', 'ann_fine', 'ann_coarse', 'cell_type_tumor',
                     'cell_type_major', 'cell_type_neutro', 'cell_type_neutro_coarse']


def load_common_cells(anno_h5ad, geneformer_h5ad, embedding_key='geneformer'):
    """Load CASCADE's annotated LUCA AnnData and the Geneformer embedding AnnData,
    restricted to the cells present in both (join on observation_joinid)."""
    anno = sc.read_h5ad(anno_h5ad, backed='r')
    adata = sc.read_h5ad(geneformer_h5ad, backed='r')
    anno.obs.index = anno.obs['observation_joinid']
    adata.obs.index = adata.obs['observation_joinid']
    common_cells = np.intersect1d(anno.obs['observation_joinid'], adata.obs['observation_joinid'])
    anno, adata = anno[common_cells, :], adata[common_cells, :]
    return anno, torch.tensor(adata.obsm[embedding_key], dtype=torch.float32)


def evaluate_classifier(y_true, y_pred, y_prob, n_classes_test):
    present_classes = np.unique(y_true)
    f1 = f1_score(y_true, y_pred, average='weighted')
    if n_classes_test > 2:
        y_true_onehot = label_binarize(y_true, classes=present_classes)
        y_prob_present = y_prob[:, present_classes]
        y_prob_present = y_prob_present / y_prob_present.sum(axis=1, keepdims=True)
        auroc = roc_auc_score(y_true, y_prob_present, labels=present_classes, multi_class='ovr', average='macro')
        auprc = average_precision_score(y_true_onehot, y_prob_present, average='macro')
    elif n_classes_test == 2:
        y_true_onehot = label_binarize(y_true, classes=present_classes)
        auroc = roc_auc_score(y_true, y_prob[:, 1])
        auprc = average_precision_score(y_true_onehot, y_prob[:, 1])
    else:
        auroc = auprc = np.nan
    return f1, auroc, auprc


def main(seed, anno_h5ad, geneformer_h5ad, output_dir, method='geneformer', level='cell', n_epochs=100):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    anno, x_all = load_common_cells(anno_h5ad, geneformer_h5ad)
    split = SPLITS_BY_DATASET['LUCA']

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"LUCA_{method}_{level}_linear_probe_results_{seed}.txt"
    done = set()
    if out_file.exists():
        done = set(pd.read_csv(out_file, header=None)[0].values)

    for key in CELL_LEVEL_TASKS:
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

        train_mask = donor_ids.isin(split['train_donors'])
        test_mask = donor_ids.isin(split['test_donors'])
        x_train, y_train = x[train_mask.values], y[train_mask.values]
        x_test, y_test = x[test_mask.values], y[test_mask.values]
        n_classes_train, n_classes_test = len(np.unique(y_train)), len(np.unique(y_test))
        print(f"{key}: train={len(x_train)} test={len(x_test)} classes(train={n_classes_train}, test={n_classes_test})")

        if n_classes_train < 2 or len(x_train) < 10 or len(x_test) < 10:
            print(f"  Skipping {key}: insufficient classes/samples.")
            continue

        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=512, shuffle=True)
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=512, shuffle=False)

        model = LinearProbe(x_train.shape[1], len(np.unique(y))).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(n_epochs):
            model.train()
            total_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_X.to(device)), batch_y.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss:.4f}")

        model.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X.to(device))
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
