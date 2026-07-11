"""
Cell-level MLP probe: maps a single cell's embedding directly to a
classification or regression target, without donor-level aggregation
(Methods 9.11).
"""
import copy
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, r2_score
from torch.utils.data import DataLoader, TensorDataset

from cascade.explainer.config import EARLY_STOPPING_PATIENCE, LEARNING_RATE, TEST_BATCH_SIZE, TRAIN_BATCH_SIZE
from cascade.explainer.utils import calculate_classification_metrics, calculate_regression_metrics


class CellLevelMLP(nn.Module):
    """MLP mapping a single cell embedding to a classification or regression target."""

    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128], dropout=0.3, task_type='classification'):
        super().__init__()
        self.task_type = task_type

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_saved_cell_level_mlp(checkpoint_path: Union[Path, str], input_dim: int, task_type: str,
                               n_classes: Optional[int], device: torch.device):
    """
    Load a cell-level MLP saved by `cascade.explainer.save_trained_models.save_trained_model`.
    For classification, the output dimension is taken from the checkpoint weights
    (handles binary softmax K=2 vs single-logit heads).
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if task_type == "classification":
        out_dim = None
        for _k, v in reversed(list(state_dict.items())):
            if hasattr(v, "ndim") and v.ndim == 2:
                out_dim = int(v.shape[0])
                break
        if out_dim is None:
            raise ValueError(f"Could not infer classification output dim from {checkpoint_path}")
        model = CellLevelMLP(input_dim, out_dim, task_type="classification").to(device)
    else:
        model = CellLevelMLP(input_dim, 1, task_type="regression").to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def _last_linear_out_features(model):
    for m in reversed(model.network):
        if isinstance(m, nn.Linear):
            return m.out_features
    return None


class CellLevelTrainer:
    """Trainer/evaluator for cell-level MLP classification and regression tasks."""

    def __init__(self, device):
        self.device = device

    def _build_train_test_masks(self, donor_ids, donors_split):
        train_donor_ids = set(donors_split["train_donors"])
        test_donor_ids = set(donors_split["test_donors"])

        unique_donor_ids_in_data = np.unique(donor_ids)
        try:
            if not isinstance(unique_donor_ids_in_data[0], (int, np.integer)):
                donor_ids_normalized = np.array([int(float(d)) for d in donor_ids])
            else:
                donor_ids_normalized = donor_ids
            train_donor_ids = set(int(d) if not isinstance(d, int) else d for d in train_donor_ids)
            test_donor_ids = set(int(d) if not isinstance(d, int) else d for d in test_donor_ids)
        except (ValueError, TypeError) as e:
            print(f"  ⚠️ Could not convert donor IDs to int: {e}, using original values")
            donor_ids_normalized = donor_ids

        train_mask = np.isin(donor_ids_normalized, list(train_donor_ids))
        test_mask = np.isin(donor_ids_normalized, list(test_donor_ids))

        print(f"  Train mask: {train_mask.sum()} True values out of {len(train_mask)}")
        print(f"  Test mask: {test_mask.sum()} True values out of {len(test_mask)}")
        return train_mask, test_mask, train_donor_ids, test_donor_ids

    def create_cell_level_split(self, embeddings, labels, donor_ids, donors_split, debug_mode=False, downsample_majority=False):
        """
        Create train/test dataloaders split by donor ID (no cell-level leakage between splits).

        downsample_majority: if True, undersample majority classes in the train
            set so each class has at most as many samples as the minority class.
        """
        train_mask, test_mask, train_donor_ids, test_donor_ids = self._build_train_test_masks(donor_ids, donors_split)

        X_train, y_train = embeddings[train_mask], labels[train_mask]
        X_test, y_test = embeddings[test_mask], labels[test_mask]

        print(f"Train set: {len(X_train)} cells from {len(train_donor_ids)} donors")
        print(f"Test set: {len(X_test)} cells from {len(test_donor_ids)} donors")

        if debug_mode:
            debug_train_size = min(1000, len(X_train))
            debug_test_size = min(500, len(X_test))
            X_train, y_train = X_train[:debug_train_size], y_train[:debug_train_size]
            X_test, y_test = X_test[:debug_test_size], y_test[:debug_test_size]
            print(f"🔧 DEBUG MODE: Limited to {len(X_train)} train, {len(X_test)} test cells")

        if downsample_majority:
            y_train_np = np.asarray(y_train)
            y_cls = np.round(y_train_np).astype(int) if y_train_np.dtype.kind in ('f', 'c') else y_train_np.astype(int)
            classes, counts = np.unique(y_cls, return_counts=True)
            if len(classes) < 2:
                print("  ⚠️ Downsampling skipped: only one class in train set", flush=True)
            else:
                target = int(counts.min())
                rng = np.random.RandomState(42)
                kept_idx = []
                for c in classes:
                    idx_c = np.where(y_cls == c)[0]
                    kept_idx.append(rng.choice(idx_c, size=target, replace=False) if len(idx_c) > target else idx_c)
                selected = np.sort(np.concatenate(kept_idx))
                X_train = np.asarray(X_train)[selected]
                y_train = np.asarray(y_train)[selected]
                print(f"  Downsampled train set to {len(y_train)} samples (target per class: {target})", flush=True)

        unique_train, counts_train = np.unique(np.asarray(y_train), return_counts=True)
        unique_test, counts_test = np.unique(np.asarray(y_test), return_counts=True)
        print(f"  Train set - counts per class: {dict(zip(unique_train.tolist(), counts_train.tolist()))}")
        print(f"  Test set  - counts per class: {dict(zip(unique_test.tolist(), counts_test.tolist()))}")

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test))

        # drop_last on train avoids batch-norm failures when the final batch has a single example
        train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

        return train_loader, test_loader

    def _safe_classification_metrics(self, y_true, y_pred, y_prob, split_name):
        try:
            m = calculate_classification_metrics(y_true, y_pred, y_prob)
        except Exception as e:
            print(f"    ⚠️ {split_name}: metric computation warning: {e}", flush=True)
            m = {}
        if not m.get("f1"):
            try:
                m["f1"] = float(f1_score(y_true, y_pred, average="weighted"))
            except Exception as e:
                print(f"    ⚠️ {split_name}: could not compute fallback F1: {e}", flush=True)
                m["f1"] = float("nan")
        if not m.get("auroc"):
            m["auroc"] = float("nan")
        return m

    def _predict_loader(self, model, loader, is_binary_logits):
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                outputs = model(X_batch)
                if is_binary_logits:
                    probs = torch.sigmoid(outputs.squeeze())
                    probs_2class = torch.stack([1 - probs, probs], dim=1)
                    preds = (probs > 0.5).long()
                    all_probs.extend(probs_2class.cpu().numpy())
                else:
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        return np.array(all_preds), np.array(all_labels), (np.array(all_probs) if all_probs else None)

    def evaluate_saved_cell_level_model(self, model, embeddings, labels, donor_ids, donors_split, task_type,
                                         debug_mode=False, skip_train_metrics=False):
        """Run a saved model on the fixed train/test donor split (metrics computed on the test set)."""
        train_mask, test_mask, train_donor_ids, test_donor_ids = self._build_train_test_masks(donor_ids, donors_split)

        X_train, y_train = embeddings[train_mask], labels[train_mask]
        X_test, y_test = embeddings[test_mask], labels[test_mask]
        print(f"Eval-only Train set: {len(X_train)} cells from {len(train_donor_ids)} donors")
        print(f"Eval-only Test set: {len(X_test)} cells from {len(test_donor_ids)} donors")

        if debug_mode:
            debug_train_size, debug_test_size = min(1000, len(X_train)), min(500, len(X_test))
            X_train, y_train = X_train[:debug_train_size], y_train[:debug_train_size]
            X_test, y_test = X_test[:debug_test_size], y_test[:debug_test_size]
            print(f"🔧 DEBUG MODE: Limited eval-only to {len(X_train)} train, {len(X_test)} test cells")

        train_loader = DataLoader(TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(y_train)), batch_size=TEST_BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(TensorDataset(torch.as_tensor(X_test, dtype=torch.float32), torch.as_tensor(y_test)), batch_size=TEST_BATCH_SIZE, shuffle=False)

        model.eval()
        is_binary_logits = _last_linear_out_features(model) == 1

        all_preds, all_labels, all_probs = self._predict_loader(model, test_loader, is_binary_logits)
        metrics = (
            self._safe_classification_metrics(all_labels, all_preds, all_probs, "Test set")
            if task_type == "classification"
            else calculate_regression_metrics(all_labels, all_preds)
        )

        if not skip_train_metrics:
            train_preds, train_labels, train_probs = self._predict_loader(model, train_loader, is_binary_logits)
            if task_type == "classification":
                train_metrics = self._safe_classification_metrics(train_labels, train_preds, train_probs, "Train set")
                print(f"    Train set (full): F1={train_metrics.get('f1', 0):.4f}, AUROC={train_metrics.get('auroc', 0):.4f}", flush=True)
                print(f"    Test set:        F1={metrics.get('f1', 0):.4f}, AUROC={metrics.get('auroc', 0):.4f}", flush=True)
            else:
                train_metrics = calculate_regression_metrics(train_labels, train_preds)
                print(f"    Train set (full): R2={train_metrics.get('r2', 0):.4f}, Pearson={train_metrics.get('pearson', 0):.4f}", flush=True)
                print(f"    Test set:        R2={metrics.get('r2', 0):.4f}, Pearson={metrics.get('pearson', 0):.4f}", flush=True)
        else:
            print("    Train set metrics: skipped (--skip-train-metrics)", flush=True)

        return metrics, all_preds, all_labels, all_probs

    def predict_all_labeled_cells(self, model, embeddings, labels, task_type, debug_mode=False):
        """Run inference on every labeled row, with no train/test split."""
        X, y = np.asarray(embeddings), np.asarray(labels)
        if debug_mode:
            n_cap = min(2000, len(X))
            X, y = X[:n_cap], y[:n_cap]
            print(f"🔧 DEBUG MODE: full-data predict limited to {n_cap} cells", flush=True)

        loader = DataLoader(TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y)), batch_size=TEST_BATCH_SIZE, shuffle=False)

        model.eval()
        is_binary_logits = _last_linear_out_features(model) == 1
        all_preds, all_labels, all_probs = self._predict_loader(model, loader, is_binary_logits)

        if task_type == "classification":
            metrics = self._safe_classification_metrics(all_labels, all_preds, all_probs, "Full labeled set")
            if all_probs is not None and all_probs.ndim >= 2:
                try:
                    ncf = all_probs.shape[1]
                    if ncf == 2:
                        metrics["auprc"] = float(average_precision_score(all_labels, all_probs[:, 1]))
                    else:
                        metrics["auprc"] = float(average_precision_score(all_labels, all_probs, average="weighted"))
                except Exception as e:
                    print(f"    ⚠️ Full labeled set: could not compute AUPRC: {e}", flush=True)
                    metrics["auprc"] = float("nan")
        else:
            metrics = calculate_regression_metrics(all_labels, all_preds)

        return metrics, all_preds, all_labels, all_probs

    def train_classification_model(self, model, train_loader, test_loader, n_epochs):
        """Train a cell-level classification MLP, weighting the loss for class imbalance.

        NOTE: model selection (early stopping / best checkpoint) uses F1 on `test_loader`
        itself, i.e. there is no held-out validation set distinct from the reported test set.
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        is_binary_logits = _last_linear_out_features(model) == 1

        train_labels = train_loader.dataset.tensors[1].numpy()
        n_samples = len(train_labels)
        n_classes = int(train_labels.max()) + 1 if not is_binary_logits else 2
        if is_binary_logits:
            count_pos = (train_labels >= 0.5).sum()
            count_neg = n_samples - count_pos
            if count_pos > 0 and count_neg > 0:
                pos_weight = torch.tensor([count_neg / count_pos], dtype=torch.float32, device=self.device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                print(f"    Class weights (binary): pos_weight={pos_weight.item():.4f}", flush=True)
            else:
                criterion = nn.BCEWithLogitsLoss()
        else:
            counts = np.bincount(train_labels.astype(int), minlength=n_classes)
            if np.all(counts > 0):
                weights = n_samples / (n_classes * counts)
                weight_t = torch.tensor(weights, dtype=torch.float32, device=self.device)
                criterion = nn.CrossEntropyLoss(weight=weight_t)
                print(f"    Class weights (multiclass): {dict(zip(range(n_classes), np.round(weights, 4).tolist()))}", flush=True)
            else:
                criterion = nn.CrossEntropyLoss()

        best_val_f1 = -float("inf")
        best_model_state = None
        patience_counter = 0

        for epoch in range(n_epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch.float()) if is_binary_logits else criterion(outputs, y_batch.long())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            val_preds, val_labels, _ = self._predict_loader(model, test_loader, is_binary_logits)
            f1 = f1_score(val_labels, val_preds, average='weighted')

            if f1 > best_val_f1:
                best_val_f1 = f1
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"    Early stopping at epoch {epoch+1}", flush=True)
                    break

            if epoch % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{n_epochs}: Loss={train_loss/len(train_loader):.4f}, F1={f1:.4f} (best: {best_val_f1:.4f})", flush=True)

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"    ✅ Restored best checkpoint (best F1={best_val_f1:.4f})", flush=True)

        all_preds, all_labels, all_probs = self._predict_loader(model, test_loader, is_binary_logits)
        metrics = calculate_classification_metrics(all_labels, all_preds, all_probs)
        n_classes_final = all_probs.shape[1]
        metrics["auprc"] = float(
            average_precision_score(all_labels, all_probs[:, 1]) if n_classes_final == 2
            else average_precision_score(all_labels, all_probs, average='weighted')
        )
        metrics["best_selection_f1"] = float(best_val_f1)

        train_preds, train_labels, train_probs = self._predict_loader(model, train_loader, is_binary_logits)
        train_metrics = calculate_classification_metrics(train_labels, train_preds, train_probs)
        train_auprc = (
            average_precision_score(train_labels, train_probs[:, 1]) if n_classes_final == 2
            else average_precision_score(train_labels, train_probs, average='weighted')
        )
        print(f"    Train set: F1={train_metrics.get('f1', 0):.4f}, AUROC={train_metrics.get('auroc', 0):.4f}, AUPRC={train_auprc:.4f}", flush=True)
        print(f"    Test set:  F1={metrics.get('f1', 0):.4f}, AUROC={metrics.get('auroc', 0):.4f}, AUPRC={metrics.get('auprc', 0):.4f}", flush=True)

        return metrics, all_preds, all_labels, all_probs

    def train_regression_model(self, model, train_loader, test_loader, n_epochs):
        """Train a cell-level regression MLP.

        NOTE: model selection (early stopping / best checkpoint) uses R2 on `test_loader`
        itself, i.e. there is no held-out validation set distinct from the reported test set.
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        best_val_r2 = -float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(n_epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch.float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    outputs = model(X_batch.to(self.device)).squeeze()
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(y_batch.cpu().numpy())
            r2 = r2_score(np.array(val_labels), np.array(val_preds))

            if r2 > best_val_r2:
                best_val_r2 = r2
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"    Early stopping at epoch {epoch+1}", flush=True)
                    break

            if epoch % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{n_epochs}: Loss={train_loss/len(train_loader):.4f}, R2={r2:.4f} (best: {best_val_r2:.4f})", flush=True)

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"    ✅ Restored best checkpoint (best R2={best_val_r2:.4f})", flush=True)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch.to(self.device)).squeeze()
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        all_preds, all_labels = np.array(all_preds), np.array(all_labels)

        metrics = calculate_regression_metrics(all_labels, all_preds)
        metrics["best_selection_r2"] = float(best_val_r2)

        train_preds, train_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in train_loader:
                outputs = model(X_batch.to(self.device)).squeeze()
                train_preds.extend(outputs.cpu().numpy())
                train_labels.extend(y_batch.cpu().numpy())
        train_metrics = calculate_regression_metrics(np.array(train_labels), np.array(train_preds))
        print(f"    Train set: R2={train_metrics.get('r2', 0):.4f}, Pearson={train_metrics.get('pearson', 0):.4f}", flush=True)
        print(f"    Test set:  R2={metrics.get('r2', 0):.4f}, Pearson={metrics.get('pearson', 0):.4f}", flush=True)

        return metrics, all_preds, all_labels

    def train_cell_level_task(self, embeddings, labels, donor_ids, donors_split, task_type, input_dim,
                               n_classes=None, debug_mode=False, downsample_majority=False, n_epochs=None):
        from cascade.explainer.config import MAX_EPOCHS
        n_epochs = n_epochs or MAX_EPOCHS

        train_loader, test_loader = self.create_cell_level_split(
            embeddings, labels, donor_ids, donors_split, debug_mode,
            downsample_majority=(downsample_majority and task_type == 'classification'),
        )

        if task_type == 'classification':
            model = CellLevelMLP(input_dim, n_classes, task_type='classification').to(self.device)
            metrics, test_preds, test_labels, test_probs = self.train_classification_model(model, train_loader, test_loader, n_epochs)
        else:
            model = CellLevelMLP(input_dim, 1, task_type='regression').to(self.device)
            metrics, test_preds, test_labels = self.train_regression_model(model, train_loader, test_loader, n_epochs)
            test_probs = None

        return metrics, model, test_preds, test_labels, test_probs
