"""
Trainer for donor-level PatientAggregator-based classification/regression
(Methods 9.8, 9.11): early stopping on held-out donors, then comprehensive metrics.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import f1_score, r2_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from cascade.explainer import config as cfg
from cascade.explainer.utils import (
    calculate_classification_metrics, calculate_regression_metrics,
    create_train_test_split, group_and_pad,
)
from cascade.model.patient_module import AttentionClassifier, AttentionRegressor


def _normalize_donor_id(value):
    if isinstance(value, str):
        v = value.strip()
        try:
            num = float(v)
            return int(num) if num.is_integer() else num
        except ValueError:
            return v
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value) if float(value).is_integer() else float(value)
    return value


@dataclass
class TrainerConfig:
    """Training hyperparameters and model architecture for donor-level training."""
    learning_rate: float = cfg.LEARNING_RATE
    early_stopping_patience: int = cfg.EARLY_STOPPING_PATIENCE
    max_epochs: int = cfg.MAX_EPOCHS
    train_batch_size: int = cfg.TRAIN_BATCH_SIZE
    test_batch_size: int = cfg.TEST_BATCH_SIZE
    weight_decay: float = 0.0
    print_every_n_epochs: int = 10
    attention_heads: Optional[int] = None
    attention_layers: Optional[int] = None
    attention_dropout: Optional[float] = None


class Trainer:
    """Trains and evaluates PatientAggregator-based donor-level models."""

    def __init__(self, device, config: Optional[TrainerConfig] = None):
        self.device = device
        self.config = config if config is not None else TrainerConfig()

    def train_and_evaluate_classification(self, model, train_loader, val_loader, n_epochs=None, is_binary=False):
        if n_epochs is None:
            n_epochs = self.config.max_epochs

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        criterion = torch.nn.BCEWithLogitsLoss() if is_binary else torch.nn.CrossEntropyLoss()

        best_val_f1 = 0
        patience_counter = 0
        best_model_state = None
        best_epoch = 0

        for epoch in range(n_epochs):
            model.train()
            for batch_data in train_loader:
                x, y, mask = batch_data
                x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)

                optimizer.zero_grad()
                outputs = model(x, mask)
                loss = criterion(outputs.squeeze(), y.float()) if is_binary else criterion(outputs, y)
                loss.backward()
                optimizer.step()

            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for batch_data in val_loader:
                    x, y, mask = batch_data
                    x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
                    outputs = model(x, mask)
                    if is_binary:
                        preds = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
                    else:
                        preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(y.cpu().numpy())

            f1 = f1_score(np.array(val_labels), np.array(val_preds), average='weighted')

            if f1 > best_val_f1:
                best_val_f1 = f1
                best_epoch = epoch + 1
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"    Early stopping at epoch {epoch+1}. Best epoch was {best_epoch} with F1={best_val_f1:.4f}")
                    break

            if epoch % self.config.print_every_n_epochs == 0:
                print(f"    Epoch {epoch+1}: F1={f1:.4f}")

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"    Restored model from best epoch {best_epoch}")

        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch_data in val_loader:
                x, y, mask = batch_data
                x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
                outputs = model(x, mask)
                if is_binary:
                    probs = torch.sigmoid(outputs.squeeze())
                    preds = (probs > 0.5).long()
                else:
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        metrics = calculate_classification_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
        metrics['best_epoch'] = best_epoch
        metrics['best_val_f1'] = best_val_f1
        return metrics

    def train_and_evaluate_regression(self, model, train_loader, val_loader, n_epochs=None, label_scaler=None):
        if n_epochs is None:
            n_epochs = self.config.max_epochs

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        criterion = torch.nn.MSELoss()

        best_val_r2 = -float('inf')
        patience_counter = 0
        best_model_state = None
        best_epoch = 0

        for epoch in range(n_epochs):
            model.train()
            for batch_data in train_loader:
                x, y, mask = batch_data
                x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)

                optimizer.zero_grad()
                outputs = model(x, mask)
                loss = criterion(outputs, y.float())
                loss.backward()
                optimizer.step()

            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for batch_data in val_loader:
                    x, y, mask = batch_data
                    x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
                    outputs = model(x, mask)
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(y.cpu().numpy())

            val_preds, val_labels = np.array(val_preds), np.array(val_labels)
            if label_scaler is not None:
                val_preds_o = label_scaler.inverse_transform(val_preds.reshape(-1, 1)).flatten()
                val_labels_o = label_scaler.inverse_transform(val_labels.reshape(-1, 1)).flatten()
            else:
                val_preds_o, val_labels_o = val_preds, val_labels

            r2 = r2_score(val_labels_o, val_preds_o)

            if r2 > best_val_r2:
                best_val_r2 = r2
                best_epoch = epoch + 1
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"    Early stopping at epoch {epoch+1}. Best epoch was {best_epoch} with R2={best_val_r2:.4f}")
                    break

            if epoch % self.config.print_every_n_epochs == 0:
                print(f"    Epoch {epoch+1}: R2={r2:.4f}")

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"    Restored model from best epoch {best_epoch}")

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                x, y, mask = batch_data
                x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
                outputs = model(x, mask)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        all_preds, all_labels = np.array(all_preds), np.array(all_labels)
        if label_scaler is not None:
            all_preds = label_scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
            all_labels = label_scaler.inverse_transform(all_labels.reshape(-1, 1)).flatten()

        metrics = calculate_regression_metrics(all_labels, all_preds)
        metrics['best_epoch'] = best_epoch
        metrics['best_val_r2'] = best_val_r2
        return metrics

    def create_model(self, task_type, input_dim, n_classes=None):
        num_heads = self.config.attention_heads if self.config.attention_heads is not None else cfg.ATTENTION_HEADS
        num_layers = self.config.attention_layers if self.config.attention_layers is not None else cfg.ATTENTION_LAYERS
        dropout = self.config.attention_dropout if self.config.attention_dropout is not None else cfg.ATTENTION_DROPOUT

        if task_type == 'classification':
            model = AttentionClassifier(input_dim=input_dim, num_classes=n_classes, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
        else:
            model = AttentionRegressor(input_dim=input_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout)

        print(f"    Model architecture: {num_layers} layers, {num_heads} heads, dropout={dropout}")
        return model.to(self.device)

    def prepare_donor_level_data(self, donor_embeddings, donor_labels, donor_id_list, donors_split, debug_mode=False):
        donor_embeddings = [torch.tensor(emb, dtype=torch.float32) for emb in donor_embeddings]
        donor_labels_array = np.array(donor_labels)
        donor_id_list = [_normalize_donor_id(d) for d in donor_id_list]

        donor_id_to_index = {donor_id: idx for idx, donor_id in enumerate(donor_id_list)}
        all_donor_embeddings = torch.cat(donor_embeddings, dim=0)
        donor_ids_for_grouping = []
        for i, donor_emb in enumerate(donor_embeddings):
            donor_idx = donor_id_to_index[donor_id_list[i]]
            donor_ids_for_grouping.extend([donor_idx] * len(donor_emb))
        donor_ids_tensor = torch.tensor(donor_ids_for_grouping, dtype=torch.long)

        x_grouped, unique_donor_ids, attn_masks = group_and_pad(all_donor_embeddings, donor_ids_tensor)

        donor_id_to_label = {donor_id_to_index[d]: label for d, label in zip(donor_id_list, donor_labels_array)}
        y_grouped = torch.tensor([donor_id_to_label[int(d.item())] for d in unique_donor_ids])

        print(f"Grouped data shape: {x_grouped.shape}")
        print(f"Labels shape: {y_grouped.shape}")
        print(f"Attention masks shape: {attn_masks.shape}")

        train_indices, test_indices = create_train_test_split(donor_id_list, donors_split, debug_mode)

        x_train, y_train, train_mask = x_grouped[train_indices].to(self.device), y_grouped[train_indices].to(self.device), attn_masks[train_indices].to(self.device)
        x_test, y_test, test_mask = x_grouped[test_indices].to(self.device), y_grouped[test_indices].to(self.device), attn_masks[test_indices].to(self.device)

        train_loader = DataLoader(TensorDataset(x_train, y_train, train_mask), batch_size=self.config.train_batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(x_test, y_test, test_mask), batch_size=self.config.test_batch_size, shuffle=False)

        return train_loader, test_loader, x_grouped, y_grouped, attn_masks

    def train_donor_level_task(self, donor_embeddings, donor_labels, donor_id_list, donors_split,
                                task_type, debug_mode=False, label_scaler=None):
        donor_labels_array = np.array(donor_labels)
        label_encoder = None

        if task_type == 'classification':
            if not np.issubdtype(donor_labels_array.dtype, np.number):
                label_encoder = LabelEncoder()
                donor_labels_encoded = label_encoder.fit_transform(donor_labels_array)
            else:
                donor_labels_encoded = donor_labels_array
        else:
            donor_labels_encoded = donor_labels_array

        train_loader, test_loader, x_grouped, y_grouped, attn_masks = self.prepare_donor_level_data(
            donor_embeddings, donor_labels_encoded, donor_id_list, donors_split, debug_mode
        )

        if task_type == 'classification':
            n_classes = len(np.unique(donor_labels_encoded))
            model = self.create_model(task_type, x_grouped.shape[2], n_classes)
        else:
            model = self.create_model(task_type, x_grouped.shape[2])

        if task_type == 'classification':
            metrics = self.train_and_evaluate_classification(model, train_loader, test_loader, n_epochs=self.config.max_epochs, is_binary=False)
            if label_encoder is not None:
                class_labels = label_encoder.classes_.tolist()
                label_mapping = {int(i): str(lbl) for i, lbl in enumerate(class_labels)}
                metrics['class_labels'] = class_labels
                metrics['label_mapping'] = label_mapping
                metrics['label_metadata'] = {'class_labels': class_labels, 'label_mapping': label_mapping}
        else:
            metrics = self.train_and_evaluate_regression(model, train_loader, test_loader, n_epochs=self.config.max_epochs, label_scaler=label_scaler)

        return metrics, model
