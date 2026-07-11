"""
Shared helpers for donor-level and cell-level downstream prediction analysis:
donor grouping/padding, metrics, context filtering, and task-label preparation.
"""
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings('ignore')


def str_list_to_unique_index(str_list):
    """
    Convert an iterable of strings to deterministic integer indices without collisions.

    Uses numpy.unique with return_inverse to avoid lossy hashing that could collapse
    different labels/donors sharing characters into the same index.
    """
    if not isinstance(str_list, (list, tuple, np.ndarray)):
        str_list = list(str_list)
    arr = np.asarray(str_list, dtype=str)
    _, inverse = np.unique(arr, return_inverse=True)
    return inverse.astype(np.int64)


def group_and_pad(x, y):
    """
    Group cell embeddings by donor and pad each donor's cells to the same length,
    for PatientAggregator models.

    Args:
        x: Cell embeddings tensor
        y: Donor IDs tensor

    Returns:
        tuple: (padded_x, unique_ids, attention_masks)
    """
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)

    y_sorted, sort_idx = torch.sort(y)
    x_sorted = x[sort_idx]
    unique_ids, counts = torch.unique_consecutive(y_sorted, return_counts=True)
    split_sections = torch.split(x_sorted, counts.tolist())
    padded_x = pad_sequence(split_sections, batch_first=True)

    attn_mask = torch.zeros(padded_x.shape[0], padded_x.shape[1], dtype=torch.bool)
    for i, count in enumerate(counts):
        attn_mask[i, :count] = True

    return padded_x, unique_ids, attn_mask


def calculate_classification_metrics(y_true, y_pred, y_probs):
    """Comprehensive classification metrics (accuracy, F1, precision, recall, AUROC, AUPRC)."""
    present_classes = np.unique(y_true)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    if len(present_classes) == 2:
        try:
            auroc = roc_auc_score(y_true, y_probs[:, 1])
            y_true_onehot = label_binarize(y_true, classes=present_classes)
            auprc = average_precision_score(y_true_onehot, y_probs[:, 1])
        except (ValueError, IndexError) as e:
            print(f"        Debug: AUROC calculation failed: {e}")
            auroc = np.nan
            auprc = np.nan
    elif len(present_classes) > 2:
        try:
            auroc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
            y_true_onehot = label_binarize(y_true, classes=present_classes)
            auprc = average_precision_score(y_true_onehot, y_probs[:, present_classes], average='macro')
        except (ValueError, IndexError):
            auroc = np.nan
            auprc = np.nan
    else:
        # Only one class in test set - AUROC is undefined
        auroc = np.nan
        auprc = np.nan

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'balanced_accuracy': balanced_acc,
        'auroc': auroc,
        'auprc': auprc,
    }


def calculate_regression_metrics(y_true, y_pred):
    """Comprehensive regression metrics (R2, MSE, MAE, Pearson correlation)."""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)

    return {'r2': r2, 'mse': mse, 'mae': mae, 'pearson': pearson_corr}


def filter_data_by_context(cell_data, context_combo):
    """
    Filter cell data to cells whose augmentation context is in `context_combo`.

    Returns:
        tuple: (filtered_cell_data, filtered_embeddings, context_suffix)
    """
    if context_combo == ['ALL']:
        return cell_data, cell_data["embedding"].numpy(), 'all'

    context_array = np.array(cell_data['context'])
    context_mask = np.isin(context_array, context_combo)

    filtered_cell_data = {}
    for key, values in cell_data.items():
        if key == 'embedding':
            filtered_cell_data[key] = cell_data[key][context_mask]
        elif isinstance(values, (list, np.ndarray)) and len(values) == len(context_array):
            if isinstance(values, list):
                filtered_cell_data[key] = [values[i] for i in range(len(values)) if context_mask[i]]
            else:
                filtered_cell_data[key] = values[context_mask]
        else:
            filtered_cell_data[key] = values

    filtered_embeddings = filtered_cell_data["embedding"].numpy()
    context_suffix = '_'.join(context_combo).lower()

    return filtered_cell_data, filtered_embeddings, context_suffix


def prepare_task_data(task, filtered_cell_data, filtered_embeddings, classification_tasks, regression_tasks):
    """
    Prepare (labels, embeddings, task_type) for a task, dropping rows with missing labels.
    Returns None if the task isn't present or has no valid data.
    """
    if task not in filtered_cell_data:
        print(f"⚠️ Task '{task}' not found in data")
        return None

    labels = np.array(filtered_cell_data[task])

    if labels.dtype == 'object':
        valid_mask = labels != None  # noqa: E711
    else:
        valid_mask = ~pd.isna(labels)

    if valid_mask.sum() == 0:
        print(f"⚠️ No valid data for {task}. Skipping.")
        return None

    labels = labels[valid_mask]
    embeddings = filtered_embeddings[valid_mask]

    if task == "mutation":
        if "disease" not in filtered_cell_data:
            print("⚠️ 'mutation' task: 'disease' not found; cannot apply Control->WT relabel.")
        else:
            disease = np.array(filtered_cell_data["disease"])[valid_mask]
            labels = np.array(labels, dtype=object)
            control_mask = (disease == "Control")
            n_relabel = int(control_mask.sum())
            labels[control_mask] = "WT"
            print(f"✅ Activation-aware mutation relabel applied: {n_relabel} Control cells set to 'WT'")

    if task in classification_tasks:
        task_type = 'classification'
    elif task in regression_tasks:
        task_type = 'regression'
        labels = pd.to_numeric(labels, errors='coerce')
        if pd.isna(labels).any():
            print(f"⚠️ Some values in {task} could not be converted to numeric. Skipping.")
            return None
    else:
        print(f"⚠️ Task '{task}' not in predefined lists. Using automatic detection...")
        labels_numeric = pd.to_numeric(labels, errors='coerce')
        if pd.isna(labels_numeric).any():
            task_type = 'classification'
        else:
            task_type = 'regression'
            labels = labels_numeric

    return labels, embeddings, task_type


def create_donor_level_data(labels, embeddings, filtered_cell_data, valid_mask):
    """Group cell-level (labels, embeddings) into per-donor lists for donor-level tasks."""
    if 'donor_id' not in filtered_cell_data:
        print("⚠️ No 'donor_id' found in data. Cannot perform donor-level prediction.")
        return None

    donor_ids = np.array(filtered_cell_data['donor_id'])
    donor_ids_filtered = donor_ids[valid_mask]

    unique_donors = np.unique(donor_ids_filtered)
    donor_embeddings = []
    donor_labels = []
    donor_id_list = []

    for donor_id in unique_donors:
        donor_mask = donor_ids_filtered == donor_id
        donor_emb = embeddings[donor_mask]
        donor_label = labels[donor_mask][0]  # all cells from the same donor share a donor-level label

        if len(donor_emb) > 0:
            donor_embeddings.append(donor_emb)
            donor_labels.append(donor_label)
            donor_id_list.append(donor_id)

    return donor_embeddings, donor_labels, donor_id_list, donor_ids_filtered


def print_task_info(task, task_type, labels, context_combo):
    print(f"\n{'='*60}")
    print(f"TASK: {task} (Context: {context_combo})")
    print(f"{'='*60}")
    print(f"Task type: {task_type}")
    print(f"Number of samples: {len(labels)}")

    if task_type == 'classification':
        print(f"Unique labels: {np.unique(labels)}")
    else:
        print(f"Value range: {np.min(labels):.4f} to {np.max(labels):.4f}")


def create_train_test_split(donor_id_list, donors_split, debug_mode=False):
    """Split donor indices into train/test based on `donors_split['train_donors'/'test_donors']`."""
    train_donor_ids = set(donors_split['train_donors'])
    test_donor_ids = set(donors_split['test_donors'])

    train_indices = []
    test_indices = []

    for i, donor_id in enumerate(donor_id_list):
        if donor_id in train_donor_ids:
            train_indices.append(i)
        elif donor_id in test_donor_ids:
            test_indices.append(i)

    train_indices = torch.tensor(train_indices)
    test_indices = torch.tensor(test_indices)

    if debug_mode:
        if len(train_indices) == 0:
            print("⚠️ DEBUG MODE WARNING: No train donors found in debug subset!")
        if len(test_indices) == 0:
            print("⚠️ DEBUG MODE WARNING: No test donors found in debug subset!")

    print(f"Train donors: {len(train_indices)} (indices: {train_indices.tolist()})")
    print(f"Test donors: {len(test_indices)} (indices: {test_indices.tolist()})")

    return train_indices, test_indices


def print_donor_info(donor_embeddings, donor_id_list, donor_labels, debug_mode=False, donors_split=None):
    """Print donor-level data summary and, if a split is given, per-split label balance."""
    print(f"Found {len(donor_embeddings)} donors with cells")
    print(f"Donor IDs: {donor_id_list}")
    print(f"Donor labels: {donor_labels}")

    try:
        unique_labels, counts = np.unique(np.array(donor_labels), return_counts=True)
        overall_dist = dict(zip(unique_labels.tolist(), counts.tolist()))
        print(f"Overall donor label distribution: {overall_dist}")
    except Exception:
        print("Could not compute overall label distribution")

    if donors_split is not None:
        try:
            train_indices, test_indices = create_train_test_split(donor_id_list, donors_split, debug_mode=debug_mode)

            def _counts_for_indices(indices):
                if len(indices) == 0:
                    return {}
                sel = [donor_labels[i] for i in indices.tolist()]
                u, c = np.unique(np.array(sel), return_counts=True)
                return dict(zip(u.tolist(), c.tolist()))

            print(f"Train split label distribution: {_counts_for_indices(train_indices)}")
            print(f"Test split label distribution: {_counts_for_indices(test_indices)}")
        except Exception as e:
            print(f"⚠️ Could not compute split label distributions: {e}")

    if debug_mode:
        print(f"🔧 DEBUG MODE: Available donors in subset: {donor_id_list}")


# ============================================================================
# HUNTINGTON'S DISEASE CAG-REPEAT CLASSIFICATION (Methods 9.1)
# ============================================================================

def cag_to_class_4_class(cag_value):
    """
    Convert CAG repeat length to a 4-class pathogenicity grouping:
    <=26 Normal (benign); 27-35 Intermediate ("mutable normal");
    36-39 Pathogenic, reduced penetrance; >=40 Pathogenic, full penetrance.
    """
    if pd.isna(cag_value):
        return np.nan
    cag = float(cag_value)
    if cag <= 26:
        return 0
    if cag <= 35:
        return 1
    if cag <= 39:
        return 2
    return 3


def cag_to_class_6_class(cag_value):
    """
    Convert CAG repeat length to a 6-class pathogenicity/onset grouping, splitting
    full-penetrance (>=40) into usual (40-49), earlier (50-59), and juvenile-onset (>=60) ranges.
    """
    if pd.isna(cag_value):
        return np.nan
    cag = float(cag_value)
    if cag <= 26:
        return 0
    if cag <= 35:
        return 1
    if cag <= 39:
        return 2
    if cag <= 49:
        return 3
    if cag <= 59:
        return 4
    return 5
