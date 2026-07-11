#!/usr/bin/env python3
"""
Attention-based cell explainer (Methods 9.9, first half): extracts per-layer
attention weights from a trained PatientAggregator donor-level model (the [PAT]
token attending to each of a donor's cells) and saves them alongside donor-level
predictions and metadata for downstream per-cell importance scoring
(m^(q)_i(j) = max over layers of attention from [PAT] to cell j).
"""
import argparse
import gc
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import trange

from cascade.data import splits as donor_splits
from cascade.explainer.chunked_io import load_chunked_embeddings, load_embeddings_from_subset
from cascade.explainer.utils import cag_to_class_4_class, cag_to_class_6_class, group_and_pad, str_list_to_unique_index
from cascade.explainer.config import (
    CLASSIFICATION_TASKS, HH_ATTENTION_DROPOUT, HH_ATTENTION_HEADS,
    HH_ATTENTION_LAYERS, HH_DONOR_LEVEL_TASKS, REGRESSION_TASKS,
)

# Datasets whose donor-level models were trained on a single fixed context (no
# per-context-combination sweep), matching main_donor_level.py's valid_combinations.
NO_CONTEXT_SWEEP_DATASETS = {'HH', 'SEATTLE'}
from cascade.explainer.main_donor_level import SEATTLE_DONOR_LEVEL_TASKS
from cascade.model.patient_module import ATTENTION_DROPOUT, ATTENTION_HEADS, ATTENTION_LAYERS, AttentionClassifier, AttentionRegressor

TRAINED_MODELS_DIR = Path("./trained_models")
ATTENTION_OUTPUT_DIR = Path("./attention_analysis_results")
DEBUG_MODE = False

CONTEXT_COMBINATIONS = [
    ['ALL'], ['CELLS'], ['TISSUE'], ['DISEASE'], ['CELLS', 'TISSUE'], ['CELLS', 'TISSUE', 'DISEASE'],
]


def get_chunk_files_from_directory(embedding_dir):
    embedding_path = Path(embedding_dir)
    if not embedding_path.exists():
        raise ValueError(f"Embedding directory does not exist: {embedding_dir}")
    chunk_files = sorted(embedding_path.glob("*.pkl"))
    if not chunk_files:
        raise ValueError(f"No .pkl files found in directory: {embedding_dir}")
    print(f"📁 Found {len(chunk_files)} chunk files in {embedding_dir}")
    return [str(f) for f in chunk_files]


def load_and_merge_embeddings(chunk_paths):
    """Load embeddings from multiple chunk files and merge them (simple concatenation,
    for the moderate number of chunks produced by a single-node extraction run)."""
    from collections import defaultdict
    merged_data = defaultdict(list)
    total_cells = 0

    for i, chunk_path in enumerate(chunk_paths, 1):
        with open(chunk_path, 'rb') as f:
            chunk_data = pickle.load(f)
        if "cell_data" in chunk_data:
            chunk_data = chunk_data["cell_data"]
        if 'embedding' not in chunk_data:
            print(f"  ⚠️ No 'embedding' key in chunk {i}, skipping")
            continue
        total_cells += len(chunk_data['embedding'])

        for key, value in chunk_data.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                merged_data[key].append(value)
            elif isinstance(value, list):
                merged_data[key].extend(value)
            else:
                merged_data[key] = value

    cell_data = {}
    for key, value_list in merged_data.items():
        if isinstance(value_list, list) and len(value_list) > 0:
            if isinstance(value_list[0], np.ndarray):
                cell_data[key] = np.concatenate(value_list, axis=0)
            elif isinstance(value_list[0], torch.Tensor):
                cell_data[key] = torch.cat(value_list, dim=0)
            else:
                cell_data[key] = value_list
        else:
            cell_data[key] = value_list

    print(f"✅ Merged {total_cells} cells from {len(chunk_paths)} chunks")
    return cell_data


def get_dataset_config(dataset_name):
    """Tasks and default context combinations to analyze attention for, per dataset."""
    if dataset_name.upper() == 'SEATTLE':
        return SEATTLE_DONOR_LEVEL_TASKS, [['ALL']]
    if dataset_name.upper() == 'HH':
        return HH_DONOR_LEVEL_TASKS, [['ALL']]
    if dataset_name.upper() == 'AUTISM':
        return ['disease'], CONTEXT_COMBINATIONS
    print(f"⚠️ Unknown dataset '{dataset_name}', using default configuration")
    return ['disease'], [['ALL']]


def load_context_aware_model(task, task_type, input_dim, num_classes=None, context_suffix=None,
                              model_dir=TRAINED_MODELS_DIR, device=None, dataset_name=None, mode=None,
                              num_heads=None, num_layers=None, dropout=None):
    """
    Load a trained donor-level attention model from disk.

    Returns:
        (model, label_scaler, label_metadata)
    """
    model_dir = Path(model_dir)

    context_task_name = f"{task}_{mode}_context_{context_suffix}" if mode is not None else f"{task}_context_{context_suffix}"
    if context_suffix and context_suffix != 'all' and (not dataset_name or dataset_name.upper() != 'HH'):
        filename = f"{context_task_name}_attention_attention_{task_type}_context_{context_suffix}.pt"
    elif mode is not None:
        filename = f"{task}_{mode}_attention_attention_{task_type}_context_all.pt"
    else:
        filename = f"{task}_attention_attention_{task_type}_context_all.pt"

    filepath = model_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Trained model not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)
    additional_info = checkpoint.get('additional_info', {}) or {}

    label_scaler = None
    if task_type == 'regression' and 'label_scaler_mean' in additional_info:
        label_scaler = StandardScaler()
        label_scaler.mean_ = np.array([additional_info['label_scaler_mean']])
        label_scaler.scale_ = np.array([additional_info['label_scaler_scale']])
        label_scaler.n_features_in_ = 1

    label_metadata = additional_info.get('label_metadata')
    if task_type == 'classification' and num_classes is None:
        if 'n_classes' in additional_info:
            num_classes = int(additional_info['n_classes'])
        else:
            state_dict = checkpoint.get('model_state_dict', {})
            for k, v in reversed(list(state_dict.items())):
                if hasattr(v, 'ndim') and v.ndim == 2:
                    num_classes = int(v.shape[0])
                    break
            if num_classes is None:
                raise ValueError(f"Could not infer num_classes for task {task}")

    if dataset_name and dataset_name.upper() == 'HH':
        num_heads = num_heads if num_heads is not None else HH_ATTENTION_HEADS
        num_layers = num_layers if num_layers is not None else HH_ATTENTION_LAYERS
        dropout = dropout if dropout is not None else HH_ATTENTION_DROPOUT
    else:
        num_heads = num_heads if num_heads is not None else ATTENTION_HEADS
        num_layers = num_layers if num_layers is not None else ATTENTION_LAYERS
        dropout = dropout if dropout is not None else ATTENTION_DROPOUT

    print(f"  Loading model with architecture: {num_layers} layers, {num_heads} heads, dropout={dropout}")

    if task_type == 'classification':
        model = AttentionClassifier(input_dim, num_classes, num_heads=num_heads, num_layers=num_layers, dropout=dropout).to(device)
    else:
        model = AttentionRegressor(input_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, label_scaler, label_metadata


def get_attention_weights_context_aware(model, filtered_cell_data, device, context_key='disease',
                                         task_type='classification', label_scaler=None):
    """Run the trained model over donor-grouped cells and collect per-layer attention
    weights from [PAT] to each cell, plus donor-level predictions and metadata."""
    latent_vectors = filtered_cell_data['embedding']
    if isinstance(latent_vectors, np.ndarray):
        latent_vectors = torch.from_numpy(latent_vectors)
    latent_vectors = latent_vectors.float()

    y0 = np.array(filtered_cell_data[context_key])
    y = torch.tensor(str_list_to_unique_index(y0), dtype=torch.long) if isinstance(y0[0], str) else torch.tensor(y0, dtype=torch.long)

    donor_vals_raw = np.array(filtered_cell_data['donor_id'])
    donor_val_lookup = None
    if isinstance(donor_vals_raw[0], str):
        donor_indices = np.array(str_list_to_unique_index(donor_vals_raw), dtype=np.int64)
        donor_val_lookup = {}
        for idx, donor_label in zip(donor_indices, donor_vals_raw):
            donor_val_lookup.setdefault(idx, donor_label)
    else:
        donor_indices = donor_vals_raw.astype(int)

    ind = torch.tensor(donor_indices, dtype=torch.long)
    padded_x, patient_ids, attn_mask = group_and_pad(latent_vectors, ind)

    donor_vals = donor_indices.copy()
    donor_to_labels = {}
    for d, lab in zip(donor_vals, y0):
        donor_to_labels.setdefault(d, []).append(lab)

    patient_y = []
    for pid in patient_ids:
        labels = donor_to_labels.get(pid.item(), [])
        if not labels:
            patient_y.append(np.nan)
        elif isinstance(labels[0], str):
            vals, counts = np.unique(labels, return_counts=True)
            patient_y.append(vals[np.argmax(counts)])
        else:
            try:
                patient_y.append(float(np.nanmean(np.array(labels, dtype=float))))
            except Exception:
                patient_y.append(np.nan)
    patient_y = np.array(patient_y, dtype=object)

    patient_donor_labels = [
        (donor_val_lookup.get(pid.item(), pid.item()) if donor_val_lookup is not None else pid.item())
        for pid in patient_ids
    ]

    num_cells = donor_vals.shape[0]

    def _to_array(values):
        if isinstance(values, torch.Tensor):
            return values.cpu().numpy()
        if isinstance(values, np.ndarray):
            return values
        if isinstance(values, list):
            return np.array(values, dtype=object)
        return None

    patient_metadata = {}
    for key, values in filtered_cell_data.items():
        if key == 'embedding':
            continue
        values_array = _to_array(values)
        if values_array is None or len(values_array) != num_cells:
            continue
        donor_to_values = {}
        for d, val in zip(donor_vals, values_array):
            donor_to_values.setdefault(d, []).append(val)
        patient_metadata[key] = [donor_to_values.get(pid.item(), []) for pid in patient_ids]

    patient_cell_types = patient_metadata.get('cell_type', [[] for _ in range(len(patient_ids))])
    patient_gRNAs = patient_metadata.get('gRNA', [[] for _ in range(len(patient_ids))])

    batch_size = 16
    attention_weights_all_layers = []
    predictions = []

    with torch.no_grad():
        for i in trange(0, len(padded_x), batch_size):
            batch_x = padded_x[i:i + batch_size].to(device)
            batch_mask = attn_mask[i:i + batch_size].to(device)
            outputs, batch_attention = model(batch_x, batch_mask, return_attention=True)

            if i == 0:
                attention_weights_all_layers = [[] for _ in range(len(batch_attention))]
            for layer_idx, layer_attention in enumerate(batch_attention):
                attention_weights_all_layers[layer_idx].append(layer_attention.detach().clone().cpu())

            if task_type == 'classification':
                predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            else:
                preds = outputs.cpu().numpy()
                if label_scaler is not None:
                    preds = label_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
                predictions.extend(preds)

    if attention_weights_all_layers and attention_weights_all_layers[0]:
        # CrossAttentionBlock uses nn.MultiheadAttention's default head-averaged weights,
        # so each layer's tensor is (n_patients, query_len=1, key_len=n_cells); stacking
        # across layers gives (num_layers, n_patients, query_len, key_len).
        attention_weights = np.stack(
            [torch.cat(layer_batches, dim=0).numpy() for layer_batches in attention_weights_all_layers], axis=0
        )
    else:
        attention_weights = np.array([])

    return attention_weights, predictions, patient_ids, patient_donor_labels, patient_y, patient_cell_types, patient_gRNAs, patient_metadata


def main(models_dir=None, data_path=None, output_dir=None, dataset_name=None,
         embedding_chunks=None, embedding_dir=None, use_subset=False):
    print("=" * 80)
    print("CONTEXT-AWARE ATTENTION ANALYSIS")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")

    global TRAINED_MODELS_DIR, ATTENTION_OUTPUT_DIR
    if models_dir is not None:
        TRAINED_MODELS_DIR = Path(models_dir)
    if output_dir is not None:
        ATTENTION_OUTPUT_DIR = Path(output_dir)

    if use_subset:
        if embedding_dir is None:
            raise ValueError("--use-subset requires --embedding-dir pointing to the subset directory")
        donors_split = donor_splits.SPLITS_BY_DATASET.get(dataset_name.upper())
        cell_data = load_embeddings_from_subset(embedding_dir, donors_split=donors_split)
    elif embedding_dir is not None:
        chunk_paths = get_chunk_files_from_directory(embedding_dir)
        cell_data = load_and_merge_embeddings(chunk_paths)
    elif embedding_chunks:
        cell_data = load_and_merge_embeddings(embedding_chunks)
    else:
        if data_path is None:
            raise ValueError("Provide --data-path, --embedding-dir, or --embedding-chunks")
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        cell_data = data.get("cell_data", data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ATTENTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks_to_analyze, default_context_combinations = get_dataset_config(dataset_name)
    print(f"📊 Tasks to analyze: {tasks_to_analyze}")

    if dataset_name.upper() in NO_CONTEXT_SWEEP_DATASETS:
        valid_combinations = [['ALL']]
    elif 'context' in cell_data:
        available_contexts = set(cell_data['context'])
        valid_combinations = [c for c in default_context_combinations if c == ['ALL'] or all(x in available_contexts for x in c)]
    else:
        valid_combinations = [['ALL']]

    all_embeddings = cell_data["embedding"] if isinstance(cell_data["embedding"], torch.Tensor) else torch.tensor(cell_data["embedding"])

    for context_combo in valid_combinations:
        print(f"\n{'='*80}\nANALYZING ATTENTION FOR CONTEXT COMBINATION: {context_combo}\n{'='*80}")

        if context_combo == ['ALL']:
            filtered_cell_data = cell_data
            filtered_embeddings = all_embeddings
            context_suffix = 'all'
            if 'context' in cell_data:
                available = sorted(set(cell_data['context']))
                context_suffix = '_'.join(available).lower()
        else:
            context_array = np.array(cell_data['context'])
            context_mask = np.isin(context_array, context_combo)
            filtered_cell_data = {}
            for key, values in cell_data.items():
                if key == 'embedding':
                    filtered_cell_data[key] = cell_data[key][context_mask]
                elif isinstance(values, (list, np.ndarray)) and len(values) == len(context_array):
                    filtered_cell_data[key] = [v for v, keep in zip(values, context_mask) if keep] if isinstance(values, list) else values[context_mask]
                else:
                    filtered_cell_data[key] = values
            filtered_embeddings = filtered_cell_data["embedding"] if isinstance(filtered_cell_data["embedding"], torch.Tensor) else torch.tensor(filtered_cell_data["embedding"])
            context_suffix = '_'.join(context_combo).lower()
            print(f"Filtered data: {len(filtered_embeddings)} cells (from {len(all_embeddings)} total)")

        for task in tasks_to_analyze:
            if task not in filtered_cell_data:
                print(f"⚠️ Task '{task}' not found in filtered data for context {context_combo}")
                continue

            if task == 'CAG_2' and dataset_name.upper() == 'HH':
                modes_to_run = [
                    {'mode': 'regression', 'classification_mode': None},
                    {'mode': '4-class', 'classification_mode': '4-class'},
                    {'mode': '6-class', 'classification_mode': '6-class'},
                ]
            else:
                modes_to_run = [{'mode': None, 'classification_mode': None}]

            for mode_config in modes_to_run:
                classification_mode = mode_config['classification_mode']
                mode_name = mode_config['mode']

                if classification_mode is not None:
                    task_type = 'classification'
                elif task in CLASSIFICATION_TASKS:
                    task_type = 'classification'
                elif task in REGRESSION_TASKS:
                    task_type = 'regression'
                else:
                    print(f"⚠️ Task '{task}' is neither classification nor regression. Skipping.")
                    continue

                task_specific_cell_data = filtered_cell_data.copy() if (task == 'CAG_2' and classification_mode is not None) else filtered_cell_data
                if task == 'CAG_2' and classification_mode is not None:
                    original = np.array(task_specific_cell_data[task])
                    converter = cag_to_class_4_class if classification_mode == '4-class' else cag_to_class_6_class
                    task_specific_cell_data[task] = [converter(v) for v in original]

                print(f"\n--- Analyzing attention for task: {task}{f' - Mode: {mode_name}' if mode_name else ''} ({task_type}) (context: {context_combo}) ---")

                input_dim = filtered_embeddings.shape[1]
                try:
                    model, label_scaler, label_metadata = load_context_aware_model(
                        task=task, task_type=task_type, input_dim=input_dim, context_suffix=context_suffix,
                        device=device, model_dir=TRAINED_MODELS_DIR, dataset_name=dataset_name, mode=mode_name,
                    )
                except FileNotFoundError as e:
                    print(f"  ⚠️ {e}")
                    continue

                print("  🔍 Extracting attention weights...")
                attention_weights, predictions, patient_ids, patient_donor_labels, patient_y, patient_cell_types, patient_gRNAs, patient_metadata = (
                    get_attention_weights_context_aware(model, task_specific_cell_data, device, context_key=task, task_type=task_type, label_scaler=label_scaler)
                )

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                suffix = f"_{mode_name}" if mode_name else ""
                out_path = ATTENTION_OUTPUT_DIR / f"attention_{task}{suffix}_attention_context_{context_suffix}_{ts}.npz"

                print(f"  📊 Attention weights shape: {attention_weights.shape}")
                metadata_to_save = {k: np.array(v, dtype=object) for k, v in patient_metadata.items()}

                np.savez_compressed(
                    out_path,
                    attention_weights=attention_weights,
                    predictions=np.array(predictions),
                    patient_ids=np.array([pid.item() for pid in patient_ids]),
                    patient_donor_ids=np.array(patient_donor_labels, dtype=object),
                    patient_y=np.array(patient_y, dtype=object),
                    patient_cell_types=np.array(patient_cell_types, dtype=object),
                    patient_gRNAs=np.array(patient_gRNAs, dtype=object),
                    patient_metadata=np.array([metadata_to_save], dtype=object),
                    context_combination=context_combo,
                    context_suffix=context_suffix,
                    task=task,
                    mode=mode_name,
                    task_type=task_type,
                    label_metadata=np.array([label_metadata], dtype=object) if label_metadata is not None else np.array([None], dtype=object),
                )
                print(f"  💾 Saved attention outputs to {out_path}")

        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Context-aware attention analysis for donor-level models")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. SEATTLE, AUTISM, HH)")
    parser.add_argument("--models-dir", type=str, default=None)
    embedding_group = parser.add_mutually_exclusive_group()
    embedding_group.add_argument("--data-path", type=str, default=None)
    embedding_group.add_argument("--embedding-chunks", nargs='+')
    embedding_group.add_argument("--embedding-dir", type=str)
    parser.add_argument("--use-subset", action='store_true')
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.use_subset and args.embedding_dir is None:
        parser.error("--use-subset requires --embedding-dir pointing to the subset directory")

    main(models_dir=args.models_dir, data_path=args.data_path, output_dir=args.output_dir,
         dataset_name=args.dataset, embedding_chunks=args.embedding_chunks,
         embedding_dir=args.embedding_dir, use_subset=args.use_subset)
