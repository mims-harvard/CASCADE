#!/usr/bin/env python3
"""
Cell-Level Classification Script

Trains (or evaluates, with --eval-only) cell-level MLP probes that map cell
embeddings directly to classification/regression targets, without donor-level
aggregation (Methods 9.11).

Usage:
    # Standard mode (extract embeddings from model):
    python -m cascade.explainer.main_cell_level \\
        --dataset AUTISM --model-path /path/to/model --save-model-dir ./models

    # Using pre-subsetted embeddings:
    python -m cascade.explainer.main_cell_level \\
        --embedding-dir /path/to/subset/directory --use-subset \\
        --dataset SEATTLE --save-model-dir ./models

    # Using chunked embeddings (see get_embeddings_parallel.py), eval-only:
    python -m cascade.explainer.main_cell_level \\
        --dataset M2 --chunk-dir /path/to/chunks --chunk-prefix embeddings_parallel_rank \\
        --save-model-dir ./models --eval-only --predict-full-data
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from cascade.data import splits as donor_splits
from cascade.explainer.cell_level import CellLevelTrainer, load_saved_cell_level_mlp
from cascade.explainer.chunked_io import load_chunked_embeddings, load_embeddings_from_subset
from cascade.explainer.config import (
    CELL_LEVEL_TASKS, CLASSIFICATION_TASKS, CONTEXT_COMBINATIONS,
    DEBUG_MODE, DEBUG_MAX_BATCHES, REGRESSION_TASKS,
)
from cascade.explainer.embeddings import EmbeddingDataLoader
from cascade.explainer.save_trained_models import create_model_registry, save_trained_model
from cascade.explainer.utils import filter_data_by_context, prepare_task_data, print_task_info

# M2 mouse-thyroid Cre-transcript filtering (Methods 9.1): only cells with a detected
# Cre transcript are considered to have active DN-THR/WT-THR receptor expression.
EXCLUDE_NON_NEURONS = True

NON_NEURONAL_CELL_TYPES = {
    "astrocyte", "vascular leptomeningeal cell (VLMC)", "perivascular macrophage",
    "oligodendrocyte", "oligodendrocyte precursor cell (OPC)", "endothelial cell",
    "microglial cell", "pericyte",
}


def get_cre_positive_mask(cell_data):
    """Boolean mask for rows where Cre is 'bigger than 0'. Does not modify cell_data."""
    if 'Cre' not in cell_data:
        raise ValueError("This dataset requires 'Cre' key in cell_data")

    cre = np.asarray(cell_data['Cre'], dtype=object)

    def keep_row(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return False
        if isinstance(v, str) and v.strip() == '':
            return False
        try:
            return float(v) > 0
        except (TypeError, ValueError):
            return False

    mask = np.array([keep_row(v) for v in cre], dtype=bool)
    print(f"\n📌 Cre filter mask: {mask.sum()} / {len(cre)} cells have Cre > 0", flush=True)
    return mask


def _apply_mask_in_place(cell_data, mask, n_before):
    for key in list(cell_data.keys()):
        val = cell_data[key]
        if isinstance(val, torch.Tensor) and val.shape[0] == n_before:
            cell_data[key] = val[mask]
        elif isinstance(val, np.ndarray) and val.shape[0] == n_before:
            cell_data[key] = val[mask]
        elif isinstance(val, list) and len(val) == n_before:
            cell_data[key] = np.asarray(val)[mask]
    return cell_data


def get_dataset_config(dataset_name, cell_data=None):
    """
    Dataset-specific tasks and (for chunked datasets) default chunk locations.
    Donor splits come from cascade.data.splits.SPLITS_BY_DATASET.

    Returns:
        (split, context_tasks, donor2id, chunk_config)
    """
    split = donor_splits.SPLITS_BY_DATASET.get(dataset_name.upper())

    if dataset_name.upper() == 'HH':
        return split, ['VS_Grade', 'Onset/Motor', 'Onset/Cog', 'CAG_1', 'CAG_2', 'CAG_max', 'CAG_min', 'CAG_diff'], None, None

    if dataset_name.upper() == 'AUTISM':
        # Only 'cell_type' is enabled by default; other candidate tasks (disease, tissue,
        # medication ATC codes, comorbidities - see preprocessing/seattle_ad_metadata.py
        # for the analogous Seattle-AD mapping) can be added here as needed.
        return split, ['cell_type'], None, None

    if dataset_name.upper() == 'M2':
        return split, ['treatment', 'THR'], None, None

    print(f"⚠️ Unknown dataset '{dataset_name}', using default configuration")
    return split, None, None, None


def analyze_context_combination(cell_data, context_combo, dataset_name, data_loader, trainer, save_model_dir,
                                 all_results, tasks_to_analyze=None, downsample_majority=False,
                                 eval_only=False, predict_full_data=False, skip_train_metrics=False):
    """
    Args:
        eval_only: load a saved checkpoint from save_model_dir instead of training.
        predict_full_data: score every labeled cell (no train/test split); primary
            metrics come from that full set, with hold-out test metrics also stored
            under holdout_* when a fresh model is trained in the same run.
    """
    if tasks_to_analyze is None:
        tasks_to_analyze = CELL_LEVEL_TASKS
    print(f"\n{'='*80}\nANALYZING CONTEXT COMBINATION: {context_combo}\n{'='*80}")

    filtered_cell_data, filtered_embeddings, context_suffix = filter_data_by_context(cell_data, context_combo)
    if context_combo != ['ALL']:
        print(f"Filtered data: {len(filtered_embeddings)} cells (from {len(cell_data['embedding'])} total)")

    donors_split = cell_data.get('donors_split') or (data_loader.get_donor_split() if data_loader is not None else None)
    if donors_split is None:
        print("❌ Error: No donors_split available and no data_loader")
        return

    print(f"\n📝 Processing {len(tasks_to_analyze)} task(s): {tasks_to_analyze}", flush=True)
    for task_idx, task in enumerate(tasks_to_analyze, 1):
        print(f"\n{'='*80}\nTASK {task_idx}/{len(tasks_to_analyze)}: {task}\n{'='*80}", flush=True)

        task_filtered_cell_data = filtered_cell_data.copy()
        task_filtered_embeddings = filtered_embeddings

        if dataset_name.upper() == 'M2':
            task_filtered_cell_data, task_filtered_embeddings = _m2_task_filter(
                task, task_filtered_cell_data, task_filtered_embeddings
            )
            if task_filtered_cell_data is None:
                continue

        classification_tasks_to_use = CLASSIFICATION_TASKS
        if dataset_name.upper() == 'M2':
            classification_tasks_to_use = list(set(CLASSIFICATION_TASKS) | {'treatment', 'THR'})

        task_data = prepare_task_data(task, task_filtered_cell_data, task_filtered_embeddings, classification_tasks_to_use, REGRESSION_TASKS)
        if task_data is None:
            print(f"  ⚠️ Skipping task '{task}' - no valid data found", flush=True)
            continue

        labels, embeddings, task_type = task_data
        print_task_info(task, task_type, labels, context_combo)

        if 'donor_id' not in task_filtered_cell_data:
            print("⚠️ No 'donor_id' found in data. Skipping task.", flush=True)
            continue
        donor_ids = np.array(task_filtered_cell_data['donor_id'])
        if task in task_filtered_cell_data:
            original_labels = np.array(task_filtered_cell_data[task])
            valid_mask = (original_labels != None) & ~pd.isna(original_labels) if original_labels.dtype == 'object' else ~pd.isna(original_labels)  # noqa: E711
            donor_ids = donor_ids[valid_mask]

        context_task_name = f"{task}_context_{context_suffix}" if context_suffix != "all" else task
        pred_dir = save_model_dir or "./trained_models"
        ckpt_path = Path(pred_dir) / f"{context_task_name}_none_mlp_{task_type}_context_{context_suffix}.pt"

        label_encoder = None
        if task_type == "classification":
            label_encoder = LabelEncoder()
            if eval_only and ckpt_path.is_file():
                ck_meta = torch.load(ckpt_path, map_location="cpu")
                classes_ck = (ck_meta.get("additional_info", {}) or {}).get("label_encoder_classes_")
                if classes_ck is not None:
                    label_encoder.fit(np.asarray(classes_ck))
                    labels_encoded = label_encoder.transform(np.asarray(labels))
                    n_classes = len(label_encoder.classes_)
                else:
                    labels_encoded = label_encoder.fit_transform(labels)
                    n_classes = len(label_encoder.classes_)
                    print("  ⚠️ No label_encoder_classes_ in checkpoint; fit on current labels (order may differ)", flush=True)
            else:
                labels_encoded = label_encoder.fit_transform(labels)
                n_classes = len(label_encoder.classes_)
        else:
            labels_encoded = labels.astype(np.float32)
            n_classes = None

        input_dim = embeddings.shape[1]
        holdout_metrics = None

        if eval_only:
            if not ckpt_path.exists():
                print(f"  ⚠️ Checkpoint not found for eval-only: {ckpt_path}", flush=True)
                continue
            model, _ = load_saved_cell_level_mlp(ckpt_path, input_dim, task_type, n_classes, trainer.device)
            if predict_full_data:
                metrics, test_preds, test_labels, test_probs = trainer.predict_all_labeled_cells(model, embeddings, labels_encoded, task_type, debug_mode=DEBUG_MODE)
            else:
                metrics, test_preds, test_labels, test_probs = trainer.evaluate_saved_cell_level_model(
                    model, embeddings, labels_encoded, donor_ids, donors_split, task_type,
                    debug_mode=DEBUG_MODE, skip_train_metrics=skip_train_metrics,
                )
        else:
            holdout_metrics, model, test_preds, test_labels, test_probs = trainer.train_cell_level_task(
                embeddings, labels_encoded, donor_ids, donors_split, task_type, input_dim, n_classes,
                DEBUG_MODE, downsample_majority=downsample_majority,
            )
            if predict_full_data:
                metrics, test_preds, test_labels, test_probs = trainer.predict_all_labeled_cells(model, embeddings, labels_encoded, task_type, debug_mode=DEBUG_MODE)
            else:
                metrics = holdout_metrics

        result = {
            "dataset": dataset_name,
            "task": task,
            "task_type": task_type,
            "prediction_level": "cell",
            "model_architecture": "mlp",
            "split_type": "full_labeled_cells" if predict_full_data else "fixed",
            "context_combination": "_".join(context_combo),
            "n_samples": len(labels),
            "n_classes": n_classes if task_type == "classification" else None,
        }
        result.update(metrics)
        if holdout_metrics is not None and predict_full_data:
            for k, v in holdout_metrics.items():
                if isinstance(v, (float, int, str, bool, np.floating, np.integer)) or v is None:
                    result[f"holdout_{k}"] = v

        pred_suffix = "_full_dataset" if predict_full_data else ""
        pred_path = Path(pred_dir) / f"predictions_{context_task_name}{pred_suffix}.csv"
        try:
            if task_type == "classification":
                out = pd.DataFrame({
                    "label": label_encoder.inverse_transform(test_labels.astype(int)),
                    "label_encoded": test_labels,
                    "prediction": label_encoder.inverse_transform(test_preds.astype(int)),
                    "prediction_encoded": test_preds,
                })
                if test_probs is not None and test_probs.ndim >= 2:
                    for c in range(test_probs.shape[1]):
                        out[f"prob_class_{c}"] = test_probs[:, c]
            else:
                out = pd.DataFrame({"label": test_labels, "prediction": test_preds})
            if predict_full_data and len(donor_ids) == len(out):
                out.insert(0, "donor_id", np.asarray(donor_ids))
            out.to_csv(pred_path, index=False)
        except Exception as e:
            print(f"  ⚠️ Could not write predictions CSV: {e}", flush=True)

        if not eval_only:
            metrics_for_registry = holdout_metrics if holdout_metrics is not None else metrics
            extra_info = {
                "n_classes": n_classes, "input_dim": input_dim, "metrics": metrics_for_registry,
                "n_samples": len(labels), "context_combination": context_combo,
                "context_suffix": context_suffix, "prediction_level": "cell",
            }
            if predict_full_data and holdout_metrics is not None:
                extra_info["metrics_full_labeled_cells"] = metrics
            if task_type == "classification":
                extra_info["label_encoder_classes_"] = label_encoder.classes_.tolist()

            result["saved_model_path"] = save_trained_model(
                model=model, task=context_task_name, aggregation_method="none", model_type="mlp",
                task_type=task_type, save_dir=pred_dir, context_suffix=context_suffix, additional_info=extra_info,
            )

        all_results.append(result)
        print(f"    Results: {metrics}")


def _m2_task_filter(task, cell_data, embeddings):
    """
    M2 (mouse thyroid, Methods 9.1) task-specific cell filtering:
      - treatment: keep (Cre is None) OR (Cre == 0 AND THR == 1); no THR filter when Cre is None.
      - THR: keep treatment==1, then restrict to Cre>0 and (by default) neuronal cell types.
    """
    if 'Cre' not in cell_data or (task == 'treatment' and 'THR' not in cell_data):
        print(f"  ⚠️ Missing Cre/THR columns; cannot filter task '{task}'", flush=True)
        return None, None

    n_before = len(embeddings)

    def _is_one(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return False
        try:
            return float(v) == 1
        except (TypeError, ValueError):
            return False

    if task == 'treatment':
        cre_values = np.asarray(cell_data['Cre'], dtype=object)
        thr_values = np.asarray(cell_data['THR'])
        cre_none = np.array([v is None for v in cre_values], dtype=bool)
        cre_zero = np.array([_is_zero(v) for v in cre_values], dtype=bool)
        thr_one = np.array([_is_one(v) for v in thr_values], dtype=bool)
        keep_mask = cre_none | (cre_zero & thr_one)
        cell_data = _apply_mask_in_place(cell_data, keep_mask, n_before)
        embeddings = np.asarray(embeddings)[keep_mask]
        return cell_data, embeddings

    if task == 'THR':
        if 'treatment' not in cell_data:
            print(f"  ⚠️ Skipping task '{task}' - 'treatment' column not found", flush=True)
            return None, None
        treatment_mask = np.array([_is_one(v) for v in np.asarray(cell_data['treatment'])], dtype=bool)
        cell_data = _apply_mask_in_place(cell_data, treatment_mask, n_before)
        embeddings = np.asarray(embeddings)[treatment_mask]

        n_before_cre = len(embeddings)
        cre_mask = get_cre_positive_mask(cell_data)
        cell_data = _apply_mask_in_place(cell_data, cre_mask, n_before_cre)
        embeddings = np.asarray(embeddings)[cre_mask]

        if EXCLUDE_NON_NEURONS and 'cell_type' in cell_data:
            n_before_ct = len(embeddings)
            ct_values = np.asarray(cell_data['cell_type'], dtype=object)
            neuron_mask = ~np.isin(ct_values, list(NON_NEURONAL_CELL_TYPES))
            cell_data = _apply_mask_in_place(cell_data, neuron_mask, n_before_ct)
            embeddings = np.asarray(embeddings)[neuron_mask]

        return cell_data, embeddings

    return cell_data, embeddings


def _is_zero(v):
    try:
        return float(v) == 0.0
    except (TypeError, ValueError):
        return False


def main(dataset_name=None, model_path=None, existing_embeddings_path=None, save_model_dir=None,
         save_embeddings_path=None, embedding_dir=None, use_subset=False, cell_level_tasks=None,
         chunk_dir=None, chunk_prefix=None, downsample_majority=False, eval_only=False,
         predict_full_data=False, skip_train_metrics=False):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print("=" * 80)
    print("CELL-LEVEL CLASSIFICATION ANALYSIS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {dataset_name}, device: {device}")

    trainer = CellLevelTrainer(device=device)
    data_loader = None

    if chunk_dir is not None:
        if chunk_prefix is None:
            print("❌ Error: --chunk-dir requires --chunk-prefix")
            return
        cell_data = load_chunked_embeddings(chunk_dir=chunk_dir, chunk_prefix=chunk_prefix)
    elif use_subset:
        if not embedding_dir:
            print("❌ Error: --use-subset requires --embedding-dir")
            return
        dataset_split, _, _, _ = get_dataset_config(dataset_name, None)
        cell_data = load_embeddings_from_subset(embedding_dir, donors_split=dataset_split)
    else:
        data_loader = EmbeddingDataLoader(dataset_name=dataset_name, model_path=model_path)
        cell_data = data_loader.get_embeddings(existing_embeddings_path=existing_embeddings_path, save_embeddings_path=save_embeddings_path)

    cell_data["embedding"] = cell_data["embedding"] if isinstance(cell_data["embedding"], torch.Tensor) else torch.tensor(cell_data["embedding"])
    print(f"✅ Ready to analyze {len(cell_data['embedding'])} cells", flush=True)

    if dataset_name.upper() == 'M2' and 'Cre' in cell_data:
        cre_mask = get_cre_positive_mask(cell_data)
        print(f"   Cells with Cre>0 (global, informational only): {int(cre_mask.sum())} / {len(cre_mask)}", flush=True)

    dataset_split, dataset_tasks, donor2id, _ = get_dataset_config(dataset_name, cell_data)

    if donor2id is not None and 'donor_id' in cell_data:
        donor_id_raw = np.asarray(cell_data['donor_id'])
        if donor_id_raw.dtype.kind in {"U", "S", "O"}:
            cell_data['donor_id'] = np.array([donor2id[d] for d in donor_id_raw], dtype=np.int64)

    if dataset_split is not None:
        cell_data['donors_split'] = dataset_split
    elif not use_subset and data_loader is not None:
        cell_data['donors_split'] = data_loader.get_donor_split()
    elif 'donor_id' in cell_data:
        unique_donors = np.unique(cell_data['donor_id'])
        n_test = max(1, len(unique_donors) // 5)
        np.random.seed(42)
        test_donors = np.random.choice(unique_donors, n_test, replace=False).tolist()
        cell_data['donors_split'] = {
            'train_donors': [d for d in unique_donors if d not in test_donors],
            'test_donors': test_donors,
        }
    else:
        print("❌ Error: No donor_id found in data and no dataset-specific split available")
        return

    tasks_to_analyze = cell_level_tasks or dataset_tasks or CELL_LEVEL_TASKS
    print(f"🎯 Tasks to analyze: {tasks_to_analyze}")

    if 'context' in cell_data:
        available_contexts = set(cell_data['context'])
        valid_combinations = [c for c in CONTEXT_COMBINATIONS if c == ['ALL'] or all(x in available_contexts for x in c)]
    else:
        valid_combinations = [['ALL']]

    all_results = []
    for context_combo in valid_combinations:
        analyze_context_combination(
            cell_data, context_combo, dataset_name, data_loader, trainer, save_model_dir, all_results,
            tasks_to_analyze=tasks_to_analyze, downsample_majority=downsample_majority,
            eval_only=eval_only, predict_full_data=predict_full_data, skip_train_metrics=skip_train_metrics,
        )

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv("cell_level_prediction_results.csv", index=False)
        print(f"✅ Saved {len(all_results)} result(s) to cell_level_prediction_results.csv")
    else:
        print("❌ No results to save")

    if not eval_only:
        try:
            create_model_registry(save_model_dir or "./trained_models")
        except Exception as e:
            print(f"⚠️ Warning: Could not create model registry: {e}")

    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cell-level classification analysis", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. AUTISM, M2, HH, SEATTLE)")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--save-model-dir", type=str, default=None)
    parser.add_argument("--save-embeddings-path", type=str, default=None)
    parser.add_argument("--load-embeddings-path", type=str, default=None)
    parser.add_argument("--embedding-dir", type=str, default=None)
    parser.add_argument("--use-subset", action='store_true')
    parser.add_argument("--cell-level-tasks", nargs='+', default=None)
    parser.add_argument("--chunk-dir", type=str, default=None)
    parser.add_argument("--chunk-prefix", type=str, default=None)
    parser.add_argument("--downsample-majority", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="Load checkpoints from --save-model-dir instead of training")
    parser.add_argument("--predict-full-data", action="store_true", help="Score all labeled cells (no split); primary metrics are full-dataset")
    parser.add_argument("--skip-train-metrics", action="store_true")
    args = parser.parse_args()

    if args.use_subset and args.embedding_dir is None:
        parser.error("--use-subset requires --embedding-dir")

    main(
        dataset_name=args.dataset, model_path=args.model_path,
        existing_embeddings_path=args.load_embeddings_path, save_model_dir=args.save_model_dir,
        save_embeddings_path=args.save_embeddings_path, embedding_dir=args.embedding_dir,
        use_subset=args.use_subset, cell_level_tasks=args.cell_level_tasks,
        chunk_dir=args.chunk_dir, chunk_prefix=args.chunk_prefix,
        downsample_majority=args.downsample_majority, eval_only=args.eval_only,
        predict_full_data=args.predict_full_data, skip_train_metrics=args.skip_train_metrics,
    )
