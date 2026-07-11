#!/usr/bin/env python3
"""
Donor-Level Disease Prediction Analysis - Main Script

Orchestrates donor-level prediction (Methods 9.8, 9.11) using PatientAggregator-
based attention models on frozen CASCADE embeddings.

Supports AUTISM, HH (Huntington's), and SEATTLE (Alzheimer's neuropathology
staging - ADNC/Braak/CERAD/Thal/LATE-NC/Lewy body/cognitive status, Methods 9.1).
SEATTLE is loaded from pre-extracted chunked embeddings (see
get_embeddings_parallel.py) rather than run through the transformer directly,
since it is too large to fit comfortably in memory as a single collator.

Usage:
    python -m cascade.explainer.main_donor_level --dataset AUTISM --model-path /path/to/model.pt
    python -m cascade.explainer.main_donor_level --dataset HH
    python -m cascade.explainer.main_donor_level --dataset SEATTLE \\
        --chunk-dir /path/to/chunks --chunk-prefix embeddings_parallel_rank
"""
import argparse
import builtins
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

from cascade.data import splits as donor_splits
from cascade.explainer.chunked_io import load_chunked_embeddings
from cascade.explainer.config import (
    CLASSIFICATION_TASKS, CONTEXT_COMBINATIONS, DEBUG_MODE, DEBUG_MAX_BATCHES,
    DONOR_LEVEL_TASKS, HH_DONOR_LEVEL_TASKS, REGRESSION_TASKS,
)
from cascade.explainer.embeddings import EmbeddingDataLoader
from cascade.explainer.save_trained_models import create_model_registry, save_trained_model
from cascade.explainer.trainer import Trainer, TrainerConfig
from cascade.explainer.utils import (
    cag_to_class_4_class, cag_to_class_6_class, create_donor_level_data,
    filter_data_by_context, prepare_task_data, print_donor_info, print_task_info,
)

# Seattle (AD/neuropathology) donor-level tasks (Methods 9.1, 9.11)
SEATTLE_DONOR_LEVEL_TASKS = [
    "ADNC", "Braak stage", "CERAD score", "Thal phase", "LATE-NC stage",
    "Lewy body disease pathology", "Cognitive status",
]


# Always flush prints so logs appear live during long SLURM runs.
def print(*args, **kwargs):  # noqa: A001
    kwargs.setdefault("flush", True)
    return builtins.print(*args, **kwargs)


def _all_id_forms(x):
    """Common string/numeric forms for a donor id, to make split matching robust
    to '10.0' vs 10.0 vs 10 mismatches between the split and the loaded data."""
    forms = {x, str(x)}
    try:
        xf = float(x)
        forms.add(xf)
        if xf.is_integer():
            forms.add(int(xf))
            forms.add(str(int(xf)))
            forms.add(f"{int(xf)}.0")
    except (TypeError, ValueError):
        pass
    return forms


def _normalize_donor_split_for_matching(donors_split):
    if donors_split is None:
        return None
    normalized = {}
    for split_key in ("train_donors", "test_donors"):
        expanded = set()
        for donor_id in donors_split.get(split_key, []):
            expanded.update(_all_id_forms(donor_id))
        normalized[split_key] = list(expanded)
    return normalized


def _summarize_unique_ids(values, max_show=20):
    unique_vals = np.unique(np.asarray(values))
    return len(unique_vals), unique_vals[:max_show].tolist()


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_context_combination(cell_data, context_combo, dataset_name, data_loader, trainer,
                                 save_model_dir, all_results, tasks_to_analyze=None):
    """Train/evaluate a PatientAggregator model for each task within one context combination."""
    if tasks_to_analyze is None:
        tasks_to_analyze = DONOR_LEVEL_TASKS
    print(f"\n{'='*80}\nANALYZING CONTEXT COMBINATION: {context_combo}\n{'='*80}")
    print(f"Tasks to analyze in this pass: {tasks_to_analyze}")

    filtered_cell_data, filtered_embeddings, context_suffix = filter_data_by_context(cell_data, context_combo)

    if context_combo != ['ALL']:
        print(f"Filtered data: {len(filtered_embeddings)} cells (from {len(cell_data['embedding'])} total)")
        print(f"Context distribution: {np.unique(np.array(filtered_cell_data['context']), return_counts=True)}")
    print(f"Filtered cell_data keys: {sorted(filtered_cell_data.keys())}")
    if "donor_id" in filtered_cell_data:
        n_unique_donors, preview_donors = _summarize_unique_ids(filtered_cell_data["donor_id"])
        print(f"Filtered unique donor IDs: {n_unique_donors} total; first {len(preview_donors)}: {preview_donors}")

    donors_split = cell_data.get("donors_split")
    if donors_split is None and data_loader is not None:
        donors_split = data_loader.get_donor_split()

    for task in tasks_to_analyze:
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

            # Avoid deepcopy for very large datasets (e.g. millions of cells);
            # only one or two task-specific label arrays are overridden below.
            modified_cell_data = dict(filtered_cell_data)

            if task == 'CAG_2' and classification_mode is not None:
                original_cag_labels = np.array(modified_cell_data[task])
                if classification_mode == '4-class':
                    modified_cell_data[task] = [cag_to_class_4_class(v) for v in original_cag_labels]
                    task_type_override = 'classification'
                    print(f"\n=== Processing {task} - Mode: {mode_name} (4-class classification) ===")
                else:
                    modified_cell_data[task] = [cag_to_class_6_class(v) for v in original_cag_labels]
                    task_type_override = 'classification'
                    print(f"\n=== Processing {task} - Mode: {mode_name} (6-class classification) ===")
                converted_labels = np.array(modified_cell_data[task])
                valid_labels = converted_labels[~pd.isna(converted_labels)]
                print(f"Class distribution: {dict(zip(*np.unique(valid_labels, return_counts=True)))}")
            else:
                task_type_override = None

            # HH VS_Grade: replace NaN with -1 only for Control donors (missing Case values stay
            # NaN and are excluded downstream by prepare_task_data).
            if task == 'VS_Grade' and dataset_name.upper() == 'HH':
                disease_labels_array = np.array(modified_cell_data['disease'])
                vs_grade_labels = np.array(modified_cell_data['VS_Grade'])
                is_control = disease_labels_array == 'Control'
                is_nan = pd.isna(vs_grade_labels)
                control_nan_mask = is_control & is_nan
                if control_nan_mask.sum() > 0:
                    print(f"  Setting {control_nan_mask.sum()} NaN values in VS_Grade to -1 (Control samples)")
                    vs_grade_labels = vs_grade_labels.copy()
                    vs_grade_labels[control_nan_mask] = -1
                    modified_cell_data['VS_Grade'] = vs_grade_labels.tolist()
                case_nan_count = ((~is_control) & is_nan).sum()
                if case_nan_count > 0:
                    print(f"  Keeping {case_nan_count} NaN values in VS_Grade as NaN (Case samples - will be excluded)")

            classification_tasks_to_use = CLASSIFICATION_TASKS
            if dataset_name.upper() == "SEATTLE":
                # Seattle neuropathology stages are categorical even when numerically encoded.
                classification_tasks_to_use = list(set(CLASSIFICATION_TASKS) | set(SEATTLE_DONOR_LEVEL_TASKS))

            task_data = prepare_task_data(task, modified_cell_data, filtered_embeddings, classification_tasks_to_use, REGRESSION_TASKS)
            if task_data is None:
                continue

            labels, embeddings, task_type = task_data
            if task_type_override is not None:
                task_type = task_type_override
            elif dataset_name.upper() == "SEATTLE" and task in SEATTLE_DONOR_LEVEL_TASKS and task_type != "classification":
                print(f"  Forcing task '{task}' to classification for SEATTLE dataset")
                task_type = "classification"

            if not (task == 'CAG_2' and classification_mode is not None):
                if mode_name is not None:
                    print(f"\n=== Processing {task} - Mode: {mode_name} (regression) ===")
                else:
                    print_task_info(task, task_type, labels, context_combo)

            print("\n--- DONOR-LEVEL PREDICTION (PatientAggregator) ---")

            label_encoder = None
            label_scaler = None
            if task_type == 'classification':
                label_encoder = LabelEncoder()
                labels = label_encoder.fit_transform(labels)
                print(f"  Encoding labels: {label_encoder.classes_} -> {list(range(len(label_encoder.classes_)))}")
            else:
                label_scaler = StandardScaler()
                labels = label_scaler.fit_transform(np.asarray(labels).reshape(-1, 1)).flatten()
                print(f"  Scaling labels: mean={label_scaler.mean_[0]:.4f}, std={label_scaler.scale_[0]:.4f}")

            if task in modified_cell_data:
                original_labels = np.array(modified_cell_data[task])
                valid_mask = (original_labels != None) if original_labels.dtype == 'object' else ~pd.isna(original_labels)  # noqa: E711
            else:
                valid_mask = np.ones(len(labels), dtype=bool)

            donor_data = create_donor_level_data(labels, embeddings, modified_cell_data, valid_mask)
            if donor_data is None:
                continue
            donor_embeddings, donor_labels, donor_id_list, donor_ids_filtered = donor_data
            print_donor_info(donor_embeddings, donor_id_list, donor_labels, DEBUG_MODE, donors_split)

            print("\n  Training attention-based PatientAggregator model...")
            compatible_split = _normalize_donor_split_for_matching(donors_split)
            metrics, model = trainer.train_donor_level_task(
                donor_embeddings, donor_labels, donor_id_list, compatible_split,
                task_type, DEBUG_MODE, label_scaler=label_scaler,
            )

            result = {
                'dataset': dataset_name,
                'task': task,
                'task_type': task_type,
                'prediction_level': 'donor',
                'model_architecture': 'attention',
                'split_type': 'fixed',
                'context_combination': '_'.join(context_combo),
                'n_samples': len(donor_labels),
                'n_classes': len(np.unique(donor_labels)) if task_type == 'classification' else None,
            }
            if mode_name is not None:
                result['mode'] = mode_name
                if classification_mode is not None:
                    result['classification_mode'] = classification_mode
            result.update(metrics)

            context_task_name = (
                f"{task}_{mode_name}_context_{context_suffix}" if mode_name is not None and context_suffix != 'all'
                else f"{task}_{mode_name}" if mode_name is not None
                else f"{task}_context_{context_suffix}" if context_suffix != 'all'
                else task
            )

            additional_info = {
                'n_classes': len(np.unique(donor_labels)) if task_type == 'classification' else None,
                'input_dim': len(donor_embeddings[0][0]) if donor_embeddings else None,
                'metrics': metrics,
                'n_samples': len(donor_labels),
                'context_combination': context_combo,
                'context_suffix': context_suffix,
                'mode': mode_name,
                'classification_mode': classification_mode,
            }
            if task_type == 'classification' and label_encoder is not None:
                additional_info['label_encoder_classes'] = label_encoder.classes_.tolist()
                additional_info['label_mapping'] = {label: idx for idx, label in enumerate(label_encoder.classes_)}
            if task_type == 'regression' and label_scaler is not None:
                additional_info['label_scaler_mean'] = float(label_scaler.mean_[0])
                additional_info['label_scaler_scale'] = float(label_scaler.scale_[0])

            result['saved_model_path'] = save_trained_model(
                model=model, task=context_task_name, aggregation_method='attention', model_type='attention',
                task_type=task_type, save_dir=save_model_dir or "./trained_models",
                context_suffix=context_suffix, additional_info=additional_info,
            )

            all_results.append(result)
            print(f"    Results: {metrics}")


def main(dataset_name: str = None, model_path: str = None, existing_embeddings_path: str = None,
         save_model_dir: str = None, save_embeddings_path: str = None, trainer_config: TrainerConfig = None,
         chunk_dir: str = None, chunk_prefix: str = None):
    print("=" * 80)
    print("DONOR-LEVEL DISEASE PREDICTION ANALYSIS (PatientAggregator)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n📋 CONFIGURATION:")
    print(f"  Debug mode: {DEBUG_MODE} (max batches: {DEBUG_MAX_BATCHES if DEBUG_MODE else 'N/A'})")
    print(f"  Dataset: {dataset_name}")
    print(f"  Model path: {model_path or 'Using default'}")
    if chunk_dir and chunk_prefix:
        print(f"  Chunk dir: {chunk_dir}")
        print(f"  Chunk prefix: {chunk_prefix}")
    print(f"  Device: {device}")
    print("=" * 80)

    trainer = Trainer(device=device, config=trainer_config) if trainer_config is not None else Trainer(device=device)

    data_loader = None
    print("\n📊 GETTING EMBEDDINGS...")
    if dataset_name.upper() == "SEATTLE" and chunk_dir and chunk_prefix:
        print("  Loading SEATTLE chunked embeddings...")
        cell_data = load_chunked_embeddings(chunk_dir=chunk_dir, chunk_prefix=chunk_prefix)
        if not isinstance(cell_data.get("embedding"), torch.Tensor):
            cell_data["embedding"] = torch.tensor(cell_data["embedding"])
        seattle_split = donor_splits.SPLITS_BY_DATASET["SEATTLE"]
        cell_data["donors_split"] = seattle_split
        print(f"  SEATTLE: {len(seattle_split['train_donors'])} train, {len(seattle_split['test_donors'])} test donors")
    else:
        data_loader = EmbeddingDataLoader(dataset_name=dataset_name, model_path=model_path)
        cell_data = data_loader.get_embeddings(
            existing_embeddings_path=existing_embeddings_path, save_embeddings_path=save_embeddings_path,
        )

    if isinstance(cell_data["embedding"], torch.Tensor):
        if cell_data["embedding"].dtype == torch.float64:
            cell_data["embedding"] = cell_data["embedding"].float()  # float64 doubles memory; models train in float32
    else:
        cell_data["embedding"] = torch.as_tensor(cell_data["embedding"])
        if cell_data["embedding"].dtype == torch.float64:
            cell_data["embedding"] = cell_data["embedding"].float()

    print(f"✅ Ready to analyze {len(cell_data['embedding'])} cells")
    print(f"Available cell_data keys: {sorted(cell_data.keys())}")
    if "donor_id" in cell_data:
        n_unique_donors, preview_donors = _summarize_unique_ids(cell_data["donor_id"])
        print(f"Unique donor IDs in loaded data: {n_unique_donors} total; first {len(preview_donors)}: {preview_donors}")

    if dataset_name.upper() == "HH":
        tasks_to_analyze = HH_DONOR_LEVEL_TASKS
        print(f"📊 HH dataset: Using HH-specific tasks: {tasks_to_analyze}")
    elif dataset_name.upper() == "SEATTLE":
        tasks_to_analyze = SEATTLE_DONOR_LEVEL_TASKS
        print(f"📊 SEATTLE dataset: Using tasks: {tasks_to_analyze}")
    else:
        tasks_to_analyze = DONOR_LEVEL_TASKS
        print(f"📊 Using standard donor-level tasks: {tasks_to_analyze}")

    if dataset_name.upper() in ("HH", "SEATTLE"):
        print(f"🔬 {dataset_name} dataset: Using context-agnostic embeddings (no context filtering)")
        valid_combinations = [["DISEASE"]]
    elif "context" in cell_data:
        available_contexts = set(cell_data["context"])
        print(f"Available contexts in data: {available_contexts}")
        valid_combinations = [
            combo for combo in CONTEXT_COMBINATIONS
            if combo == ["ALL"] or all(ctx in available_contexts for ctx in combo)
        ]
        print(f"Valid context combinations: {valid_combinations}")
    else:
        print("⚠️ No 'context' key found in cell_data. Running without context filtering.")
        valid_combinations = [["ALL"]]

    all_results = []
    for context_combo in valid_combinations:
        analyze_context_combination(
            cell_data, context_combo, dataset_name, data_loader, trainer, save_model_dir, all_results,
            tasks_to_analyze=tasks_to_analyze,
        )

    if all_results:
        results_df = pd.DataFrame(all_results)
        output_path = Path("donor_level_prediction_results.csv")
        results_df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to: {output_path}")
        print(f"Total results: {len(all_results)}")

        print("\n" + "=" * 80)
        print("SUMMARY (PatientAggregator-based Results)")
        print("=" * 80)
        for task in tasks_to_analyze:
            task_results = results_df[results_df['task'] == task]
            if not task_results.empty:
                print(f"\n{task.upper()} (Donor-level):")
                for _, row in task_results.iterrows():
                    mode_str = f" ({row['mode']})" if 'mode' in row and pd.notna(row['mode']) else ""
                    if row['task_type'] == 'classification':
                        print(f"  Attention model{mode_str}: F1={row['f1']:.4f}, AUROC={row['auroc']:.4f}")
                    else:
                        print(f"  Attention model{mode_str}: R2={row['r2']:.4f}, Pearson={row['pearson']:.4f}")
    else:
        print("❌ No results to save")

    print("\n📁 Creating model registry...")
    try:
        create_model_registry(save_model_dir or './trained_models')
        print("✅ Model registry created successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not create model registry: {e}")

    print("\n🎉 Analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Donor-level disease prediction analysis (PatientAggregator)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. AUTISM, HH, SEATTLE)")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model checkpoint (not needed for HH/SEATTLE)")
    parser.add_argument("--save-model-dir", type=str, default=None, help="Directory where trained models will be saved")
    parser.add_argument("--save-embeddings-path", type=str, default=None, help="Path to pickle file where embeddings+metadata will be saved")
    parser.add_argument("--load-embeddings-path", type=str, default=None, help="Path to pickle file with saved embeddings to load instead of extracting")
    parser.add_argument("--chunk-dir", type=str, default=None, help="For SEATTLE: directory containing chunked embedding .pkl files")
    parser.add_argument("--chunk-prefix", type=str, default=None, help="For SEATTLE: prefix of chunk files")
    args = parser.parse_args()

    if args.dataset.upper() == "SEATTLE" and (not args.chunk_dir or not args.chunk_prefix):
        parser.error("SEATTLE requires --chunk-dir and --chunk-prefix")

    if args.dataset.upper() == 'HH':
        trainer_config = TrainerConfig(
            learning_rate=0.0001, early_stopping_patience=50, max_epochs=500,
            train_batch_size=128, test_batch_size=128,
            attention_heads=4, attention_layers=6, attention_dropout=0.1,
        )
    else:
        trainer_config = None

    main(
        dataset_name=args.dataset, model_path=args.model_path,
        existing_embeddings_path=args.load_embeddings_path, save_model_dir=args.save_model_dir,
        save_embeddings_path=args.save_embeddings_path, trainer_config=trainer_config,
        chunk_dir=args.chunk_dir, chunk_prefix=args.chunk_prefix,
    )
