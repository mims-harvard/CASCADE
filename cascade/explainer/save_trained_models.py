"""
Saves trained donor-level/cell-level prediction models with metadata, and
builds a registry so later scripts (attention analysis, gene explainer) can
look models up by task.
"""
import pickle
from pathlib import Path

import torch


def save_trained_model(model, task, aggregation_method, model_type, task_type, save_dir="./trained_models",
                        context_suffix=None, additional_info=None):
    """
    Save a trained model to disk with metadata needed to reload and identify it later.

    Args:
        model: the trained PyTorch model.
        task: prediction task name (e.g. 'disease', 'cell_type').
        aggregation_method: 'attention' (donor-level) or 'none' (cell-level).
        model_type: 'attention' or 'mlp'.
        task_type: 'classification' or 'regression'.
        save_dir: directory to save the model.
        context_suffix: optional context combination suffix (e.g. 'cells_tissue').
        additional_info: optional dict of extra metadata (e.g. n_classes, metrics).

    Returns:
        str: path to the saved model file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{task}_{aggregation_method}_{model_type}_{task_type}_context_{context_suffix}.pt"
        if context_suffix else
        f"{task}_{aggregation_method}_{model_type}_{task_type}.pt"
    )
    filepath = save_dir / filename

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'task': task,
        'aggregation_method': aggregation_method,
        'model_type': model_type,
        'task_type': task_type,
        'context_suffix': context_suffix,
        'model_class': model.__class__.__name__,
        'model_architecture': str(model),
    }
    if additional_info:
        checkpoint['additional_info'] = additional_info

    torch.save(checkpoint, filepath)
    print(f"    Model saved: {filepath}")
    return str(filepath)


def create_model_registry(save_dir="./trained_models"):
    """Build (and save) a registry mapping model IDs to file paths and metadata."""
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return {}

    registry = {}
    for model_file in save_dir.glob("*.pt"):
        try:
            checkpoint = torch.load(model_file, map_location='cpu')

            context_suffix = checkpoint.get('context_suffix', '')
            model_id = (
                f"{checkpoint['task']}_{checkpoint['aggregation_method']}_{checkpoint['model_type']}_{checkpoint['task_type']}_context_{context_suffix}"
                if context_suffix else
                f"{checkpoint['task']}_{checkpoint['aggregation_method']}_{checkpoint['model_type']}_{checkpoint['task_type']}"
            )

            registry[model_id] = {
                'filepath': str(model_file),
                'task': checkpoint['task'],
                'aggregation_method': checkpoint['aggregation_method'],
                'model_type': checkpoint['model_type'],
                'task_type': checkpoint['task_type'],
                'context_suffix': context_suffix,
                'model_class': checkpoint['model_class'],
            }
            if 'additional_info' in checkpoint:
                registry[model_id]['additional_info'] = checkpoint['additional_info']

        except Exception as e:
            print(f"Warning: Could not process {model_file}: {e}")

    registry_file = save_dir / "model_registry.pkl"
    with open(registry_file, 'wb') as f:
        pickle.dump(registry, f)

    print(f"Model registry created: {registry_file}")
    print(f"Found {len(registry)} trained models")
    return registry
