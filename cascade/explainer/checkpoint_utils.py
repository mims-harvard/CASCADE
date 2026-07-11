"""
DDP checkpoint loading with automatic "module." prefix handling, shared by the
embedding-extraction and gene-explainer entrypoints.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


def load_ddp_checkpoint(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Load a checkpoint into `model`, automatically handling the "module." key
    prefix mismatch between DDP-wrapped and unwrapped models in either direction,
    then falling back to partial loading of shape-matching weights.

    Returns a dict with 'epoch', 'global_step', 'args', and the raw 'checkpoint'.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    device = torch.device(device) if isinstance(device, str) else (device or torch.device('cpu'))

    if verbose:
        print(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'model_state_dict' key")

    state_dict = checkpoint["model_state_dict"]
    is_ddp_wrapped = hasattr(model, 'module')
    target_model = model.module if is_ddp_wrapped else model

    if verbose:
        print(f"Model is {'DDP-wrapped' if is_ddp_wrapped else 'not DDP-wrapped'}")

    def _partial_load(into_model, candidate_state_dict):
        model_dict = into_model.state_dict()
        pretrained_dict = {
            k: v for k, v in candidate_state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        if len(pretrained_dict) == 0:
            raise RuntimeError("No matching keys between checkpoint and model")
        if verbose:
            print(f"Matched {len(pretrained_dict)} / {len(candidate_state_dict)} keys from checkpoint")
            for k, v in pretrained_dict.items():
                print(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        into_model.load_state_dict(model_dict)
        print(f"⚠️ Only {len(pretrained_dict)}/{len(model_dict)} of model weights were loaded")

    load_successful = False
    try:
        target_model.load_state_dict(state_dict, strict=strict)
        if verbose:
            print("✓ Loaded checkpoint state dict directly (keys matched)")
        load_successful = True
    except RuntimeError:
        sample_keys = list(state_dict.keys())[:5]
        has_module_prefix = any(key.startswith("module.") for key in sample_keys)

        if has_module_prefix:
            if verbose:
                print("Detected checkpoint with 'module.' prefix, stripping...")
            new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
            try:
                target_model.load_state_dict(new_state_dict, strict=strict)
                if verbose:
                    print("✓ Successfully loaded checkpoint after stripping 'module.' prefix")
                load_successful = True
            except Exception as e2:
                print(f"⚠️ Strict loading failed: {e2}")
                print("Attempting partial loading of matching weights...")
                _partial_load(target_model, new_state_dict)
                load_successful = True
        elif is_ddp_wrapped:
            if verbose:
                print("Trying to add 'module.' prefix to state dict keys...")
            new_state_dict = {f"module.{k}": v for k, v in state_dict.items()}
            try:
                model.load_state_dict(new_state_dict, strict=strict)
                if verbose:
                    print("✓ Successfully loaded checkpoint after adding 'module.' prefix")
                load_successful = True
            except Exception as e2:
                print(f"⚠️ Strict loading failed: {e2}")
                print("Attempting partial loading of matching weights...")
                _partial_load(model, new_state_dict)
                load_successful = True
        else:
            print("Attempting partial loading of matching weights...")
            _partial_load(target_model, state_dict)
            load_successful = True

    if not load_successful:
        raise RuntimeError(f"Failed to load checkpoint into model from {checkpoint_path}")

    for name, obj, key in (("optimizer", optimizer, "optimizer_state_dict"),
                            ("scheduler", scheduler, "scheduler_state_dict"),
                            ("scaler", scaler, "scaler_state_dict")):
        if obj is not None and key in checkpoint:
            try:
                obj.load_state_dict(checkpoint[key])
                if verbose:
                    print(f"✓ Loaded {name} state")
            except Exception as e:
                if verbose:
                    print(f"⚠ Warning: Could not load {name} state: {e}")

    metadata = {
        'epoch': checkpoint.get('epoch'),
        'global_step': checkpoint.get('global_step'),
        'args': checkpoint.get('args'),
        'checkpoint': checkpoint,
    }
    if verbose:
        if metadata['epoch'] is not None:
            print(f"Checkpoint from epoch: {metadata['epoch']}")
        if metadata['global_step'] is not None:
            print(f"Global step: {metadata['global_step']}")

    return metadata, model
