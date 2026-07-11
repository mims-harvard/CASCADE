import pickle

import torch
from tqdm import tqdm


def check_cuda_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        total_memory = torch.cuda.get_device_properties(current_device).total_memory
        allocated_memory = torch.cuda.memory_allocated(current_device)
        cached_memory = torch.cuda.memory_reserved(current_device)

        print(f"GPU: {gpu_name}")
        print(f"Total Memory: {total_memory / 1e9:.5f} GB")
        print(f"Allocated Memory: {allocated_memory / 1e9:.5f} GB")
        print(f"Cached Memory: {cached_memory / 1e9:.5f} GB")
    else:
        print("CUDA is not available.")


def load_and_merge_pickles(directory, path_to_collator, merged_context="disease_cell_type_tissue"):
    """
    Load and merge the chunked pickle files written by `collator.DataCollatorContrastiveLearning`,
    keeping only rows whose `context` matches `merged_context`.

    Args:
        directory: directory containing the collator's chunk pickle files.
        path_to_collator: base path used to derive the chunk filename pattern.
        merged_context: one of 'cell_type', 'disease', 'tissue' (or a merged
            name whose value isn't one of those three, in which case rows are
            kept unfiltered).

    Returns:
        dict: merged, filtered data.
    """
    all_data = {}
    context_value_by_merge = {'cell_type': 'CELLS', 'disease': 'DISEASE', 'tissue': 'TISSUE'}
    context_value = context_value_by_merge.get(merged_context)

    base_name = path_to_collator.stem

    if directory.is_file():
        candidate_files = [directory]
    else:
        candidate_files = [p for p in directory.rglob("*.pkl") if base_name in p.name]

    if not candidate_files:
        print(f"[load_and_merge_pickles] No files found for pattern '{base_name}*.pkl' under {directory}")
        return all_data

    print(f"[load_and_merge_pickles] Found {len(candidate_files)} file(s) matching '{base_name}':")
    for p in candidate_files:
        print("  -", p)

    for pickle_path in tqdm(candidate_files):
        with open(pickle_path, "rb") as f:
            data_dict = pickle.load(f)

        if 'context' in data_dict and isinstance(data_dict['context'], list):
            context_values = data_dict['context']
            if context_value is not None:
                valid_indices = [i for i, val in enumerate(context_values) if str(val).upper() == context_value]

            for key, value in data_dict.items():
                if context_value is not None:
                    if isinstance(value, list):
                        filtered_values = [value[i] for i in valid_indices if i < len(value)]
                    elif isinstance(value, torch.Tensor):
                        filtered_values = value[valid_indices] if len(valid_indices) > 0 else torch.tensor([])
                else:
                    filtered_values = value

                if key not in all_data:
                    all_data[key] = filtered_values
                elif isinstance(all_data[key], torch.Tensor) and isinstance(filtered_values, torch.Tensor):
                    all_data[key] = torch.cat((all_data[key], filtered_values))
                elif isinstance(all_data[key], list) and isinstance(filtered_values, list):
                    all_data[key].extend(filtered_values)

    return all_data
