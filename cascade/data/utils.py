import pickle
from pathlib import Path

import torch
from datasets import Dataset, Sequence, Value, concatenate_datasets
from tqdm import tqdm

# Covariate columns that get inconsistently inferred as int/float vs. string across
# per-context tokenization shards (Methods 9.4); always cast to string so shards can
# be concatenated. Harmless no-op for datasets/shards that don't have these columns.
_COVARIATE_COLUMNS_TO_STRING = ["Cre", "THR_expr", "THR", "batch", "treatment"]


def load_and_shuffle_tokenized_dataset(list_data):
    """
    Loads every `.arrow` shard under each directory in `list_data` (the output of
    `cascade.data.tokenizer.TranscriptomeTokenizer`, e.g. one directory per batched
    tokenization run) directly via `Dataset.from_file`, aligns dtypes that can drift
    between shards, concatenates them, and shuffles. This is the `--list_data` input
    to `cascade.training.train_ddp`.

    Args:
        list_data: a directory path, or list of directory paths (each containing
            `.arrow` shard files), e.g. `args.list_data` from `train_ddp.py`.

    Returns:
        datasets.Dataset: the concatenated, shuffled dataset.
    """
    if not isinstance(list_data, list):
        list_data = [list_data]

    arrow_files = []
    for path in list_data:
        arrow_files.extend(
            str(p) for p in Path(path).iterdir() if "arrow" in str(p) and "cache" not in str(p)
        )
    print(f"[load_and_shuffle_tokenized_dataset] Found {len(arrow_files)} shard(s)")

    shards = []
    common_columns = None
    for arrow_file in arrow_files:
        ds = Dataset.from_file(arrow_file)

        if "input_ids" in ds.column_names:
            inner_dtype = getattr(getattr(ds.features["input_ids"], "feature", None), "dtype", None)
            if inner_dtype != "int32":
                ds = ds.cast_column("input_ids", Sequence(Value("int32")))

        for col in _COVARIATE_COLUMNS_TO_STRING:
            if col in ds.column_names and getattr(ds.features[col], "dtype", None) != "string":
                ds = ds.cast_column(col, Value("string"))

        shards.append(ds)
        common_columns = set(ds.column_names) if common_columns is None else common_columns & set(ds.column_names)

    # Some shards may be missing columns others have (e.g. if tokenization ran in
    # separate chunks); restrict to the columns common to every shard before concatenating.
    shards = [ds.select_columns(list(common_columns)) for ds in shards]
    return concatenate_datasets(shards).shuffle()


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
