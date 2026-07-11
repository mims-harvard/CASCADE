"""
Incrementally merges the chunked embedding pickles written by
get_embeddings_parallel.py, without holding every chunk in memory at once -
needed for datasets too large to load as a single collator (e.g. SEATTLE).
"""
import gc
import glob
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def load_embeddings_from_subset(subset_dir, donors_split=None):
    """
    Load embeddings saved by a donor-subsetting preprocessing step, expecting:
        subset_dir/embeddings/donor_{id}.pt
        subset_dir/metadata/donor_{id}.pkl
    """
    subset_path = Path(subset_dir)
    embeddings_dir = subset_path / "embeddings"
    metadata_dir = subset_path / "metadata"

    if not embeddings_dir.exists():
        raise ValueError(f"Embeddings directory not found: {embeddings_dir}")
    if not metadata_dir.exists():
        raise ValueError(f"Metadata directory not found: {metadata_dir}")

    embedding_files = sorted(embeddings_dir.glob("donor_*.pt"))
    if not embedding_files:
        raise ValueError(f"No donor embedding files found in {embeddings_dir}")

    def extract_donor_id(filepath):
        donor_str = filepath.stem.replace("donor_", "")
        try:
            return int(donor_str)
        except ValueError:
            try:
                return float(donor_str)
            except ValueError:
                return donor_str

    available_donors = [extract_donor_id(f) for f in embedding_files]
    if donors_split is not None:
        all_split_donors = set(donors_split.get('train_donors', [])) | set(donors_split.get('test_donors', []))
        donors_to_load = [d for d in available_donors if d in all_split_donors]
        print(f"  Filtering to {len(donors_to_load)} donors from split (out of {len(available_donors)} available)")
    else:
        donors_to_load = available_donors

    all_embeddings = []
    all_metadata = defaultdict(list)
    total_cells = 0

    for i, donor_id in enumerate(donors_to_load, 1):
        emb_file = embeddings_dir / f"donor_{donor_id}.pt"
        meta_file = metadata_dir / f"donor_{donor_id}.pkl"
        if not emb_file.exists():
            print(f"  ⚠️ Warning: Embedding file not found for donor {donor_id}")
            continue

        donor_emb = torch.load(emb_file, weights_only=False)
        n_cells = len(donor_emb)
        total_cells += n_cells
        all_embeddings.append(donor_emb)

        if meta_file.exists():
            with open(meta_file, 'rb') as f:
                donor_meta = pickle.load(f)
            for key, value in donor_meta.items():
                if isinstance(value, (np.ndarray, list)) and len(value) == n_cells:
                    all_metadata[key].append(np.asarray(value))

        if i % 25 == 0 or i == len(donors_to_load):
            print(f"  Loaded {i}/{len(donors_to_load)} donors ({total_cells} cells)")

    if not all_embeddings:
        raise ValueError("No embeddings were loaded from the subset directory")

    cell_data = {'embedding': torch.cat(all_embeddings, dim=0)}
    for key, values in all_metadata.items():
        try:
            cell_data[key] = np.concatenate(values, axis=0)
        except ValueError:
            cell_data[key] = np.array(values, dtype=object)

    print(f"✅ Loaded {total_cells} cells across {len(all_embeddings)} donors")
    return cell_data


def _merge_values(v1, v2, keypath=""):
    if isinstance(v1, list) and isinstance(v2, list):
        return v1 + v2
    if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
        return np.concatenate([v1, v2], axis=0)
    if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
        return torch.cat([v1, v2], dim=0)
    if isinstance(v1, dict) and isinstance(v2, dict):
        return _merge_dicts_same_keys(v1, v2, parent_keypath=keypath)
    try:
        if v1 == v2:
            return v1
    except Exception:
        pass
    raise TypeError(f"Cannot merge at {keypath}: {type(v1)} vs {type(v2)}")


def _merge_dicts_same_keys(d1, d2, parent_keypath=""):
    out = {}
    for k in set(d1.keys()) | set(d2.keys()):
        kp = f"{parent_keypath}.{k}" if parent_keypath else k
        out[k] = _merge_values(d1[k], d2[k], keypath=kp)
    return out


def _init_merge_buffer(obj):
    """Recursive merge buffer that defers large concatenations until finalize."""
    if isinstance(obj, dict):
        return ("dict", {k: _init_merge_buffer(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return ("list", [obj])
    if isinstance(obj, np.ndarray):
        return ("ndarray", [obj])
    if isinstance(obj, torch.Tensor):
        return ("tensor", [obj])
    return ("scalar", obj)


def _append_to_merge_buffer(buffer, obj, keypath=""):
    kind, payload = buffer
    if kind == "dict":
        if not isinstance(obj, dict):
            raise TypeError(f"Cannot merge at {keypath}: dict vs {type(obj)}")
        for k, v in obj.items():
            kp = f"{keypath}.{k}" if keypath else k
            payload[k] = _append_to_merge_buffer(payload[k], v, keypath=kp) if k in payload else _init_merge_buffer(v)
        return ("dict", payload)
    if kind == "list":
        if not isinstance(obj, list):
            raise TypeError(f"Cannot merge at {keypath}: list vs {type(obj)}")
        payload.append(obj)
        return ("list", payload)
    if kind in ("ndarray", "tensor"):
        payload.append(obj)
        return (kind, payload)
    return buffer


def _finalize_merge_buffer(buffer):
    kind, payload = buffer
    if kind == "dict":
        return {k: _finalize_merge_buffer(v) for k, v in payload.items()}
    if kind == "list":
        if len(payload) == 1:
            return payload[0]
        out = []
        for lst in payload:
            out.extend(lst)
        return out
    if kind == "ndarray":
        return payload[0] if len(payload) == 1 else np.concatenate(payload, axis=0)
    if kind == "tensor":
        return payload[0] if len(payload) == 1 else torch.cat(payload, dim=0)
    return payload


def load_chunked_embeddings(chunk_dir, chunk_prefix):
    """Load and merge chunked embedding pickles (pattern: {chunk_prefix}_*_chunk_*.pkl),
    merging incrementally to bound peak memory on large datasets."""
    chunk_dir = Path(chunk_dir)
    pattern = str(chunk_dir / f"{chunk_prefix}_*_chunk_*.pkl")
    all_files = glob.glob(pattern)
    if not all_files:
        raise RuntimeError(f"No chunk files found matching pattern: {pattern}")

    print(f"Discovered {len(all_files)} chunk files in {chunk_dir}")

    chunk_regex = re.compile(r"chunk_(\d+)\.pkl$")
    files_by_chunk = defaultdict(list)
    for f in all_files:
        m = chunk_regex.search(f)
        if not m:
            print(f"Skipping file without chunk id: {f}")
            continue
        files_by_chunk[int(m.group(1))].append(f)

    chunk_ids_sorted = sorted(files_by_chunk.keys())
    print(f"Found chunks: {chunk_ids_sorted}")

    merge_buffer = None
    for chunk_id in chunk_ids_sorted:
        for f in sorted(files_by_chunk[chunk_id]):
            with open(f, "rb") as handle:
                obj = pickle.load(handle)
            merge_buffer = _init_merge_buffer(obj) if merge_buffer is None else _append_to_merge_buffer(merge_buffer, obj)
            del obj
        gc.collect()

    print("✅ Successfully loaded and merged chunked embeddings")
    return _finalize_merge_buffer(merge_buffer)
