#!/usr/bin/env python3
"""
Flattens the per-context-value batch directories written by
`cascade.data.tokenizer.TranscriptomeTokenizer.save_batch` (Methods 9.4): each batch
is saved under `<context_value><batch_index>/<context>.dataset/`, containing one
`.arrow` file; this moves each `.arrow` file up into the top-level context directory,
prefixed with its context-value subfolder name, so all batches for a context end up
as sibling files instead of nested one level per batch.

Usage:
    python -m preprocessing.rename_tokenized_chunks --dataset-dir $CASCADE_DATA_ROOT/processed/LUCA/CELLS
"""
import argparse
import os
import shutil
from pathlib import Path


def move_and_rename_arrow_files(dataset_dir, context_suffix=".dataset"):
    dataset_dir = Path(dataset_dir)
    for root, _dirs, files in os.walk(dataset_dir):
        root = Path(root)
        if not root.name.endswith(context_suffix):
            continue
        # The context-value + batch-index subfolder name, two levels up from
        # <context>.dataset/, e.g. "normal3" for the 3rd batch of "normal" cells.
        batch_name = root.parent.name

        for file in files:
            if not file.endswith(".arrow"):
                continue
            new_filename = f"{batch_name}_{file}"
            src_path = root / file
            dest_path = dataset_dir / new_filename

            if dest_path.exists():
                print(f"Skipping (already exists): {dest_path}")
                continue
            try:
                shutil.move(str(src_path), str(dest_path))
                print(f"Moved: {src_path} -> {dest_path}")
            except OSError as e:
                print(f"Error moving {src_path} -> {dest_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True,
                         help="Top-level context directory, e.g. $CASCADE_DATA_ROOT/processed/LUCA/CELLS")
    args = parser.parse_args()
    move_and_rename_arrow_files(args.dataset_dir)


if __name__ == "__main__":
    main()
