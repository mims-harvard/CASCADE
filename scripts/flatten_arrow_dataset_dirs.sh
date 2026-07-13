#!/bin/bash
# Flattens HuggingFace `Dataset.save_to_disk()` output (Methods 9.4): each
# tokenized shard is saved as its own directory (data-<n>-of-<n>/data-00000-of-00001.arrow);
# this collapses each shard directory into a single flat <shard>.arrow file so the
# tokenized dataset can be loaded as a plain directory of .arrow files. Run from
# inside the directory containing the data-*-of-*/ shard directories.
#
# Usage:
#   cd $CASCADE_DATA_ROOT/processed/LUCA/tokenized_data_CELLS_out-ref
#   bash flatten_arrow_dataset_dirs.sh

for dir in data-*-of-*; do
    if [[ -d "$dir" ]]; then
        old_file="$dir/data-00000-of-00001.arrow"
        temp_file="${dir%/}.renamed.arrow"
        final_file="${dir%/}.arrow"

        if [[ -f "$old_file" ]]; then
            mv "$old_file" "$temp_file"
            rm -rf "$dir"
            mv "$temp_file" "$final_file"
            echo "Moved and renamed: $old_file -> $final_file, and deleted $dir"
        else
            echo "File not found in: $dir"
        fi
    fi
done

for file in *.arrow.arrow; do
    [[ -e "$file" ]] || continue
    mv "$file" "${file%.arrow}"
done
