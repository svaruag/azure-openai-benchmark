#!/bin/bash
# Source and target directories
src_dir="/home/svaruag/repos/azure-openai-benchmark/Phi-3-Medium-4k_results_8xa100_tailed"
target_dir="/home/svaruag/repos/azure-openai-benchmark/Phi-3-Medium-4k_results_8xa100_tailed_profiled"

# Create target directory if it doesn't exist
mkdir -p "$target_dir"

# Loop over each jsonl file in the source directory
for file in "$src_dir"/*.jsonl; do
    filename=$(basename "$file" .jsonl)

    # Process each line in the file, adding the profile key
    while IFS= read -r line; do
        echo "{\"profile\": \"$filename\", ${line:1}" >> "$target_dir/$(basename "$file")"
    done < "$file"
done
