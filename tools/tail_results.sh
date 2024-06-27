#!/bin/bash
# Source and target directories
src_dir="/home/svaruag/repos/azure-openai-benchmark/Phi-3-Medium-4k_results_8xa100"
target_dir="/home/svaruag/repos/azure-openai-benchmark/Phi-3-Medium-4k_results_8xa100_tailed"

# Create target directory if it doesn't exist
mkdir -p "$target_dir"

# Loop over each jsonl file in the source directory
for file in "$src_dir"/*.jsonl; do
    # Extract the last line and save it to the corresponding file in the target directory
    tail -n 1 "$file" > "$target_dir/$(basename "$file")"
done
