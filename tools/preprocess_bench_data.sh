#!/bin/bash
set -e


# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <source_directory> <output_file> <plots_output_dir>"
    exit 1
fi

# Source and target directories
src_dir="$1"
tail_dir="/tmp/${src_dir}_tailed"
profile_dir="${tail_dir}_profiled"
output_file="$2"
plots_output_dir="$3"

# Create target directories if they don't exist
mkdir -p "$tail_dir"
mkdir -p "$profile_dir"

# Step 1: Tail the results
for file in "$src_dir"/*.jsonl; do
    tail -n 1 "$file" > "$tail_dir/$(basename "$file")"
done
echo "Tailed results saved to $tail_dir"

# Step 2: Add profile key
for file in "$tail_dir"/*.jsonl; do
    filename=$(basename "$file" .jsonl)
    while IFS= read -r line; do
        echo "{\"profile\": \"$filename\", ${line:1}" >> "$profile_dir/$(basename "$file")"
    done < "$file"
done
echo "Profiled results saved to $profile_dir"

# Step 3: Concatenate JSONL files
cat "$profile_dir"/*.jsonl > /tmp/draft_results.jsonl
echo "Concatenated results saved to /tmp/draft_results.jsonl"

# Step 4: Enhance JSONL data
python3 tools/enrich_jsonl_results.py /tmp/draft_results.jsonl "$output_file"
echo "Enhanced results saved to $output_file"

# Step 5: Generate plots
echo "Generating plots, command: python3 tools/gen_plots.py $output_file $plots_output_dir"
python3 tools/gen_plots.py "$output_file" "$plots_output_dir"
echo "Plots saved to $plots_output_dir"

# Clean up intermediate files
rm -rf "$tail_dir" "$profile_dir" /tmp/draft_results.jsonl
echo "Intermediate files cleaned up"

echo "Preprocessing complete. Final results saved to $output_file" and plots saved to $plots_output_dir
