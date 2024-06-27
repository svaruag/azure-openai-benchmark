#!/bin/bash

# model_name="Phi-3-Medium-4k"
endpoints=(
    # 4k
    "4xa100 https://phi3a100bench.westus3.inference.ml.azure.com/chat/completions phi-3-mini-4k-baseline-opt2 <API_KEY>"
    # "phi3.1rc 4xa100 https://phi3a100bench.westus3.inference.ml.azure.com/chat/completions phi31-mini4k-rc-opt2 <API_KEY>"
    # 128k
    # "4xa100 https://phi3a100bench.westus3.inference.ml.azure.com/chat/completions phi-3-mini-128k-baseline-opt2 <API_KEY>"
    # "4xa100 https://phi3a100bench.westus3.inference.ml.azure.com/chat/completions phi31-mini128k-rc-opt2 <API_KEY>"
)
shape_profiles=("balanced" "context" "generation")
clients=(1 2 4 8 16 32 64)
duration=120
dry_run=false

# Check for --dry-run argument
if [[ "$1" == "--dry-run" ]]; then
    dry_run=true
fi

for endpoint in "${endpoints[@]}"; do
    IFS=' ' read -r -a ep <<< "$endpoint"

    sku="${ep[0]}"
    url="${ep[1]}"
    deployment="${ep[2]}"
    api_key="${ep[3]}"

    for shape_profile in "${shape_profiles[@]}"; do
        output_dir="./current_run/${deployment}_results"
        for client in "${clients[@]}"; do
            output_file="${output_dir}/${sku}_${shape_profile}_${client}.jsonl"
            command="OPENAI_API_KEY=$api_key python -m benchmark.bench load --clients $client --duration $duration --shape-profile $shape_profile --output-format jsonl --output-file $output_file --retry exponential --deployment $deployment $url"

            # if shape is custom, append --context-tokens = 500 and --max-tokens = 1500 to the command
            if [ "$shape_profile" == "custom" ]; then
                command="$command --context-tokens 200 --max-tokens 200"
            fi

            if $dry_run; then
                echo "$command"
            else
                echo "Running load test for SKU: $sku, Shape Profile: $shape_profile, Clients: $client"
                mkdir -p "$output_dir"
                eval "$command"
		sleep 60
            fi
        done
    done
done
