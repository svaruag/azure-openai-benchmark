import json
import sys

def enhance_jsonl(input_file, output_file):
    enhanced_data = []

    with open(input_file, 'r') as file:
        for line in file:
            entry = json.loads(line.strip())
            
            # Split 'profile' into 'sku', 'profile', 'concurrency'
            sku, profile, concurrency = entry['profile'].split('_')
            concurrency = int(concurrency)
            
            num_replicas = "n/a"
            tensor_parallel = "n/a"
            # Replace 'sku' values
            if sku == 'a100':
                sku = "Standard_NC96ads_A100_v4"
                num_replicas = 4
                tensor_parallel = 1
            elif sku == "8xa100":
                sku = "Standard_NC96amsr_A100_v4"
                num_replicas = 8
                tensor_parallel = 8
            elif sku == 'v100':
                sku = "Standard_ND40rs_v2"
                num_replicas = 4
                tensor_parallel = 2
            elif sku == 'gpt4o':
                sku = "Standard_ND96amsr_A100_v4"
            
            # Add 'num_replicas' and 'tensor_parallel'
            entry['sku'] = sku
            entry['profile'] = profile
            entry['concurrency'] = concurrency
            entry['num_replicas'] = num_replicas
            entry['tensor_parallel'] = tensor_parallel
            
            # Add 'context_tokens' and 'gen_tokens'
            if profile == 'balanced':
                context_tokens = 500
                gen_tokens = 500
            elif profile == 'context':
                context_tokens = 2000
                gen_tokens = 200
            elif profile == 'generation':
                context_tokens = 500
                gen_tokens = 1000
            elif profile == 'custom':
                context_tokens = 500
                gen_tokens = 1500
                
            entry['context_tokens'] = context_tokens
            entry['gen_tokens'] = gen_tokens

            del entry['timestamp']
            enhanced_data.append(entry)
    
    # Write enhanced data back to a new JSONL file
    with open(output_file , 'w') as file:
        for entry in enhanced_data:
            file.write(json.dumps(entry) + '\n')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 enrich_jsonl_results.py <input_file> <output_file>")
        sys.exit(1)
    enhance_jsonl(sys.argv[1], sys.argv[2])
