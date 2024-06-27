import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metric(df, metric, y_label, file_name, y_scale=1):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='concurrency', y=metric, hue='profile', marker='o')
    plt.xlabel('Concurrency')
    plt.ylabel(y_label)
    plt.title(f'{y_label} vs. Concurrency')
    plt.legend(title='Profile')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

def main(input_file, output_dir):
    # Create directories for plots
    os.makedirs(output_dir, exist_ok=True)

    # Load data from JSONL file
    data = []
    with open(input_file, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Convert TTFT and TBT to milliseconds
    df['ttft_avg_ms'] = df['ttft'].apply(lambda x: x['avg'] * 1000)
    df['tbt_avg_ms'] = df['tbt'].apply(lambda x: x['avg'] * 1000)
    df['e2e_avg'] = df['e2e'].apply(lambda x: x['avg'])
    df['tpm.context'] = df['tpm'].apply(lambda x: x['context'])
    df['tpm.gen'] = df['tpm'].apply(lambda x: x['gen'])

    plot_metric(df, 'e2e_avg', 'E2E Request Latency (Seconds)', os.path.join(output_dir, 'e2e_latency.png'))
    plot_metric(df, 'ttft_avg_ms', 'TTFT (Milliseconds)', os.path.join(output_dir, 'ttft.png'))
    plot_metric(df, 'tbt_avg_ms', 'TBT (Milliseconds)', os.path.join(output_dir, 'tbt.png'))
    plot_metric(df, 'completed', 'Queries Completed Per Minute', os.path.join(output_dir, 'completed.png'))
    plot_metric(df, 'tpm.context', 'Context Tokens Per Minute', os.path.join(output_dir, 'context_tpm.png'))
    plot_metric(df, 'tpm.gen', 'Generated Tokens Per Minute', os.path.join(output_dir, 'gen_tpm.png'))

    print(f"Plots generated and saved in {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 gen_plots.py <input_file> <output_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])