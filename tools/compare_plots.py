import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# output_path = "./plots/comparison-fmi35-fmi42-phi3"
output_path = "./plots/comparison-phi3-4k-eosignored"
os.makedirs(output_path, exist_ok=True)

# Load data from JSONL file
phi3_file_name = "/home/svaruag/repos/azure-openai-benchmark/phi-3-mini-4k-results.jsonl"
phi31_file_name = "/home/svaruag/repos/azure-openai-benchmark/phi-3.1-mini-4k-results.jsonl"
phi3_data = []
with open(phi3_file_name, 'r') as file:
    for line in file:
        phi3_data.append(json.loads(line))

gpt4o_data = []
with open(phi31_file_name, 'r') as file:
    for line in file:
        gpt4o_data.append(json.loads(line))

# Convert to DataFrame
phi3_df = pd.DataFrame(phi3_data)
phi31_df = pd.DataFrame(gpt4o_data)

# Convert TTFT and TBT to milliseconds
phi3_df['ttft_avg_ms'] = phi3_df['ttft'].apply(lambda x: x['avg'] * 1000)
phi3_df['tbt_avg_ms'] = phi3_df['tbt'].apply(lambda x: x['avg'] * 1000)
phi3_df['e2e_avg'] = phi3_df['e2e'].apply(lambda x: x['avg'])
phi3_df['tpm.context'] = phi3_df['tpm'].apply(lambda x: x['context'])
phi3_df['tpm.gen'] = phi3_df['tpm'].apply(lambda x: x['gen'])
# phi3_df = phi3_df[phi3_df['concurrency'] <= 64] # filter out rows > 64 concurrency

phi31_df['ttft_avg_ms'] = phi31_df['ttft'].apply(lambda x: x['avg'] * 1000)
phi31_df['tbt_avg_ms'] = phi31_df['tbt'].apply(lambda x: x['avg'] * 1000)
phi31_df['e2e_avg'] = phi31_df['e2e'].apply(lambda x: x['avg'])
phi31_df['tpm.context'] = phi31_df['tpm'].apply(lambda x: x['context'])
phi31_df['tpm.gen'] = phi31_df['tpm'].apply(lambda x: x['gen'])
# phi31_df = phi31_df[phi31_df['concurrency'] <= 64]  # filter out rows > 64 concurrency

# Define plot function
def plot_metric(phi3_df, phi31_df, metric, y_label, file_name):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=phi3_df, x='concurrency', y=metric, hue='profile', style='profile', markers=True, dashes=False, linewidth=2.5, errorbar=None)
    
    # Generate dash patterns for phi31 profiles
    unique_profiles = phi31_df['profile'].unique()
    dash_patterns = [(2, 2)] * len(unique_profiles)
    
    sns.lineplot(data=phi31_df, x='concurrency', y=metric, hue='profile', style='profile', markers=True, dashes=dash_patterns, linewidth=1.5, errorbar=None)
    
    plt.xlabel('Concurrency')
    plt.ylabel(y_label)
    plt.title(f'{y_label} vs. Concurrency')
    handles, labels = plt.gca().get_legend_handles_labels()
    # phi3_labels = [f'phi3-FMI:42 - {label}' for label in phi3_df['profile'].unique()]
    # phi31_labels = [f'phi3-FMI:35 - {label}' for label in phi31_df['profile'].unique()]
    phi3_labels = [f'phi-3-mini-4k - {label}' for label in phi3_df['profile'].unique()]
    phi31_labels = [f'phi-3.1-mini-4k - {label}' for label in phi31_df['profile'].unique()]
    
    plt.legend(handles=handles[:len(phi3_labels)] + handles[len(phi3_labels):], 
               labels=phi3_labels + phi31_labels,
               title='Profile')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

# Plot combined metrics
plot_metric(phi3_df, phi31_df, 'e2e_avg', 'E2E Request Latency (Seconds)', f'{output_path}/e2e_latency.png')
plot_metric(phi3_df, phi31_df, 'ttft_avg_ms', 'TTFT (Milliseconds)', f'{output_path}/ttft.png')
plot_metric(phi3_df, phi31_df, 'tbt_avg_ms', 'TBT (Milliseconds)', f'{output_path}/tbt.png')
plot_metric(phi3_df, phi31_df, 'completed', 'Queries Completed Per Minute', f'{output_path}/completed.png')
plot_metric(phi3_df, phi31_df, 'tpm.context', 'Context Tokens Per Minute', f'{output_path}/context_tpm.png')
plot_metric(phi3_df, phi31_df, 'tpm.gen', 'Generated Tokens Per Minute', f'{output_path}/gen_tpm.png')

print(f"Plots generated and saved in {output_path}")