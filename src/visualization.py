import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from rfm_calculation import calculate_rfm
from data_loader import load_data

sns.set(style="whitegrid")

def plot_rfm_distributions(rfm_df, save_dir='reports/figures'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print("Generating Recency graph...")
    
    # Recency graph
    plt.figure(figsize=(10, 6))
    sns.histplot(rfm_df['Recency'], bins=50, kde=True, color='#4A90E2')
    plt.title('Recency Distribution (Days since last purchase)')
    plt.xlabel('Days')
    plt.ylabel('Number of Customers')
    # 180 days reference line
    plt.axvline(x=180, color='r', linestyle='--', label='180 Days')
    plt.legend()
    plt.savefig(f'{save_dir}/recency_distribution.png')
    plt.close()

    print("Generating Frequency graph...")
    
    # Frequency graph
    # Since most people purchase once only, removing extreme value
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Frequency', data=rfm_df[rfm_df['Frequency'] <= 5], palette='viridis')
    plt.title('Frequency Distribution (Customers with <= 5 purchases)')
    plt.xlabel('Number of Purchases')
    plt.ylabel('Count')
    plt.savefig(f'{save_dir}/frequency_distribution.png')
    plt.close()

    print(f"graph have bveen saved to {save_dir} ")

if __name__ == "__main__":
    df = load_data('data/raw')
    rfm_df = calculate_rfm(df)
    
    plot_rfm_distributions(rfm_df)