import pandas as pd
import numpy as np
from data_loader import load_data

def calculate_rfm(df):
    print("calculating RFM")

    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

    # Set Snapshot Date
    snapshot_date = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    print(f"Snapshot Date is: {snapshot_date}")

    # Calculate RMF
    rfm = df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
        'order_id': 'count',
        'price': 'sum'
    })

    rfm.rename(columns={
        'order_purchase_timestamp': 'Recency',
        'order_id': 'Frequency',
        'price': 'Monetary'
    }, inplace=True)

    # Fix Frequency
    real_frequency = df.groupby('customer_unique_id')['order_id'].nunique()
    rfm['Frequency'] = real_frequency

    print(f"RFM calculated！We have {rfm.shape[0]} unique customers。")
    return rfm

if __name__ == "__main__":
    # load data
    try:
        df = load_data('../data/raw')
    except FileNotFoundError:
        df = load_data('data/raw')
    
    rfm_df = calculate_rfm(df)
    
    # check first 5 row of data
    print(rfm_df.head())
    
    print("\n Describe data:")
    print(rfm_df.describe())