import pandas as pd
import os

def load_data(data_path):
    print("loading data...")
    
    # path
    orders_path = os.path.join(data_path, 'olist_orders_dataset.csv')
    items_path = os.path.join(data_path, 'olist_order_items_dataset.csv')
    customers_path = os.path.join(data_path, 'olist_customers_dataset.csv')
    payments_path = os.path.join(data_path, 'olist_order_payments_dataset.csv')

    # read csv
    orders = pd.read_csv(orders_path)
    items = pd.read_csv(items_path)
    customers = pd.read_csv(customers_path)
    payments = pd.read_csv(payments_path)

    # 轉換時間格式 (這是數據清洗最重要的一步)
    time_cols = ['order_purchase_timestamp', 'order_approved_at', 
                 'order_delivered_carrier_date', 'order_delivered_customer_date', 
                 'order_estimated_delivery_date']
    for col in time_cols:
        orders[col] = pd.read_csv(orders_path, usecols=[col])[col]
        orders[col] = pd.to_datetime(orders[col])

    print("Merging data...")
    
    df = orders.merge(items, on='order_id', how='left')
    payments = payments.drop_duplicates(subset=['order_id']) 
    df = df.merge(payments, on='order_id', how='left')
    df = df.merge(customers, on='customer_id', how='left')

    print(f"data loaded！{df.shape[0]} record in total")
    return df

if __name__ == "__main__":
    df = load_data('data/raw')
    print(df.head())