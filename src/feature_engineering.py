import pandas as pd
import numpy as np
from data_loader import load_data

def create_advanced_features(df):
    """
    Diagnosis reasons of churning
    """
    print("Calculating advanced features...")
    
    # Delivery Experience
    # Actual delivered date - predicted = delay
    time_cols = ['order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in time_cols:
        df[col] = pd.to_datetime(df[col])
        
    df['delay_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
    df['is_late'] = df['delay_days'].apply(lambda x: 1 if x > 0 else 0)

    # Review Score
    reviews = pd.read_csv('data/raw/olist_order_reviews_dataset.csv')
    reviews = reviews[['order_id', 'review_score']]
    
    # calculate the avg review score of the order
    reviews_agg = reviews.groupby('order_id')['review_score'].mean().reset_index()
    df = df.merge(reviews_agg, on='order_id', how='left')

    # Order missing review score = 4, no news is good news
    df['review_score'] = df['review_score'].fillna(4)

# User Level
    features = df.groupby('customer_unique_id').agg({
        'review_score': ['mean', 'min'], 
        'delay_days': 'mean',
        'is_late': 'max',
        'payment_installments': 'mean'
    }).reset_index()
    
    features.columns = [
        'customer_unique_id', 
        'avg_review_score',
        'min_review_score',
        'avg_delivery_delay', 
        'has_late_delivery', 
        'avg_installments'
    ]
    
    print(f"Feature engineering completed with {features.shape[0]} records")
    return features

if __name__ == "__main__":
    df = load_data('data/raw')
    features = create_advanced_features(df)
    print(features.head())
    print(features.describe())