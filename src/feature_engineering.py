import pandas as pd
import numpy as np
from data_loader import load_data

def create_advanced_features(df):
    """
    Calculate behavioral features: 
    1. Delivery Experience (Delay)
    2. Satisfaction (Review Score)
    3. Payment Habits (Installments)
    4. Product Categories (What did they buy?)
    """
    print("Calculating advanced features (Reviews, Delivery, Payment, Categories)...")
    
    # --- 1. Delivery Experience ---
    time_cols = ['order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in time_cols:
        df[col] = pd.to_datetime(df[col])
        
    df['delay_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
    df['is_late'] = df['delay_days'].apply(lambda x: 1 if x > 0 else 0)

    # --- 2. Review Score ---
    reviews = pd.read_csv('data/raw/olist_order_reviews_dataset.csv')
    reviews = reviews[['order_id', 'review_score']]
    # Calculate average score per order first
    reviews_agg = reviews.groupby('order_id')['review_score'].mean().reset_index()
    df = df.merge(reviews_agg, on='order_id', how='left')
    df['review_score'] = df['review_score'].fillna(4)

    # --- 3. Product Categories ---
    print("Processing product categories...")
    try:
        # Load products and translations
        products = pd.read_csv('data/raw/olist_products_dataset.csv')
        translations = pd.read_csv('data/raw/product_category_name_translation.csv')
        
        # Merge translation to get English names
        products = products.merge(translations, on='product_category_name', how='left')
        # Fill missing translations with original name
        products['product_category_name_english'] = products['product_category_name_english'].fillna(products['product_category_name'])
        
        # Merge product info into main dataframe
        # Note: df already has product_id from data_loader logic
        df = df.merge(products[['product_id', 'product_category_name_english']], on='product_id', how='left')
        
        # Fill missing categories with 'other'
        df['product_category_name_english'] = df['product_category_name_english'].fillna('other')
        
        # Identify Top 10 Categories
        top_10_cats = df['product_category_name_english'].value_counts().head(10).index.tolist()
        # print(f"Top 10 Categories found: {top_10_cats}")
        
        # Create Dummy Variables (One-Hot Encoding) for Top 10
        # If user bought 'bed_bath_table', the column 'cat_bed_bath_table' will be 1
        cat_cols = []
        for cat in top_10_cats:
            col_name = f'cat_{cat.lower().replace(" ", "_").replace("&", "_")}'
            df[col_name] = df['product_category_name_english'].apply(lambda x: 1 if x == cat else 0)
            cat_cols.append(col_name)
            
    except Exception as e:
        print(f"Warning: Could not process product categories. Error: {e}")
        cat_cols = []

    # --- 4. Aggregate to User Level ---
    # Define aggregation logic
    agg_dict = {
        'review_score': ['mean', 'min'],
        'delay_days': 'mean',
        'is_late': 'max',
        'payment_installments': 'mean'
    }
    
    # Add category columns to aggregation
    for col in cat_cols:
        agg_dict[col] = 'max'

    features = df.groupby('customer_unique_id').agg(agg_dict).reset_index()
    
    features.columns = [
        'customer_unique_id', 
        'avg_review_score', 
        'min_review_score', 
        'avg_delivery_delay', 
        'has_late_delivery', 
        'avg_installments'
    ] + cat_cols
    
    print(f"Feature engineering completed. Generated {len(features.columns)} features for {features.shape[0]} customers.")
    return features

if __name__ == "__main__":
    df = load_data('data/raw')
    features = create_advanced_features(df)
    print(features.head())
    print(features.columns.tolist())