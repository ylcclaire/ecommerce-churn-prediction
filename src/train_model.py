import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from data_loader import load_data
from rfm_calculation import calculate_rfm
from feature_engineering import create_advanced_features

def train_churn_model():
    print("Load Raw Data")
    df = load_data('data/raw')
    
    print("Calculate RFM")
    rfm = calculate_rfm(df)
    
    print("Execute advanced feature engineering")
    features = create_advanced_features(df)

    print("Merge Data")
    final_df = rfm.merge(features, on='customer_unique_id', how='left')
    
    # Data Cleaning: if delay is null, input 0; no installments = 1
    final_df['avg_delivery_delay'] = final_df['avg_delivery_delay'].fillna(0)
    final_df['has_late_delivery'] = final_df['has_late_delivery'].fillna(0)
    final_df['avg_installments'] = final_df['avg_installments'].fillna(1)
    
    # Target Label
    # Recency > 180 days = 1 (churn)
    # Recency <= 180 days = 0 (active)
    churn_threshold = 180
    final_df['is_churn'] = final_df['Recency'].apply(lambda x: 1 if x > churn_threshold else 0)
    
    print(f"Churn definition: > {churn_threshold} days no purchase")
    print(f"Churn rate: {final_df['is_churn'].mean():.2%}")
    
    # Prepare training data
    X = final_df.drop(['customer_unique_id', 'is_churn', 'Recency'], axis=1)
    y = final_df['is_churn']
    
    print("Training features:", X.columns.tolist())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Calculate scale_pos_weight for Class Imbalance
    # Target to down-weight the majority class (Churn=1) or up-weight minority (Active=0)
    # Standard formula: sum(negative) / sum(positive) to pay more attention to the 'Active' users (Class 0)
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_weight = n_neg / n_pos

    # Model: XGBoost
    print("Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        scale_pos_weight=scale_weight,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Model evaluation
    print("Model evaluation")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nreport:\n", classification_report(y_test, y_pred))
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n========= Reason of churning (XGBoost) =========")
    print(feature_importance)
    
    return model

if __name__ == "__main__":
    train_churn_model()