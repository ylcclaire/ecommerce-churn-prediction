import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from data_loader import load_data
from rfm_calculation import calculate_rfm
from feature_engineering import create_advanced_features

def check_interaction():
    print("Step 1: Preparing Data...")
    df = load_data('data/raw')
    rfm = calculate_rfm(df)
    features = create_advanced_features(df)
    final_df = rfm.merge(features, on='customer_unique_id', how='left')
    
    # Fill NA & Target
    final_df['avg_delivery_delay'] = final_df['avg_delivery_delay'].fillna(0)
    final_df['has_late_delivery'] = final_df['has_late_delivery'].fillna(0)
    final_df['avg_installments'] = final_df['avg_installments'].fillna(1)
    churn_threshold = 180
    final_df['is_churn'] = final_df['Recency'].apply(lambda x: 1 if x > churn_threshold else 0)
    
    X = final_df.drop(['customer_unique_id', 'is_churn', 'Recency'], axis=1)
    y = final_df['is_churn']
    
    # Train Model
    print("Step 2: Training XGBoost...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(n_estimators=100, max_depth=5, scale_pos_weight=1/scale_weight, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # SHAP
    print("Step 3: Calculating Interaction Values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    print("Step 4: Generating Interaction Plot...")
    
    target_feature = 'avg_review_score'
    
    if 'cat_furniture_decor' in X.columns:
        color_feature = 'cat_furniture_decor'
        print(f"Coloring by: {color_feature}")
    else:
        color_feature = 'Monetary' 
        print(f"Furniture category not found in Top 10, utilizing {color_feature} as proxy for durable goods.")

    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        target_feature, 
        shap_values, 
        X_test, 
        interaction_index=color_feature,
        show=False
    )
    plt.title(f"Interaction: {target_feature} vs. {color_feature}", fontsize=14)
    plt.tight_layout()
    plt.savefig('reports/figures/shap_interaction_review_furniture.png')
    print("✅ Saved: reports/figures/shap_interaction_review_furniture.png")

if __name__ == "__main__":
    check_interaction()