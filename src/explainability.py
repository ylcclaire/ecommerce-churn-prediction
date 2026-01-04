import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from data_loader import load_data
from rfm_calculation import calculate_rfm
from feature_engineering import create_advanced_features

def explain_model():
    # --- 1. Prep Data ---
    print("Step 1: Preparing Data...")
    df = load_data('data/raw')
    rfm = calculate_rfm(df)
    features = create_advanced_features(df)
    final_df = rfm.merge(features, on='customer_unique_id', how='left')
    final_df['avg_delivery_delay'] = final_df['avg_delivery_delay'].fillna(0)
    final_df['has_late_delivery'] = final_df['has_late_delivery'].fillna(0)
    final_df['avg_installments'] = final_df['avg_installments'].fillna(1)
    
    # Define Churn
    churn_threshold = 180
    final_df['is_churn'] = final_df['Recency'].apply(lambda x: 1 if x > churn_threshold else 0)
    
    X = final_df.drop(['customer_unique_id', 'is_churn', 'Recency'], axis=1)
    y = final_df['is_churn']
    
    # --- 2. Retraining ---
    print("Step 2: Retraining XGBoost for explanation...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=1/scale_weight,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # --- 3. Calculate SHAP value ---
    print("Step 3: Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    print("Step 4: Generating plots for your 3 questions...")
    
    # ---------------------------------------------------------
    # Fig 1: Which 3 Category affect churning?
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("Summary Plot: Top Categories & Impact Direction", fontsize=16)
    plt.tight_layout()
    plt.savefig('reports/figures/shap_summary_categories.png')
    print("✅ Saved Plot 1: reports/figures/shap_summary_categories.png")
    plt.close()

    # ---------------------------------------------------------
    # How Logistic affect churning
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        'avg_delivery_delay', 
        shap_values, 
        X_test, 
        interaction_index=None,
        show=False,
        xmin=-50, xmax=50
    )
    plt.title("Impact of Delivery Delay on Churn", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/figures/shap_dependence_delay.png')
    print("✅ Saved Plot 2: reports/figures/shap_dependence_delay.png")
    plt.close()

    # ---------------------------------------------------------
    # Fig 3: How review score affect churning?
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        'avg_review_score', 
        shap_values, 
        X_test, 
        interaction_index=None,
        show=False
    )
    plt.title("Impact of Review Score on Churn", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/figures/shap_dependence_review.png')
    print("✅ Saved Plot 3: reports/figures/shap_dependence_review.png")
    plt.close()
    
    print("\nAll plots generated! Check the 'reports/figures' folder.")

if __name__ == "__main__":
    explain_model()