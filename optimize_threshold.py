import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engineering import SpotifyFeatureEngineer

def calculate_streaming_roi(y_true, y_prob, threshold, 
                            annual_sub_val=120.0, 
                            offer_cost=15.0, 
                            save_rate=0.40):
    preds = (y_prob >= threshold).astype(int)
    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    
    saved_rev = tp * (save_rate * annual_sub_val)
    cost = (tp + fp) * offer_cost
    return saved_rev - cost

def run_threshold_check():
    data_path = os.path.join("data", "spotify_churn_dataset.csv")
    if not os.path.exists(data_path):
        data_path = "spotify_churn_dataset.csv"

    df = pd.read_csv(data_path)
    y = df['is_churned'].astype(int)
    X = df.drop(columns=['is_churned', 'user_id'], errors='ignore')

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    pipeline = joblib.load("models/churn_pipeline.joblib")
    probs = pipeline.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.1, 0.9, 81)
    profits = [calculate_streaming_roi(y_test.values, probs, t) for t in thresholds]

    best_idx = np.argmax(profits)
    best_t = thresholds[best_idx]
    
    print(f"Optimal Probability Cutoff: {best_t:.2f}")
    print(f"Projected Net Campaign Return: ${profits[best_idx]:,.2f}")

    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, profits, color="#1DB954", lw=2.5)
    plt.axvline(best_t, color='black', linestyle='--', label=f'Optimal Cutoff ({best_t:.2f})')
    plt.title("Retention Campaign Net Profit vs. Decision Threshold")
    plt.xlabel("Probability Cutoff Threshold")
    plt.ylabel("Net Profit ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("models/threshold_curve.png", dpi=300)
    print("Saved plot to 'models/threshold_curve.png'")

if __name__ == "__main__":
    run_threshold_check()