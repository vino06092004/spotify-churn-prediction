import os
import joblib
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from xgboost import XGBClassifier

from feature_engineering import SpotifyFeatureEngineer, build_spotify_preprocessor

def run_training():
    # 1. Locate dataset
    data_path = os.path.join("data", "spotify_churn_dataset.csv")
    if not os.path.exists(data_path):
        data_path = "spotify_churn_dataset.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not locate dataset at {data_path}")

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # 2. Clean target & drop non-predictive identifiers
    target_col = 'is_churned'
    if df[target_col].dtype == 'object':
        df[target_col] = df[target_col].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0}).fillna(0)
    
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col, 'user_id'], errors='ignore')

    # 3. Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 4. Handle positive class weighting for churn imbalance
    pos_scale = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-5)

    pipeline = Pipeline([
        ('fe', SpotifyFeatureEngineer()),
        ('prep', build_spotify_preprocessor()),
        ('clf', XGBClassifier(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_scale,
            eval_metric='logloss',
            random_state=42
        ))
    ])

    print("Training XGBoost churn prediction pipeline...")
    pipeline.fit(X_train, y_train)

    # 5. Model Evaluation
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = pipeline.predict(X_test)

    print("\n" + "="*45)
    print("MODEL PERFORMANCE ON TEST SET")
    print("="*45)
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
    print(f"PR-AUC Score:  {average_precision_score(y_test, probs):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, preds))

    # 6. Fit SHAP TreeExplainer
    print("Fitting SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(pipeline.named_steps['clf'])

    # 7. Save Artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, os.path.join("models", "churn_pipeline.joblib"))
    joblib.dump(explainer, os.path.join("models", "explainer.joblib"))
    print("\nModel pipeline and SHAP explainer saved to 'models/' folder successfully.")

if __name__ == "__main__":
    run_training()