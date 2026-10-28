# 📉 End-to-End Customer Churn Prediction & Retention System

An enterprise machine learning pipeline built to identify customer churn risks, explain individual customer churn drivers using SHAP, and optimize proactive retention campaigns based on financial ROI.

---

## 📌 Project Overview
Customer acquisition costs significantly outpace retention expenses. This project delivers an automated framework to:
1. **Detect early attrition signals** from billing patterns and service usage contracts.
2. **Handle extreme target imbalance** via positive scale weighting and PR-AUC calibration.
3. **Deconstruct complex predictions** into transparent, human-readable risk drivers via Tree SHAP.
4. **Deploy a real-time web application** for single-account evaluations and bulk CSV scoring.

---

## 🛠️ Architecture & Tech Stack
- **Modeling Engine:** Scikit-Learn, XGBoost
- **Explainability (XAI):** SHAP (SHapley Additive exPlanations)
- **Data Engineering:** Pandas, NumPy
- **Interactive Interface:** Streamlit, Matplotlib
- **Artifact Serialization:** Joblib

---

## 📊 Evaluation & Financial Calibration
Standard accuracy is misleading on skewed datasets (~26% churn rate). This model is tuned for:
- **Recall (Class 1):** Captures high-risk accounts before departure.
- **PR-AUC (Precision-Recall AUC):** Focuses performance exclusively on minority positive churn cases.
- **ROI Threshold Optimization:** Replaces the default `0.5` decision cutoff with an economically optimized threshold based on Customer Lifetime Value (CLV) and discount offer expenses.

---

## 🚀 Getting Started

### 1. Clone & Set Up Virtual Environment
```bash
git clone [https://github.com/](https://github.com/)<your-username>/customer-churn-ml.git
cd customer-churn-ml
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt