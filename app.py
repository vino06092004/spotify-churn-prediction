import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import sys

# Append src directory to path
sys.path.append(os.path.abspath("src"))
from feature_engineering import SpotifyFeatureEngineer

st.set_page_config(page_title="Spotify Churn Analytics Studio", layout="wide", page_icon="🎧")

# --- Subtle, Muted Multi-Theme Palettes (No Bright Neons) ---
THEMES = {
    "Graphite Studio (Default)": {
        "bg_gradient": "linear-gradient(135deg, #18191c 0%, #1e1f23 50%, #141517 100%)",
        "card_bg": "rgba(28, 29, 33, 0.75)",
        "accent": "#738a9c",
        "accent_glow": "rgba(115, 138, 156, 0.12)",
        "text": "#d8dee9",
        "subtext": "#8892b0",
        "border": "#2c2e34"
    },
    "Muted Forest & Sage": {
        "bg_gradient": "linear-gradient(135deg, #131a17 0%, #17211d 50%, #101614 100%)",
        "card_bg": "rgba(25, 36, 31, 0.75)",
        "accent": "#6b8e76",
        "accent_glow": "rgba(107, 142, 118, 0.12)",
        "text": "#d5ded7",
        "subtext": "#829688",
        "border": "#24332b"
    },
    "Subdued Midnight Navy": {
        "bg_gradient": "linear-gradient(135deg, #10141d 0%, #151a24 50%, #0d1017 100%)",
        "card_bg": "rgba(22, 28, 38, 0.75)",
        "accent": "#5e768d",
        "accent_glow": "rgba(94, 118, 141, 0.12)",
        "text": "#d6dce5",
        "subtext": "#7f8c9b",
        "border": "#252f3d"
    },
    "Warm Monochromatic Clay": {
        "bg_gradient": "linear-gradient(135deg, #191817 0%, #201e1d 50%, #141312 100%)",
        "card_bg": "rgba(33, 30, 28, 0.75)",
        "accent": "#a89078",
        "accent_glow": "rgba(168, 144, 120, 0.12)",
        "text": "#e0dbd5",
        "subtext": "#918982",
        "border": "#35312d"
    }
}

# Sidebar Theme Selector
with st.sidebar:
    st.markdown("### 🎨 Visual Theme")
    selected_theme_name = st.selectbox("Choose UI Palette", list(THEMES.keys()))
    theme = THEMES[selected_theme_name]
    st.markdown("---")
    st.markdown("**Platform Status:** Ready 🟢")
    st.caption("XGBoost & SHAP Explainability Engine")

# Inject Muted CSS Styling
custom_css = f"""
<style>
    .stApp {{
        background: {theme['bg_gradient']};
        color: {theme['text']};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    
    div[data-testid="column"] {{
        background: {theme['card_bg']};
        padding: 22px;
        border-radius: 12px;
        border: 1px solid {theme['border']};
        box-shadow: 0 4px 16px {theme['accent_glow']};
    }}

    h1, h2, h3, h4 {{
        color: {theme['text']} !important;
        font-weight: 600 !important;
        letter-spacing: -0.3px;
    }}

    button[kind="primary"] {{
        background-color: {theme['accent']} !important;
        color: #111111!important;
        font-weight: 700 !important;
        border: 1px solid {theme['border']} !important;
        border-radius: 8px !important;
        padding: 8px 24px !important;
        box-shadow: none !important;
        transition: background-color 0.2s ease;
    }}
    button[kind="primary"]:hover {{
        filter: brightness(1.08);
        border-color: {theme['accent']} !important;
    }}

    div[data-testid="stMetricValue"] {{
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        color: {theme['accent']} !important;
    }}
    
    label, p, span {{
        color: {theme['subtext']};
    }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Load Artifacts
@st.cache_resource
def load_artifacts():
    pipe = joblib.load(os.path.join("models", "churn_pipeline.joblib"))
    explainer = joblib.load(os.path.join("models", "explainer.joblib"))
    return pipe, explainer

try:
    pipeline, explainer = load_artifacts()
except Exception as e:
    st.error(f"Models not found. Run `python src/train.py` first. Error: {e}")
    st.stop()

st.title("🎧 Spotify Churn Intelligence Studio")
st.markdown(f"<span style='color:{theme['subtext']}; font-size:1.05rem;'>Subscriber attrition modeling, behavioral feature decomposition, and batch scoring.</span>", unsafe_allow_html=True)
st.write("")

tab1, tab2 = st.tabs(["⚡ Single Subscriber Evaluation", "📂 Batch File Analytics"])

with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"<h4 style='color:{theme['text']}'>👤 Demographics</h4>", unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        age = st.slider("Subscriber Age", 13, 80, 24)
        country = st.selectbox("Registered Country", ["USA", "India", "UK", "Germany", "Canada", "Australia", "Brazil"])

    with col2:
        st.markdown(f"<h4 style='color:{theme['text']}'>🎵 Streaming Patterns</h4>", unsafe_allow_html=True)
        listening_time = st.number_input("Daily Listening Time (Hours)", min_value=0.1, max_value=24.0, value=2.5, step=0.1)
        songs_played_per_day = st.number_input("Songs Streamed / Day", min_value=1, max_value=500, value=35)
        skip_rate = st.slider("Skip Rate", min_value=0.0, max_value=1.0, value=0.18, step=0.01)

    with col3:
        st.markdown(f"<h4 style='color:{theme['text']}'>💳 Subscription Tier</h4>", unsafe_allow_html=True)
        subscription_type = st.selectbox("Plan", ["Free", "Premium", "Family", "Student"])
        device_type = st.selectbox("Primary Device", ["Smartphone", "Desktop", "Web Player", "Smart TV", "Tablet"])
        ads_listened_per_week = st.number_input("Weekly Ad Exposure", min_value=0, max_value=200, value=0 if subscription_type != "Free" else 15)
        offline_listening = st.selectbox("Offline Songs Downloaded?", ["Yes", "No"])

    st.write("")
    if st.button("Calculate Churn Risk", type="primary"):
        input_data = pd.DataFrame([{
            'gender': gender,
            'age': age,
            'country': country,
            'subscription_type': subscription_type,
            'listening_time': listening_time,
            'songs_played_per_day': songs_played_per_day,
            'skip_rate': skip_rate,
            'device_type': device_type,
            'ads_listened_per_week': ads_listened_per_week,
            'offline_listening': offline_listening
        }])

        prob = pipeline.predict_proba(input_data)[0][1]

        st.markdown("---")
        m_col1, m_col2 = st.columns([1, 2])

        with m_col1:
            st.metric("Churn Risk Probability", f"{prob:.1%}")
            
            if prob >= 0.65:
                st.error("🚨 **High Risk Tier**\nTrigger retention intervention: Push discount offer or customized discovery mix.")
            elif prob >= 0.35:
                st.warning("⚠️ **Moderate Risk Tier**\nEngagement drop detected: Send artist updates & personalized playlists.")
            else:
                st.success("✅ **Healthy Retention Tier**\nSubscriber exhibits regular listening frequency and low churn likelihood.")

        with m_col2:
            st.markdown(f"<h4 style='color:{theme['text']}'>📊 SHAP Feature Impact</h4>", unsafe_allow_html=True)
            try:
                prep_pipe = pipeline[:-1]
                transformed_input = prep_pipe.transform(input_data)
                shap_values = explainer(transformed_input)
                shap_values.feature_names = pipeline.named_steps['prep'].get_feature_names_out()

                fig, ax = plt.subplots(figsize=(8, 3.5))
                fig.patch.set_facecolor('#1a1c20')
                ax.set_facecolor('#1a1c20')
                
                shap.plots.waterfall(shap_values[0], max_display=6, show=False)
                st.pyplot(fig)
            except Exception as ex:
                st.info(f"Could not render SHAP plot: {ex}")

with tab2:
    st.markdown(f"<h4 style='color:{theme['text']}'>📂 Bulk Subscriber Scoring</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV file matching Spotify schema", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        cleaned_batch = batch_df.copy()
        user_ids = cleaned_batch['user_id'] if 'user_id' in cleaned_batch.columns else pd.Series(range(len(cleaned_batch)), name="user_id")
        cleaned_batch = cleaned_batch.drop(columns=['user_id', 'is_churned'], errors='ignore')

        probs = pipeline.predict_proba(cleaned_batch)[:, 1]
        results_df = pd.DataFrame({
            'User_ID': user_ids,
            'Churn_Probability': np.round(probs, 3),
            'Risk_Band': pd.cut(probs, bins=[-0.01, 0.35, 0.65, 1.0], labels=['Low', 'Medium', 'High'])
        })

        st.dataframe(results_df.head(25), use_container_width=True)

        csv_download = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Scored Results CSV",
            data=csv_download,
            file_name="spotify_churn_scored.csv",
            mime="text/csv"
        )