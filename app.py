import streamlit as st
import numpy as np
from utils.preprocessing import preprocess_input
from utils.prediction import predict
from PIL import Image

st.set_page_config(page_title="Bank Fraud Detector", layout="wide")
st.title("🏦 Advanced Bank Fraud Detection System")

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Enter Transaction Data:")
    amount = st.number_input("💰 Transaction Amount", min_value=0.0, step=1.0)
    time = st.number_input("🕒 Time (seconds since first)", min_value=0.0, step=1.0)

    v_features = []
    for i in range(1, 29):
        v = st.number_input(f"V{i}", value=0.0, format="%.4f")
        v_features.append(v)

    if st.button("🔎 Detect Fraud"):
        input_scaled = preprocess_input(time, amount, v_features)
        result = predict(input_scaled)
        if result == -1:
            st.error("🚨 ALERT: Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")

with col2:
    st.subheader("📊 Fraud Insights")
    st.markdown("""
        - Based on unsupervised anomaly detection (Isolation Forest).
        - Trained on over 280,000 real transaction records.
        - Imbalanced data handled using contamination strategy.
        - Scaled and standardized for performance.
    """)
    img = Image.open("assets/fraud_banner.png")
    st.image(img, use_column_width=True)
