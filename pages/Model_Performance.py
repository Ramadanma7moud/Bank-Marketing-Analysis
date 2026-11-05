import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Model Performance",
    page_icon="🎯",
    layout="wide"
)

# Page Title
st.title("🎯 Machine Learning Model Performance")

st.write(
    "Evaluate the performance of the Random Forest model "
    "on the unseen test dataset."
)

st.divider()

# Model Evaluation Metrics
st.subheader("📊 Evaluation Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "72.45%")

with col2:
    st.metric("Precision (Yes)", "75%")

with col3:
    st.metric("Recall (Yes)", "63%")

with col4:
    st.metric("F1-Score (Yes)", "68%")

st.divider()

# Confusion Matrix
st.subheader("🔍 Confusion Matrix")

confusion_data = pd.DataFrame(
    [[949, 226], [389, 669]],
    index=["Actual No", "Actual Yes"],
    columns=["Predicted No", "Predicted Yes"]
)

st.dataframe(
    confusion_data,
    use_container_width=True
)
