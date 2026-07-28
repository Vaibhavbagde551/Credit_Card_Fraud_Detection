import streamlit as st
import pandas as pd
import joblib


model = joblib.load("fraud_detection_model.pkl")

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)


st.sidebar.title("Credit Card Fraud Detection")

st.sidebar.markdown("""
### Project Information

**Model:** Random Forest (Tuned)

**Problem Type:** Binary Classification

**Goal:** Detect fraudulent credit card transactions.

**Dataset:** European Cardholders Credit Card Fraud Dataset
""")

st.sidebar.markdown("---")

st.sidebar.subheader("Model Performance")

st.sidebar.write("Precision : **97%**")
st.sidebar.write("Recall : **74%**")
st.sidebar.write("F1-Score : **84%**")

st.sidebar.markdown("---")

st.sidebar.info(
    """
This dataset contains **PCA-transformed features (V1–V28)** to protect customer privacy.

For this reason, predictions are made using uploaded CSV files rather than manual transaction entry.
"""
)

st.markdown("---")

st.header("How to Use")

st.info("""
1. Upload a CSV file containing transaction data.
2. Click **Predict Fraud**.
3. Review the prediction results.
4. Download the results as a CSV file.
""")

st.title("Credit Card Fraud Detection")

st.write("""
This application predicts whether a credit card transaction is **Fraudulent** or **Genuine**
using a tuned Random Forest machine learning model.
Upload a CSV file containing transaction records to get predictions.
""")


uploaded_file = st.file_uploader(
    "Upload a CSV File",
    type=["csv"]
)


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")

    st.dataframe(df)

    expected_columns = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15",
    "V16", "V17", "V18", "V19", "V20", "V21", "V22",
    "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]
    missing_columns = set(expected_columns) - set(df.columns)
    extra_columns = set(df.columns) - set(expected_columns)

    if len(missing_columns) == 0 and len(extra_columns) == 0:

        st.success("Valid dataset uploaded!")

        if st.button("Prediction Fraud"):
            predictions = model.predict(df)

            df["Predictions"] = predictions

            df["Predictions"] = df["Predictions"].map({
                0:"Genuine",
                1:"Fraud"
            }
                
            )
            st.subheader("Prediction Results")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Result",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )


    else:

        st.error("Invalid CSV format!")

        if len(missing_columns) > 0:
            st.write("### Missing Columns")
            st.write(sorted(missing_columns))

        if len(extra_columns) > 0:
            st.write("### Extra Columns")
            st.write(sorted(extra_columns))
    
st.markdown("---")

st.caption(
    "Developed by **Vaibhav Bagde**"
)
