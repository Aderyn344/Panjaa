import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import joblib

# Load your pre-trained model pipelines (replace with your actual loading logic)
binary_pipeline = joblib.load('binary_pipeline.joblib')

# Set app title and description
st.title("Cancer Stage Prediction App")
st.write("This app predicts the stage of Cancer based on patient information.")


def predict_stage(df, binary_pipeline):
    """
    Predicts stage using the trained pipelines.

    Args:
        df (pd.DataFrame): Input data containing features.
        binary_pipeline: Trained pipeline for binary classification.

    Returns:
        str: Predicted stage or "Unknown" if binary prediction is unknown.
    """
    # Predict binary class (known or unknown)
    y_pred_binary = binary_pipeline.predict(df)

    return y_pred_binary[0]

# Get user input for features
st.subheader("Enter Patient Information")

# Create input fields for each feature
features = {
    "TYPES OF VISIT": st.selectbox("Types of Visit", ["Revisit", "Visit"]),
    "SEX": st.selectbox("Sex", ["Male", "Female"]),
    "AGE": st.slider("Age", 0, 100),
    "COUNTY OF RESIDENCE": st.selectbox("County of Residence", ['KAKAMEGA', 'BUSIA', 'VIHIGA']),
    "DIAGNOSIS/RESULTS": st.text_input("Diagnosis/Results"),
    "HIV STATUS": st.selectbox("HIV Status", ["Positive", "Negative", "Unknown"]),
}

user_data = pd.DataFrame.from_dict([features])

# Predict stage
if st.button("Predict Stage"):
    predicted_stage = predict_stage(user_data, binary_pipeline)
    st.write(f"Predicted Stage: {predicted_stage}")
