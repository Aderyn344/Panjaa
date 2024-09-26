import streamlit as st
import pandas as pd
import joblib

# Load trained models (including preprocessing pipelines)
binary_classifier = joblib.load('binary_pipeline.joblib')  # Load binary classification model
stage_classifier = joblib.load('stage_pipeline.joblib')

# Set app title and description
st.title("Cancer Stage Prediction App")
st.write("This app predicts the stage of Cancer based on patient information.")

def predict_stage(df, binary_classifier, stage_classifier):
    """
    Predicts stage using the trained models.

    Args:
        df (pd.DataFrame): Input data containing features.
        binary_classifier: Trained model for binary classification.
        stage_classifier: Trained model for multi-class classification.

    Returns:
        str: Predicted stage or "Unknown".
    """
    # Pass the dataframe directly to the classifier since the pipeline handles preprocessing
    y_pred_binary = binary_classifier.predict(df)

    if y_pred_binary == 'known':
        # Predict specific cancer stage if the binary classification result is "known"
        y_pred_stage = stage_classifier.predict(df)
        predicted_stage = y_pred_stage[0]
        return f"Known Stage {predicted_stage}"
    else:
        return "Unknown"

# Get user input for features
st.subheader("Enter Patient Information")

# Create input fields for each feature (assuming the same features as training)
features = {
    "TYPES OF VISIT": st.selectbox("Types of Visit", ["Revisit", "Visit"]),
    "SEX": st.selectbox("Sex", ["Male", "Female"]),
    "AGE": st.text_input("Age"),
    "COUNTY OF RESIDENCE": st.selectbox("County of Residence", ['KAKAMEGA', 'BUSIA', 'VIHIGA']),
    "DIAGNOSIS/RESULTS": st.text_input("Diagnosis/Results"),
    "HIV STATUS": st.selectbox("HIV Status", ["Positive", "Negative", "Unknown"]),
}

# Create a DataFrame from the dictionary
user_df = pd.DataFrame(features, index=[0])

# Ensure 'AGE' is numeric
user_df['AGE'] = pd.to_numeric(user_df['AGE'], errors='coerce')

# Predict stage
if st.button("Predict Stage"):
    predicted_stage = predict_stage(user_df.copy(), binary_classifier, stage_classifier)
    st.write(f"Predicted Stage: {predicted_stage}")
