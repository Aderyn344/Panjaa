import streamlit as st
import pandas as pd
import joblib
# ... other code ...

# Before line 3
print("Before line 3:")
print("Variable values:")
print(some_variables)  # Replace with actual variables you want to check

# Line 3 (where the error is occurring)
result = some_function()  # Replace with your actual code

# After line 3
print("After line 3:")
print("Result:", result)
print("Other relevant variables:", other_variables)  # Replace with other variables you want to inspect

# ... rest of your code ...

# Load your pre-trained model pipelines (replace with your actual loading logic)
binary_pipeline = ...  # Load the binary classification pipeline
stage_pipeline = ...  # Load the multi-class classification pipeline

binary_pipeline = joblib.load('binary_pipeline.joblib')
stage_pipeline = joblib.load('stage_pipeline.joblib')

# Set app title and description
st.title("Cancer Stage Prediction App")
st.write("This app predicts the stage of Cancer based on patient information.")


def predict_stage(df, binary_pipeline, stage_pipeline):
  """
  Predicts stage using the trained pipelines.

  Args:
      df (pd.DataFrame): Input data containing features.
      binary_pipeline: Trained pipeline for binary classification.
      stage_pipeline: Trained pipeline for multi-class classification.

  Returns:
      str: Predicted stage or "Unknown" if binary prediction is unknown.
  """
  # Predict binary class (known or unknown)
  y_pred_binary = binary_pipeline.predict(df)

  if y_pred_binary == "unknown":
    return "Unknown"
  else:
    # Filter data based on known predictions
    known_df = df[y_pred_binary == "known"]
    # Predict stage for known cases
    y_pred_stage = stage_pipeline.predict(known_df)
    return stage_pipeline.classes_[y_pred_stage[0]]


# Get user input for features
st.subheader("Enter Patient Information")

# Create input fields for each feature
features = {
    "TYPES OF VISIT": st.selectbox("Types of Visit", ["Revisit", "Visit"]),
    "SEX": st.selectbox("Sex", ["Male", "Female"]),
    "AGE": st.number_input("Age"),
    "COUNTY OF RESIDENCE": st.selectbox("County of Residence", ['KAKAMEGA', 'BUSIA', 'VIHIGA']),
    "DIAGNOSIS/RESULTS": st.text_input("Diagnosis/Results"),
    "HIV STATUS": st.selectbox("HIV Status", ["Positive", "Negative", "Unknown"]),
}

user_data = pd.DataFrame.from_dict([features])

# Predict stage
if st.button("Predict Stage"):
  predicted_stage = predict_stage(user_data, binary_pipeline, stage_pipeline)
  st.write(f"Predicted Stage: {predicted_stage}")
