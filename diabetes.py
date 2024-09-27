import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Title and description for the app
st.title('Diabetes Prediction App')
st.write("""
This app predicts whether a person has diabetes based on health indicators.
""")


# Load the model
model = joblib.load('diabetes_model.joblib')

# Sidebar for user input
st.sidebar.header('Input Features')
def user_input_features():
    HighBP = st.sidebar.selectbox('High Blood Pressure (1: Yes, 0: No)', [1, 0])
    HighChol = st.sidebar.selectbox('High Cholesterol (1: Yes, 0: No)', [1, 0])
    BMI = st.sidebar.slider('BMI', 10.0, 50.0, 25.0)
    Smoker = st.sidebar.selectbox('Smoker (1: Yes, 0: No)', [1, 0])
    HeartDiseaseorAttack = st.sidebar.selectbox('Heart Disease or Attack (1: Yes, 0: No)', [1, 0])
    PhysActivity = st.sidebar.selectbox('Physical Activity (1: Yes, 0: No)', [1, 0])
    Fruits = st.sidebar.selectbox('Eat Fruits (1: Yes, 0: No)', [1, 0])
    Veggies = st.sidebar.selectbox('Eat Vegetables (1: Yes, 0: No)', [1, 0])
    HvyAlcoholConsump = st.sidebar.selectbox('Heavy Alcohol Consumption (1: Yes, 0: No)', [1, 0])
    Sex = st.sidebar.selectbox('Sex (1: Male, 0: Female)', [1, 0])
    Age = st.sidebar.slider('Age', 18, 90, 30)
    Income = st.sidebar.selectbox('Income (1: Lowest, 5: Highest)', [1, 2, 3, 4, 5])
    
    data = {
        'HighBP': HighBP,
        'HighChol': HighChol,
        'BMI': BMI,
        'Smoker': Smoker,
        'HeartDiseaseorAttack': HeartDiseaseorAttack,
        'PhysActivity': PhysActivity,
        'Fruits': Fruits,
        'Veggies': Veggies,
        'HvyAlcoholConsump': HvyAlcoholConsump,
        'Sex': Sex,
        'Age': Age,
        'Income': Income
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input in app
st.subheader('User Input features')
st.write(input_df)

# Model Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Display the prediction result
st.subheader('Prediction')
diabetes_status = 'Yes' if prediction[0] == 1 else 'No'
st.write(f'Do you have diabetes? **{diabetes_status}**')

# Show prediction probabilities
st.subheader('Prediction Probability')
st.write(f"Probability of having diabetes: **{prediction_proba[0][1]:.2f}**")
st.write(f"Probability of not having diabetes: **{prediction_proba[0][0]:.2f}**")
