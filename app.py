import streamlit as st
import joblib
import pandas as pd
from PIL import Image

# Load the model
model = joblib.load('G:/Collins diabetes app/diabetes_xgboost_model.pkl')

# Set the title and thumbnail image
st.set_page_config(page_title="Collins Diabetes Prediction App", page_icon=":hospital:")
thumbnail = Image.open("thumbnail.png")
thumbnail = thumbnail.resize((400, 200))  # Adjust the size as needed
st.image(thumbnail, use_column_width=False)

# Title and description
st.title("Collins Diabetes Prediction App")
st.write("""
This app predicts the likelihood of diabetes based on various health metrics. It can be used in real time to predict the likelihood of diabetes of a patient.
Please input the following details to get a prediction.
""")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=85)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=28.5)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })
    
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0][1]
    
    st.subheader("Prediction Result")
    st.write(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")
    st.write(f"Probability: {prediction_proba:.2%} chance of diabetes")

# Footer
st.markdown("""
<style>
footer {
    visibility: hidden;
}
footer:after {
    content:'Developed by Collins Asiedu Gyamfi (Health data scientist/analyst)';
    visibility: visible;
    display: block;
    position: relative;
    padding: 10px;
    top: 2px;
}
</style>
""", unsafe_allow_html=True)