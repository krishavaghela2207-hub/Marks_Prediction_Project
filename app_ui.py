import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('model.pkl','rb'))

st.title("Marks Prediction App")

# Inputs
study_hours = st.number_input("Enter Study Hours:", min_value=0, max_value=12, value=5)
attendance = st.number_input("Enter Attendance (%):", min_value=0, max_value=100, value=80)
previous_grade = st.number_input("Enter Previous Grade:", min_value=0, max_value=100, value=75)
sleep_hours = st.number_input("Enter Sleep Hours:", min_value=0, max_value=12, value=7)

if st.button("Predict"):
    # Make prediction
    result = model.predict([[previous_grade, study_hours, attendance, sleep_hours]])
    
    # Convert to scalar safely
    predicted_marks = float(np.ravel(result)[0])
    
    st.success(f"Predicted Marks: {int(predicted_marks)}")