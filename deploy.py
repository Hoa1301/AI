import streamlit as st
import pandas as pd
import pickle

# Load the pipeline from the pickle file
with open('decision_tree_pipeline.pkl', 'rb') as file:
    loaded_pipeline = pickle.load(file)

# Set up the Streamlit app title
st.title("Loan Approval Prediction App")

# Create input fields for user to enter data
age = st.number_input("Age", min_value=18, max_value=100, value=30)
annual_income = st.number_input("Annual Income", min_value=0, value=50000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-Employed"])
education_level = st.selectbox("Education Level", ["Bachelor", "Master", "PhD"])
loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
loan_duration = st.number_input("Loan Duration (months)", min_value=1, value=12)

# Create a button to trigger prediction
if st.button("Predict"):
    # Create a DataFrame from user input
    new_data = pd.DataFrame({
        'Age': [age],
        'AnnualIncome': [annual_income],
        'CreditScore': [credit_score],
        'EmploymentStatus': [employment_status],
        'EducationLevel': [education_level],
        'LoanAmount': [loan_amount],
        'LoanDuration': [loan_duration]
    })

    # Make prediction using the loaded pipeline
    prediction = loaded_pipeline.predict(new_data)

    # Display the prediction
    if prediction[0] == 1:
        st.success("Loan Approved!")
    else:
        st.error("Loan Rejected.")