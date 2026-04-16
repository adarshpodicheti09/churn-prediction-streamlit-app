import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Customer Churn Prediction App")

st.write("Enter customer details:")

tenure = st.slider("Tenure (months)", 0, 72)
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Encoding
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
contract = contract_map[contract]

# Feature array
features = np.array([[tenure, monthly_charges, total_charges, contract]])

# SCALE input
features = scaler.transform(features)

if st.button("Predict"):
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("Customer is likely to CHURN")
    else:
        st.success("Customer will STAY")