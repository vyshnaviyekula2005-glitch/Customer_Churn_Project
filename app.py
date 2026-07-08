import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("customer_churn_model.pkl")

st.title("Customer Churn Prediction System")
st.write("Enter customer details to predict churn.")

# Input fields
tenure = st.number_input("Tenure (Months)", min_value=0, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=600.0)

if st.button("Predict Churn"):
    avg_monthly_spend = total_charges / (tenure + 1)

    # Create input dataframe
    input_data = pd.DataFrame([[tenure, monthly_charges, total_charges, avg_monthly_spend]],
                              columns=["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend"])

    # Match training columns
    train_columns = model.feature_names_in_

    for col in train_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[train_columns]

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")