"""
Streamlit demo: enter a customer's profile, get a churn-risk prediction.

Run with:  streamlit run app.py
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📡")
st.title("📡 Telecom Customer Churn Predictor")
st.caption("Logistic Regression pipeline trained on Telco Customer Churn data")

model = joblib.load("artifacts/churn_model.joblib")

col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
with col2:
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

row = pd.DataFrame([{
    "gender": "Male", "SeniorCitizen": senior, "Partner": "No", "Dependents": dependents,
    "tenure": tenure, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": internet, "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": tech_support, "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": contract, "PaperlessBilling": paperless,
    "PaymentMethod": payment, "MonthlyCharges": monthly_charges,
    "TotalCharges": monthly_charges * max(tenure, 1),
}])

if st.button("Predict churn risk"):
    prob = model.predict_proba(row)[0, 1]
    st.metric("Churn probability", f"{prob:.1%}")
    if prob > 0.5:
        st.error("High risk of churn")
    else:
        st.success("Low risk of churn")
