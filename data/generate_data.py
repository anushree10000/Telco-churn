"""
Generates a synthetic dataset that mirrors the schema and general statistical
patterns of the well-known IBM/Kaggle "Telco Customer Churn" dataset
(https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

Why this exists: if you can grab the real Kaggle CSV before your interview,
do that instead -- it's a recognizable dataset and shows you can work with
real-world messy data. Drop it in this folder as `telco_churn.csv` and
train.py will use it automatically. This script is just a working fallback
so the whole pipeline runs end-to-end with zero setup.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 5000

gender = rng.choice(["Male", "Female"], N)
senior_citizen = rng.choice([0, 1], N, p=[0.84, 0.16])
partner = rng.choice(["Yes", "No"], N)
dependents = rng.choice(["Yes", "No"], N, p=[0.3, 0.7])
tenure = rng.integers(0, 73, N)

phone_service = rng.choice(["Yes", "No"], N, p=[0.9, 0.1])
multiple_lines = np.where(
    phone_service == "No", "No phone service", rng.choice(["Yes", "No"], N)
)
internet_service = rng.choice(["DSL", "Fiber optic", "No"], N, p=[0.35, 0.44, 0.21])

def dep_on_internet(col_yes_p=0.5):
    return np.where(
        internet_service == "No",
        "No internet service",
        rng.choice(["Yes", "No"], N, p=[col_yes_p, 1 - col_yes_p]),
    )

online_security = dep_on_internet(0.35)
online_backup = dep_on_internet(0.4)
device_protection = dep_on_internet(0.4)
tech_support = dep_on_internet(0.35)
streaming_tv = dep_on_internet(0.45)
streaming_movies = dep_on_internet(0.45)

contract = rng.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.21, 0.24])
paperless_billing = rng.choice(["Yes", "No"], N, p=[0.59, 0.41])
payment_method = rng.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    N,
)

base_charge = rng.normal(65, 30, N).clip(18, 120)
monthly_charges = base_charge + (internet_service == "Fiber optic") * 20
total_charges = (monthly_charges * tenure + rng.normal(0, 50, N)).clip(0, None)

# Churn probability driven by realistic risk factors (short tenure,
# month-to-month contracts, fiber+no tech support, electronic check)
churn_logit = (
    -2.0
    + (contract == "Month-to-month") * 1.4
    + (tenure < 12) * 0.9
    + (internet_service == "Fiber optic") * 0.5
    + (tech_support == "No") * 0.35
    + (payment_method == "Electronic check") * 0.4
    - (contract == "Two year") * 1.1
    - (dependents == "Yes") * 0.2
)
churn_prob = 1 / (1 + np.exp(-churn_logit))
churn = (rng.random(N) < churn_prob).astype(int)
churn_label = np.where(churn == 1, "Yes", "No")

df = pd.DataFrame({
    "customerID": [f"C{i:05d}" for i in range(N)],
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges.round(2),
    "TotalCharges": total_charges.round(2),
    "Churn": churn_label,
})

df.to_csv("data/telco_churn.csv", index=False)
print(f"Wrote data/telco_churn.csv with {len(df)} rows, churn rate = {churn.mean():.2%}")
