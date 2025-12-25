import streamlit as st
import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)

st.title("Heart Disease Prediction App")
st.markdown(
    "Predict heart disease risk using clinical features and "
    "explain predictions with **SHAP**."
)

# -----------------------------
# Load and preprocess data
# -----------------------------
import os

@st.cache_data(show_spinner="Loading and preprocessing data...")
def load_and_prepare_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw_merged_heart_dataset.csv")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df.replace(['?', '', 'unknown', 999, -1], np.nan, inplace=True)

    numeric_columns = ['trestbps', 'chol', 'thalachh', 'fbs', 'exang', 'ca']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    df = pd.get_dummies(df, columns=['restecg', 'slope', 'thal'], drop_first=True)

    X = df.drop('target', axis=1)
    y = df['target']

    scaler = StandardScaler()
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

    return X, y, scaler

X, y, scaler = load_and_prepare_data()

# -----------------------------
# Train model
# -----------------------------
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)

    return model, X_test

model, X_test = train_model(X, y)

# -----------------------------
# Sidebar input
# -----------------------------
st.sidebar.header("Patient Input")

def user_input():
    age = st.sidebar.slider("Age", 20, 90, 55)
    trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 130)
    chol = st.sidebar.slider("Cholesterol", 100, 400, 240)
    thalachh = st.sidebar.slider("Max Heart Rate", 70, 210, 150)
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
    ca = st.sidebar.slider("Major Vessels (0–3)", 0, 3, 0)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120", [0, 1])

    data = {
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalachh": thalachh,
        "oldpeak": oldpeak,
        "ca": ca,
        "exang": exang,
        "fbs": fbs
    }

    df = pd.DataFrame([data])
    df = df.reindex(columns=X.columns, fill_value=0)

    return df

input_df = user_input()

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

st.subheader("Prediction Result")

if prediction == 1:
    st.error(f"⚠️ High Risk of Heart Disease\n\nProbability: **{probability:.2f}**")
else:
    st.success(f"✅ Low Risk of Heart Disease\n\nProbability: **{probability:.2f}**")

# -----------------------------
# SHAP Explanation
# -----------------------------
st.subheader("Model Explanation (SHAP)")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Handle different SHAP return formats safely
if isinstance(shap_values, list):
    shap_values_to_plot = shap_values[1]  # binary classifier → positive class
else:
    shap_values_to_plot = shap_values

fig, ax = plt.subplots()
shap.summary_plot(shap_values_to_plot, X_test, show=False)
st.pyplot(fig)


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "⚠️ Educational use only. Not for clinical decision-making."
)
