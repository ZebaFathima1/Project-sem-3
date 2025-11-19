# app.py  →  Streamlit Heart Disease Risk Predictor (FIXED)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="red_heart",
    layout="centered"
)

st.title("Heart Disease Risk Prediction")
st.markdown("""
Predict your **continuous heart disease risk score (0–1)** using a Random Forest model  
trained on more than 135,000 real patient records.
""")

# ---------------------- Paths ----------------------
MODEL_PATH = "heart_risk_model.pkl"
SCALER_PATH = "heart_risk_scaler.pkl"
FEATURES_PATH = "feature_columns.pkl"

DATA_FILE_1 = "heart_disease_risk_dataset_earlymed111.csv"
DATA_FILE_2 = "heart_disease_risk_dataset_earlymed.csv"

# ---------------------- Train or Load Model ----------------------
@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_PATH):
        st.success("Loading pre-trained model...")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names = joblib.load(FEATURES_PATH)
    else:
        with st.spinner("First run → Training model (approximately 20–30 seconds)..."):
            df1 = pd.read_csv(DATA_FILE_1)
            df2 = pd.read_csv(DATA_FILE_2)
            df = pd.concat([df1, df2], ignore_index=True)

            # Fill missing values (exactly like your notebook)
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
            cat_cols = df.select_dtypes(include=['object']).columns
            for col in cat_cols:
                df[col] = df[col].fillna(df[col].mode()[0])

            X = df.drop("Heart_Risk", axis=1)
            y = df["Heart_Risk"]

            # One-hot encoding
            X = pd.get_dummies(X, drop_first=True)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            model.fit(X_scaled, y)

            # Save for future runs
            joblib.dump(model, MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)
            joblib.dump(X.columns.tolist(), FEATURES_PATH)

            feature_names = X.columns.tolist()

        st.success("Model trained and saved for future use!")
    
    return model, scaler, feature_names

model, scaler, feature_names = load_or_train_model()

# ---------------------- Input Form ----------------------
with st.form("heart_risk_form"):
    st.subheader("Please enter your information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 100, 50)
        gender = st.selectbox("Gender", ["Male", "Female"])
        cp = st.selectbox("Chest Pain (0–3)", [0, 1, 2, 3], help="0 = No pain, 1–3 = increasing severity")
        sob = st.selectbox("Shortness of Breath", [0, 1])
        fatigue = st.selectbox("Fatigue", [0, 1])
        palpitations = st.selectbox("Palpitations", [0, 1])
        dizziness = st.selectbox("Dizziness", [0, 1])
        swelling = st.selectbox("Swelling (Edema)", [0, 1])   # ← FIXED LINE

    with col2:
        radiating_pain = st.selectbox("Radiating Pain (arms/jaw/back)", [0, 1])
        cold_sweats = st.selectbox("Cold Sweats / Nausea", [0, 1])
        hypertension = st.selectbox("High Blood Pressure", [0, 1])
        cholesterol_high = st.selectbox("High Cholesterol", [0, 1])
        diabetes = st.selectbox("Diabetes", [0, 1])
        smoker = st.selectbox("Current Smoker", [0, 1])
        obesity = st.selectbox("Obesity", [0, 1])
        family_history = st.selectbox("Family History of Heart Disease", [0, 1])
        sedentary = st.selectbox("Sedentary Lifestyle", [0, 1])
        stress = st.selectbox("Chronic Stress", [0, 1])

    submitted = st.form_submit_button("Predict My Risk", type="primary")

# ---------------------- Prediction ----------------------
if submitted:
    # Build input dictionary
    input_data = {
        "age": age,
        "cp": cp,
        "sob": sob,
        "fatigue": fatigue,
        "palpitations": palpitations,
        "dizziness": dizziness,
        "swelling": swelling,
        "radiating_pain": radiating_pain,
        "cold_sweats": cold_sweats,
        "hypertension": hypertension,
        "cholesterol_high": cholesterol_high,
        "diabetes": diabetes,
        "smoker": smoker,
        "obesity": obesity,
        "family_history": family_history,
        "Sedentary_Lifestyle": sedentary,
        "Chronic_Stress": stress,
        "Gender": "M" if gender == "Male" else "F",
    }

    # Convert → one-hot → align with training columns
    df_input = pd.DataFrame([input_data])
    df_input = pd.get_dummies(df_input, drop_first=True)
    df_input = df_input.reindex(columns=feature_names, fill_value=0)

    # Predict
    X_scaled = scaler.transform(df_input)
    risk_score = float(model.predict(X_scaled)[0])

    # Display beautiful result
    st.markdown("---")
    st.subheader("Your Heart Disease Risk")

    risk_pct = risk_score * 100
    st.metric(label="Risk Score", value=f"{risk_score:.4f}", delta=None)

    if risk_score < 0.3:
        st.success("Low Risk – Keep up the healthy lifestyle!")
    elif risk_score < 0.7:
        st.warning("Moderate Risk – Consider lifestyle changes and consult a doctor.")
    else:
        st.error("High Risk – Please consult a cardiologist as soon as possible.")

    st.balloons()

# ---------------------- Footer ----------------------
st.markdown("---")
st.caption("Random Forest Regressor (200 trees) • 135k+ records • R² approximately 0.964 • Model auto-saved for instant future use")