import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

# === Page config ===
st.set_page_config(page_title="Telco Churn Prediction", layout="centered")
st.title("📱 Telco Customer Churn Prediction")
st.markdown("Enter customer information below to assess churn risk.")

# === Load serialized objects ===
model = pickle.load(open("models/xgboost_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
encoder = pickle.load(open("models/encoder.pkl", "rb"))
columns = pickle.load(open("models/columns.pkl", "rb"))
metrics = pickle.load(open("models/metrics.pkl", "rb"))

# === Load dataset from CSV ===
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return df

data = load_data()

# === Data Preview (optional) ===
with st.expander("🔍 View Sample Dataset"):
    st.dataframe(data.head())

# === Input sidebar ===
with st.sidebar:
    st.header("Customer Details")

    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0)

# === Predict button ===
if st.button("Predict Churn"):
    # === Define features according to training ===
    num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_features = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod"
    ]

    # Construct input DataFrame
    input_dict = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,  # Note this is numeric and in training
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
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    input_df = pd.DataFrame([input_dict])

    # Scale numerical features
    num_cols_with_senior = ["SeniorCitizen"] + num_features
    num_scaled = scaler.transform(input_df[num_cols_with_senior])

    # Encode categorical features
    cat_encoded = encoder.transform(input_df[cat_features])
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_features))

    # Combine scaled numerical and encoded categorical features
    final_input = np.concatenate([num_scaled, cat_encoded_df.values], axis=1)

    # Convert to DataFrame with full column names
    final_input_df = pd.DataFrame(final_input, columns=columns)

    # Predict
    prediction = model.predict(final_input_df)[0]
    probability = model.predict_proba(final_input_df)[0][1]

    # Output prediction
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ The customer is **likely to churn** (Probability: {probability:.2%})")
    else:
        st.success(f"✅ The customer is **not likely to churn** (Probability: {probability:.2%})")

    # Feature importance
    with st.expander("📊 Feature Importance (Top 10)"):
        booster = model.get_booster()
        importance = booster.get_score(importance_type="weight")
        importance_df = pd.DataFrame({
            "Feature": list(importance.keys()),
            "Importance": list(importance.values())
        }).sort_values(by="Importance", ascending=False).head(10)
        st.bar_chart(importance_df.set_index("Feature"))

     # Model Metrics Overview
    with st.expander("📈 Model Performance Metrics (Test Set)"):
        st.metric(label="ROC AUC Score", value=f"{metrics['roc_auc']:.3f}")

        report_df = pd.DataFrame(metrics["report"]).transpose()
        report_df.index = report_df.index.map({
            "0": "No Churn",
            "1": "Churn",
            "accuracy": "Accuracy",
            "macro avg": "Macro Avg",
            "weighted avg": "Weighted Avg"
        })
        st.dataframe(report_df.style.format(precision=3).background_gradient(cmap="Blues"))

    # Confusion Matrix
    with st.expander("🧮 Confusion Matrix"):
        cm = np.array(metrics["confusion_matrix"])
        labels = ["No Churn", "Churn"]
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

# === Footer ===
st.caption("Developed using Streamlit and XGBoost | Dataset: IBM Telco Churn on Kaggle")
