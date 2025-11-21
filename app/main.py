import pathlib
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from rag_pipeline import load_policies, simple_keyword_retrieval, build_text_justification

# Paths
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "xgb_model_demo.pkl"
FEATURE_INFO_PATH = BASE_DIR / "models" / "feature_info.pkl"
POLICY_PATH = BASE_DIR / "policy_docs" / "lending_policies.md"

# Load model & feature info
model = joblib.load(MODEL_PATH)
feature_info = joblib.load(FEATURE_INFO_PATH)
feature_names = feature_info["feature_names"]

# Prepare SHAP explainer (tree explainer)
explainer = shap.TreeExplainer(model)

# Load policies
policies_text = load_policies(str(POLICY_PATH))

st.set_page_config(page_title="Explainable Loan Approval", layout="centered")
st.title("Explainable Loan Approval – Demo")
st.write(
    "This is a small prototype that combines an XGBoost model, SHAP explanations, "
    "and simple retrieval over a lending policy document."
)

st.subheader("Enter application details")

# Simple UI based on known features
col1, col2 = st.columns(2)

with col1:
    annual_revenue = st.number_input("Annual revenue (£)", min_value=0, max_value=2_000_000, value=200_000, step=10_000)
    years_trading = st.number_input("Years trading", min_value=0, max_value=30, value=3, step=1)
    debt_to_income = st.number_input("Debt-to-income ratio", min_value=0.0, max_value=2.0, value=0.4, step=0.01)

with col2:
    late_payments = st.number_input("Late payments (last 12 months)", min_value=0, max_value=20, value=0, step=1)
    sector_risk = st.number_input("Sector risk score (0–1)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    requested_amount = st.number_input("Requested amount (£)", min_value=0, max_value=500_000, value=50_000, step=5_000)

if st.button("Evaluate application"):
    # Build input row respecting feature order
    input_dict = {
        "annual_revenue": annual_revenue,
        "years_trading": years_trading,
        "debt_to_income": debt_to_income,
        "late_payments": late_payments,
        "sector_risk": sector_risk,
        "requested_amount": requested_amount,
    }

    x = pd.DataFrame([input_dict])[feature_names]

    # Model prediction
    proba = model.predict_proba(x)[0, 1]
    pred_label = int(proba >= 0.5)

    if pred_label == 1:
        st.success(f"Predicted outcome: APPROVE (p = {proba:.2f})")
    else:
        st.error(f"Predicted outcome: REJECT (p = {proba:.2f})")

    # SHAP explanation for this instance
    shap_values = explainer.shap_values(x)
    # shap_values can be [array] depending on xgboost/shap version
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class 1

    shap_for_instance = shap_values[0]
    shap_dict = dict(zip(feature_names, shap_for_instance))

    st.subheader("Feature contributions (SHAP)")
    contrib_df = (
        pd.DataFrame({"feature": feature_names, "shap_value": shap_for_instance})
        .sort_values("shap_value", key=lambda s: s.abs(), ascending=False)
    )

    st.dataframe(contrib_df)

    try:
        st.write("SHAP bar plot:")
        shap_fig = shap.plots.bar(shap_for_instance, show=False)
        st.pyplot(shap_fig)
    except Exception as e:
        st.write("Could not render SHAP bar plot in this environment:", e)

    # Simple retrieval over policy docs
    keywords = []
    if debt_to_income > 0.5:
        keywords.append("debt-to-income")
    if annual_revenue < 150_000:
        keywords.append("revenue")
    if years_trading < 3:
        keywords.append("trading history")
    if late_payments > 0:
        keywords.append("late payments")
    if sector_risk > 0.4:
        keywords.append("high-risk sectors")

    retrieved = simple_keyword_retrieval(policies_text, keywords)
    justification_md = build_text_justification(pred_label, shap_dict, retrieved)

    st.subheader("Policy-grounded explanation")
    st.markdown(justification_md)
