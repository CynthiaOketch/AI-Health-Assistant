import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64
import numpy as np

model = joblib.load("models/best_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")
explainer = shap.TreeExplainer(model)

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

st.set_page_config(page_title="Health Risk Prediction", layout="centered")
st.title("AI Health Risk Prediction Assistant")
st.markdown("Enter patient information to predict diabetes risk and view explainability insights.")

st.sidebar.header("Patient Information")
pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 1)
glucose = st.sidebar.slider("Glucose Level", 0, 200, 100)
bp = st.sidebar.slider("Blood Pressure (mm Hg)", 0, 122, 70)
skin_thickness = st.sidebar.slider("Skin Thickness (mm)", 0, 99, 20)
insulin = st.sidebar.slider("Insulin Level (mu U/ml)", 0, 846, 79)
bmi = st.sidebar.slider("BMI", 0.0, 67.1, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.47)
age = st.sidebar.slider("Age", 21, 81, 30)

input_data = pd.DataFrame([{
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": bp,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": dpf,
    "Age": age
}])

X_input = preprocessor.transform(input_data)

if st.button("Predict Risk"):
    prediction = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]
    label = "High Risk" if prediction == 1 else "Low Risk"

    st.subheader("Prediction")
    if prediction == 1:
        st.error(f"Risk Level: {label}")
    else:
        st.success(f"Risk Level: {label}")
    st.markdown(f"**Probability of diabetes:** {proba:.2%}")

    st.subheader("Explanation (SHAP)")
    shap_values = explainer.shap_values(X_input)
    fig, ax = plt.subplots()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_input[0],
            feature_names=FEATURES
        ),
        max_display=8,
        show=False
    )
    st.pyplot(fig)

    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Patient Risk Prediction Report", ln=True, align="C")
        pdf.ln(10)
        for col, val in input_data.iloc[0].items():
            pdf.cell(200, 10, txt=f"{col}: {val}", ln=True)
        pdf.ln(5)
        pdf.cell(200, 10, txt=f"Risk Level: {label}", ln=True)
        pdf.cell(200, 10, txt=f"Probability: {proba:.2%}", ln=True)
        return pdf.output(dest="S").encode("latin1")

    pdf_bytes = generate_pdf()
    b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="prediction_report.pdf">Download Report as PDF</a>'
    st.markdown(href, unsafe_allow_html=True)
