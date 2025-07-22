import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from fpdf import FPDF
import base64
import io

# Load the trained model and preprocessor
model = joblib.load("models/best_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")
explainer = shap.TreeExplainer(model)

st.set_page_config(page_title="Health Risk Prediction", layout="centered")
st.title("🧬 AI Health Risk Prediction Assistant")
st.markdown("Enter patient information to predict disease risk and view explainability insights.")

# Sidebar form inputs
st.sidebar.header("Patient Information")
age = st.sidebar.slider("Age", 18, 90, 30)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 22.5)
glucose = st.sidebar.slider("Glucose Level", 50, 200, 100)
bp = st.sidebar.slider("Blood Pressure", 60, 180, 120)
insulin = st.sidebar.slider("Insulin Level", 15, 276, 100)
skin_thickness = st.sidebar.slider("Skin Thickness", 7, 99, 20)

# Example for categorical
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
smoker = st.sidebar.radio("Smoker?", ["Yes", "No"])

# Map categorical inputs
gender = 1 if gender == "Male" else 0
smoker = 1 if smoker == "Yes" else 0

input_data = pd.DataFrame([{
    "Age": age,
    "BMI": bmi,
    "Glucose": glucose,
    "BloodPressure": bp,
    "Insulin": insulin,
    "SkinThickness": skin_thickness,
    "Gender": gender,
    "Smoker": smoker
}])

# Preprocess input
X_input = preprocessor.transform(input_data)

# Predict
if st.button("🔍 Predict Risk"):
    prediction = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]
    label = "⚠️ High Risk" if prediction == 1 else "✅ Low Risk"

    st.subheader("Prediction")
    st.markdown(f"**Risk Level:** {label}")
    st.markdown(f"**Probability of disease:** {proba:.2%}")

    # SHAP explainability
    st.subheader("🔎 Explanation (SHAP)")
    shap_values = explainer.shap_values(X_input)
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap.Explanation(values=shap_values[1][0], base_values=explainer.expected_value[1], data=X_input[0]), max_display=8, show=False)
    st.pyplot(fig)

    # PDF export
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
    b64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="prediction_report.pdf">📄 Download Report as PDF</a>'
    st.markdown(href, unsafe_allow_html=True)
