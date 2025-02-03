import streamlit as st
import joblib
import numpy as np
import io

# Load trained model
def load_model():
    uploaded_model = st.file_uploader("Upload your trained model (.pkl)", type=["pkl"])
    if uploaded_model is not None:
        model = joblib.load(io.BytesIO(uploaded_model.read()))  # Read as bytes
        return model
    return None

# Load Label Encoder
def load_label_encoder():
    uploaded_encoder = st.file_uploader("Upload your label encoder (.pkl)", type=["pkl"])
    if uploaded_encoder is not None:
        label_encoder = joblib.load(io.BytesIO(uploaded_encoder.read()))
        return label_encoder
    return None

# Streamlit UI
st.set_page_config(page_title="H1B LCA Prediction", layout="wide")
st.title("H1B LCA Prediction System")

model = load_model()
label_encoder = load_label_encoder()

if model is not None and label_encoder is not None:
    st.subheader("Enter Features for Prediction")

    # Define input fields
    wage_rate = st.number_input("Enter Wage Rate of Pay (From)", min_value=0.0, step=0.01)
    prevailing_wage = st.number_input("Enter Prevailing Wage", min_value=0.0, step=0.01)
    employer_state = st.number_input("Enter Employer State (Numeric Code)", min_value=1, step=1)
    worksite_state = st.number_input("Enter Worksite State (Numeric Code)", min_value=1, step=1)
    soc_code = st.number_input("Enter SOC Code (Numeric Code)", min_value=1, step=1)
    total_positions = st.number_input("Enter Total Worker Positions", min_value=1, step=1)
    full_time = st.selectbox("Is it a Full-Time Position?", ["Yes", "No"])
    h1b_dependent = st.selectbox("Is Employer H-1B Dependent?", ["Yes", "No"])
    willful_violator = st.selectbox("Is Employer a Willful Violator?", ["Yes", "No"])
    naics_code = st.number_input("Enter NAICS Code", min_value=1, step=1)

    # Convert categorical values to numeric
    full_time = 1 if full_time == "Yes" else 0
    h1b_dependent = 1 if h1b_dependent == "Yes" else 0
    willful_violator = 1 if willful_violator == "Yes" else 0

    # Collect inputs into an array
    input_data = np.array([[wage_rate, prevailing_wage, employer_state, worksite_state, soc_code, total_positions, full_time, h1b_dependent, willful_violator, naics_code]])

    if st.button("Predict"):
        prediction = model.predict(input_data)
        final_prediction = label_encoder.inverse_transform(prediction)  # Convert to category
        st.success(f"Predicted Visa Status: {final_prediction[0]}")
else:
    st.warning("Please upload both the trained model (.pkl) and the label encoder (.pkl) to proceed.")
