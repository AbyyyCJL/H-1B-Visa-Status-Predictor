import streamlit as st
import joblib
import numpy as np
import io

# Function to load trained model
def load_model():
    uploaded_model = st.file_uploader("Upload your trained model (.pkl)", type=["pkl"])
    if uploaded_model is not None:
        model = joblib.load(io.BytesIO(uploaded_model.read()))  # Read as bytes
        return model
    return None

# Function to load label encoder
@st.cache_resource
def load_label_encoder():
    return joblib.load("label_encoder.pkl")

# Set page configuration
st.set_page_config(page_title="H1B LCA Prediction", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home - Prediction", "Power BI Dashboard"])

# Page 1: Prediction System
if page == "Home - Prediction":
    st.title("H1B LCA Prediction System")

    model = load_model()
    label_encoder = load_label_encoder()

    if model is not None:
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
        input_data = np.array([[wage_rate, prevailing_wage, employer_state, worksite_state, soc_code, 
                                total_positions, full_time, h1b_dependent, willful_violator, naics_code]])

        if st.button("Predict"):
            # Ensure input shape matches expected dimensions
            if input_data.shape[1] != model.n_features_in_:
                st.error(f"Incorrect input shape: Expected {model.n_features_in_} features, got {input_data.shape[1]}.")
            else:
                prediction = model.predict(input_data)
                final_prediction = label_encoder.inverse_transform(prediction)  # Convert to category
                st.success(f"Predicted Visa Status: {final_prediction[0]}")

    else:
        st.warning("Please upload a trained model (.pkl) to proceed.")

# Page 2: Power BI Dashboard
elif page == "Power BI Dashboard":
    st.title("Power BI Dashboard")
    st.subheader("Visual Insights on H1B Visa Applications")

    # Embed Power BI using iframe
    power_bi_url = "https://app.powerbi.com/view?r=eyJrIjoiMTJmNmE4MzUtZWM4YS00NTljLTg4MjAtNjk4N2E4MGE0N2RiIiwidCI6IjNjYjkxMTI3LTkyNDMtNGQ1Yy04NWJiLTM2Zjc4YTIwMDA2MiJ9"
    st.markdown(
        f'<iframe width="100%" height="600px" src="{power_bi_url}" frameborder="0" allowFullScreen="true"></iframe>',
        unsafe_allow_html=True
    )
