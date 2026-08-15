import streamlit as st
import joblib
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Hospital Bed Occupancy Predictor",
    page_icon="🏥",
    layout="centered"
)

# Load the trained model pipeline
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("🏥 Hospital Bed Occupancy Predictor")
st.write("Enter hospital capacity and demographic metrics below to predict bed occupancy outcomes.")

# Input Form
with st.form("prediction_form"):
    st.subheader("Capacity Metrics")
    col1, col2 = st.columns(2)
    with col1:
        staffed_all = st.number_input("Staffed All Beds", min_value=0.0, value=100.0)
        licensed = st.number_input("Licensed All Beds", min_value=0.0, value=120.0)
        population = st.number_input("Population", min_value=0.0, value=50000.0)
        pop_65 = st.number_input("Population (65+)", min_value=0.0, value=8000.0)
    with col2:
        icu_beds = st.number_input("Staffed ICU Beds", min_value=0.0, value=15.0)
        icu_rate = st.number_input("ICU Bed Occupancy Rate", min_value=0.0, max_value=1.0, value=0.65)
        pop_20 = st.number_input("Population (20+)", min_value=0.0, value=38000.0)

    st.subheader("Location & Source Details")
    col3, col4, col5 = st.columns(3)
    with col3:
        state = st.text_input("State (e.g., CA, NY)", value="CA")
    with col4:
        county = st.text_input("County Name", value="Los Angeles")
    with col5:
        icu_source = st.text_input("ICU Bed Source", value="Facility aggregation")

    submitted = st.form_submit_button("Predict Occupancy")

if submitted:
    input_data = {
        "Staffed All Beds": staffed_all,
        "Staffed ICU Beds": icu_beds,
        "Licensed All Beds": licensed,
        "ICU Bed Occupancy Rate": icu_rate,
        "Population": population,
        "Population (20+)": pop_20,
        "Population (65+)": pop_65,
        "State": state.strip().upper(),
        "County Name": county.strip(),
        "ICU Bed Source": icu_source.strip()
    }
    
    input_df = pd.DataFrame([input_data])
    
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Target Metric: **{round(float(prediction), 4)}**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
