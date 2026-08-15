import streamlit as st
import joblib
import pandas as pd
import sklearn  # Required so pickle/joblib resolves sklearn classes cleanly

st.set_page_config(
    page_title="Hospital Bed Occupancy Predictor",
    page_icon="🏥",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

st.title("🏥 Hospital Bed Occupancy Predictor")
st.write("Predict overall hospital bed occupancy based on capacity metrics and county demographics.")

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model.pkl: {e}")
    st.stop()

with st.form("occupancy_form"):
    st.subheader("Hospital Capacity Inputs")
    col1, col2 = st.columns(2)
    with col1:
        staffed_all = st.number_input("Staffed All Beds", min_value=0.0, value=100.0)
        licensed_all = st.number_input("Licensed All Beds", min_value=0.0, value=120.0)
        population = st.number_input("Population", min_value=0.0, value=50000.0)
        pop_65 = st.number_input("Population (65+)", min_value=0.0, value=8000.0)
    with col2:
        icu_beds = st.number_input("Staffed ICU Beds", min_value=0.0, value=15.0)
        icu_rate = st.number_input("ICU Bed Occupancy Rate (0 to 1)", min_value=0.0, max_value=1.0, value=0.65)
        pop_20 = st.number_input("Population (20+)", min_value=0.0, value=38000.0)

    st.subheader("Location Details")
    col3, col4, col5 = st.columns(3)
    with col3:
        state = st.text_input("State", value="CA")
    with col4:
        county = st.text_input("County Name", value="Los Angeles")
    with col5:
        icu_source = st.text_input("ICU Bed Source", value="Facility aggregation")

    submit_button = st.form_submit_button("Predict Occupancy Rate")

if submit_button:
    input_df = pd.DataFrame([{
        "Staffed All Beds": staffed_all,
        "Staffed ICU Beds": icu_beds,
        "Licensed All Beds": licensed_all,
        "ICU Bed Occupancy Rate": icu_rate,
        "Population": population,
        "Population (20+)": pop_20,
        "Population (65+)": pop_65,
        "State": state.strip(),
        "County Name": county.strip(),
        "ICU Bed Source": icu_source.strip()
    }])

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted All Bed Occupancy Rate: **{round(float(prediction), 4)}**")
    except Exception as e:
        st.error(f"Error generating prediction: {e}")
