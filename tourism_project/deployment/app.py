import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the tourism model
model_path = hf_hub_download(
    repo_id="Swe20311/tourism-package-prediction", 
    filename="best_tourism_model_v1.joblib"
)
model = joblib.load(model_path)

# Streamlit UI for Wellness Tourism Package Prediction
st.title("Wellness Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a customer purchasing the Wellness Tourism Package
based on their profile and interaction details. Please enter the customer information below
to get a prediction.
""")

# User input fields based on dataset description
age = st.number_input("Age", min_value=18, max_value=100, value=30)
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Other"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=2)
passport = st.selectbox("Passport", [0, 1])
own_car = st.selectbox("Own Car", [0, 1])
num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=25000)
pitch_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
num_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=2)
duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=15)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeof_contact,
    'CityTier': city_tier,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': num_person_visiting,
    'PreferredPropertyStar': preferred_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': num_trips,
    'Passport': passport,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': num_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income,
    'PitchSatisfactionScore': pitch_score,
    'ProductPitched': product_pitched,
    'NumberOfFollowups': num_followups,
    'DurationOfPitch': duration_pitch
}])

# Prediction
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Package Purchased" if prediction == 1 else "No Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
