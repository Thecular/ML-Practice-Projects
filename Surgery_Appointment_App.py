#Libraries
import streamlit as st
import pandas as pd
import joblib
from datetime import timedelta

#Variables
st.header("Surgery Appointment App",anchor="Surgery Appointment App",divider="gray",text_alignment="center")
encoder = joblib.load("Hotencoder.pkl")
model = joblib.load("Appointment_Model.pkl")

st.write("Enter the required patient's data for appointment validation.")
if "appointment_list" not in st.session_state:
    st.session_state.appointment_list = []
name = (st.text_input("Enter patient's name: ")).title()
age = st.number_input("Age",min_value=0,max_value=150,step=1)#input patient name later
gender = st.selectbox("Gender",("male","female"),index=None)
department = st.selectbox("Department",("Cardiology","Orthopedics","Neurology","General"),index=None)
currency = st.selectbox("Currency",("GBP(£)","USD($)","INR(Rs)","EUR(€)"),index=None)
amount = st.number_input("Amount",min_value=0,max_value=1000000)
booking_date = st.date_input("Booking Date",format="YYYY-MM-DD")
waiting_days = st.slider("Waiting Days",min_value=0,max_value=1000)

#Data-Frame creation
def patient(patient_data):
    if patient_data["Currency"] == "GBP(£)":
        patient_data["Amount Charge"] *= 1.1667
    elif patient_data["Currency"] == "USD($)":
        patient_data["Amount Charge"] *= 0.8580
    elif patient_data["Currency"] == "INR(Rs)":
        patient_data["Amount Charge"] *= 0.0560
    else:
        patient_data["Amount Charge"] *= 1.0000

    appointment_date = booking_date + timedelta(days=patient_data["Waiting Days"])
    patient_data["Amount Charge(€)"] = patient_data.pop("Amount Charge")
    patient_data["Appointment Date"] = appointment_date.strftime("%Y-%m-%d")
    patient_data.pop("Currency")
    return patient_data

enter = st.button("Enter")
if "patient_data" not in st.session_state:
    st.session_state.patient_data = None
if enter:
    patient_data = {"Name":name, "Age":age, "Gender":gender, "Department":department, "Currency":currency, "Amount Charge":amount,
                    "Booking Date":booking_date, "Waiting Days":waiting_days}
    patient_data = patient(patient_data)
    st.session_state.patient_data = patient_data
if st.session_state.patient_data is not None:
    st.write(pd.DataFrame([st.session_state.patient_data]))

#Prediction
def prediction(patient_data):
    patient_df = pd.DataFrame([patient_data])
    patient_df.drop(["Name", "Booking Date", "Appointment Date"], axis=1, inplace=True)
    order = ["Age", "Gender", "Department", "Amount Charge(€)", "Waiting Days"]
    patient_df = patient_df[order]

# Encode and predict
    patient_encoded = encoder.transform(patient_df)
    follow_up = model.predict(patient_encoded)[0]

# Create a list that's downloadable.
    if follow_up == "yes":
        st.success("This patient needs attention.")
        st.session_state.appointment_list.append(patient_data)
        return st.session_state.appointment_list
    else:
        st.info("This patient doesn't need attention.")
        return st.session_state.appointment_list
st.divider()
predict = st.button("Predict")
if predict:
    if st.session_state.patient_data is not None:
        result = prediction(st.session_state.patient_data)
    st.divider()
    if st.session_state.appointment_list is not None:
        st.write("**APPOINTMENT LIST**")
        st.table(result)
col1,col2 = st.columns(2)
with col1:
    csv = pd.DataFrame([st.session_state.appointment_list]).to_csv(index=False)
    st.download_button("Download File",data=csv,file_name="Surgery Appointment.csv",mime="text/csv")