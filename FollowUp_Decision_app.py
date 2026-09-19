#Libraries
import pandas as pd
from datetime import datetime,timedelta
import joblib

#Variables
"""Create a dataset in form of dictionary.
Features: Age,Gender,Department,Currency([GBP(£):1.1667,USD($):0.8580,INR(Rs):0090,EUR(€):1.0000]),Amount Charge,
Booking Date,Waiting Days"""

encoder = joblib.load("Hotencoder.pkl")
model = joblib.load("FollowUp_Decision_Model.pkl")
attention_list = []
patient_data= {"Age":25, "Gender":"female", "Department":"General","Currency":"USD($)", "Amount Charge": 3,
           "Booking Date":"2025-02-25", "Waiting Days":15}

"""This code-block runs once enter button is clicked(Function_Block) 
Note: range of waiting days is a slider and the factor that selects the appointment day."""

if patient_data["Currency"] == "GBP(£)":
    patient_data["Amount Charge"] *= 1.1667
elif patient_data["Currency"] == "USD($)":
    patient_data["Amount Charge"] *= 0.8580
elif patient_data["Currency"] == "INR(Rs)":
    patient_data["Amount Charge"] *= 0.0560
else:
    patient_data["Amount Charge"] *= 1.0000

booking_date = datetime.strptime(patient_data["Booking Date"], "%Y-%m-%d")
appointment_date = booking_date + timedelta(days=patient_data["Waiting Days"])

patient_data["Amount Charge(€)"] = patient_data.pop("Amount Charge")
patient_data["Appointment Date"] = appointment_date.strftime("%Y-%m-%d")
patient_data.pop("Currency")
print(patient_data)

"""Once predict button is clicked this code block runs(Function_Block)
Convert to dataframe, drop the booking and appointment date column, and arrange the columns in the order 
they are feed to the model:[Age,Gender,Department,Amount Charge(€),Waiting Days]"""
patient_df = pd.DataFrame([patient_data])
patient_df.drop(["Booking Date","Appointment Date"],axis=1,inplace=True)
order = ["Age","Gender","Department","Amount Charge(€)","Waiting Days"]
patient_df = patient_df[order]

#Encode and predict
patient_encoded = encoder.transform(patient_df)
follow_up = model.predict(patient_encoded)

#If the outcome is yes, add the dataset to a list that's downloadable and also create deletable.
if follow_up == "yes":
    print("This patient needs attention.")
    attention_list.append(patient_data)

""""Print attention list"""
print(attention_list)