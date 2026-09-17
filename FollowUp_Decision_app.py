"""1] Get new_dataset in form of dictionary of the features used in training the model and convert to dataframe.
2] Features: Age,Gender,Department,Amount Charge(€),Apointment Date,Booking Date,Waiting Days
3] Types of currency to input: [GBP(£),USD($),INR(Rs),EUR(€)]
4] Create a new column for currency conversion to euros with the exchange rate:
rates = {"GBP":1.1667,"USD": 0.8580,"INR": 0.0090, "EUR": 1.0000}
5] Create a new column for waiting days, ie difference between appointment date and booking date
using the format that the waiting days get picked and appointment date is assigned.
6] Drop the booking date column
7] Arrange the columns in the order they are feed to the model:[Age,Gender,Department,Amount Charge(€),Apointment Date,Waiting Days]
8] Use sklearn onehotencode on the dataset: check if it applies the same way as the training dataset and checkout the date.
Also drop the original department.
9] Gender: female and male, Department:Cardiology,General,Neurology,Orthopedics
10] Predict the outcome
11] If the outcome is yes, add the dataset to a list and make it downloadable
12] Create an option to clear the list"""