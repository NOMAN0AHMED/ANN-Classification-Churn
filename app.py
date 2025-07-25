import streamlit as st
import numpy as np
import pickle
import pandas as pd 
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder ,StandardScaler ,OneHotEncoder
#load the train model pickle file 
model = tf.keras.models.load_model("models.h5")

#load  pickle LEG file
with open("LEG.pkl","rb") as file:
    LED=pickle.load(file) 
#load  pickle OHE file

with open("OHE.pkl","rb") as file:
    OHE=pickle.load(file) 
#load  pickle Scalar file
with open("scalar.pkl","rb") as file:
    scalar=pickle.load(file) 
#streamlit APP
st.title("Custumer churn pridiction")

#user input
Geography = st.selectbox("Geography", OHE.categories_[0])
Gender = st.selectbox("Gender", LED.classes_)
Age = st.slider("Age", 18, 92)
Balance = st.number_input("Balance")
CreditScore = st.number_input("Credit Score")
EstimatedSalary = st.number_input("Estimated Salary")
Tenure = st.slider("Tenure", 0, 10)
NumOfProducts = st.slider("Number of Products", 1, 4)
HasCrCard = st.selectbox("Has Credit Card", [0, 1])
IsActiveMember = st.selectbox("Is Active Member", [0, 1])

# Encode gender
Gender_encoded = LED.transform([Gender])[0]

# Prepare input
input_data = pd.DataFrame([{
    "CreditScore": CreditScore,
    "Geography": Geography,
    "Gender": Gender_encoded,
    "Age": Age,
    "Tenure": Tenure,
    "Balance": Balance,
    "NumOfProducts": NumOfProducts,
    "HasCrCard": HasCrCard,
    "IsActiveMember": IsActiveMember,
    "EstimatedSalary": EstimatedSalary
}])

# One-hot encode Geography
input_data = pd.get_dummies(input_data, columns=["Geography"], drop_first=False)

# Ensure all columns match training set
for col in scalar.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0  # Add missing column with 0

# Reorder to match training set
input_data = input_data[scalar.feature_names_in_]

# Scale the input
input_data_scaled = scalar.transform(input_data)

# Predict
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Output
st.write(f"Churn Probability: {prediction_proba:.2f}")

if prediction_proba < 0.5:
    st.success("Customer is NOT likely to churn.")
else:
    st.warning("Customer is LIKELY to churn.")
#source /Users/nomanmacbook/Desktop/ANN\ clas/myenv/bin/activate
