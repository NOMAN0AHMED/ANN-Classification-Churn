import numpy as np
import pickle
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.models import load_model
#load the train model pickle file 
model = load_model("models.h5")




#load  pickle LEG file
with open("LEG.pkl","rb") as file:
    LED=pickle.load(file) 
#load  pickle OHE file

with open("OHE.pkl","rb") as file:
    OHE=pickle.load(file) 
#load  pickle Scalar file
with open("scalar.pkl","rb") as file:
    scalar=pickle.load(file) 

input_data = [
    {"gender": "Male", "hours_studied": 5, "attendance_percent": 92, "Geography": "Pakistan"},
    {"gender": "Female", "hours_studied": 4, "attendance_percent": 85, "Geography": "India"},
    {"gender": "Male", "hours_studied": 2, "attendance_percent": 70, "Geography": "USA"},
    {"gender": "Female", "hours_studied": 6, "attendance_percent": 95, "Geography": "UK"},
    {"gender": "Male", "hours_studied": 3, "attendance_percent": 78, "Geography": "Canada"},
    {"gender": "Female", "hours_studied": 5, "attendance_percent": 88, "Geography": "Germany"},
    {"gender": "Male", "hours_studied": 1, "attendance_percent": 60, "Geography": "Australia"},
    {"gender": "Female", "hours_studied": 4, "attendance_percent": 90, "Geography": "France"},
    {"gender": "Male", "hours_studied": 3, "attendance_percent": 72, "Geography": "Japan"},
    {"gender": "Female", "hours_studied": 5, "attendance_percent": 93, "Geography": "Pakistan"}
]


df = pd.DataFrame(input_data)
input_data = pd.get_dummies(df, columns=["Geography"], drop_first=False)
print(input_data)
input_data["gender"]=LED.transform(df["gender"])
print(input_data)
input_data=scalar.fit_transform(input_data)
print(input_data)
pridiction=model.predict(input_data)
print(pridiction)
pridiction_proba=pridiction[0][0]
print(pridiction_proba)
if pridiction_proba<0.5:
    print("Custromer likely to churn")
else:
    print("Custromer not likely to churn")