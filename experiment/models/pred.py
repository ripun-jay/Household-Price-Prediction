import joblib
import pandas as pd
import numpy as np

# RELOAD THE MODEL
final_model_reloaded= joblib.load("HouseXprice.pkl")

# User Interation


print("provide the details following\n\n")
longitude = int(input("longitude:  "))
latitude = int(input("latitude:  "))
housing_median_age = int(input("housing_median_age:  "))
total_rooms = int(input("total_rooms:  "))
total_bedrooms = int(input("total_bedrooms:  "))
population = int(input("population:  "))
households = int(input("households:  "))
median_income = int(input("median_income:  "))
ocean_proximity = input("ocean_proximity:  ")

# prediction logic
data = np.array([longitude, latitude, housing_median_age, total_rooms,
       total_bedrooms, population, households, median_income,
       ocean_proximity]).reshape(1,-1)

clms = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'ocean_proximity']

df = pd.DataFrame(data, columns= clms)

# prediction
result= final_model_reloaded.predict(df)

# print the prediction
print(f"\n\t\t This House Shound be Around: {result}")