from flask import Flask, render_template, request
import main
from main import final_model_reloaded as model
import numpy as np
import pandas as pd


app = Flask(__name__)

@app.route("/", methods= ["GET", "POST"])
def Home():
    if request.method == "POST":
        longitude = request.form.get("longitude")
        latitude = request.form.get("latitude")
        housing_median_age = request.form.get("housing_median_age")
        total_rooms = request.form.get("total_rooms")
        total_bedrooms = request.form.get("total_bedrooms")
        population = request.form.get("population")
        households = request.form.get("households")
        median_income = request.form.get("median_income")
        ocean_proximity = request.form.get("ocean_proximity")


        # prediction logic
        data = np.array([longitude, latitude, housing_median_age, total_rooms,
            total_bedrooms, population, households, median_income,
            ocean_proximity]).reshape(1,-1)

        clms = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income',
            'ocean_proximity']

        df = pd.DataFrame(data, columns= clms)

        val = model.predict(df)

        val = f"Your house Price should be around {val[0]}"




        
        return render_template('index.html', house_value= val)
    
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/noise")
def noise():
    return render_template("noise.html")

@app.route("/image")
def image():
    return render_template("image.html")


# app.run(debug=True)
