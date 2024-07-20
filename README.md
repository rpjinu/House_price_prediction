# House_price_prediction
ALL python jupyter notebook

Here's a comprehensive README file for your GitHub repository, detailing all the steps for the house price prediction model, saving the model, using a Flask app, and then creating an HTML interface for user input and prediction.

markdown
Copy code
# House Price Prediction Model

This repository contains the code and resources for building a house price prediction model, saving the model, deploying it using a Flask web application, and creating an HTML interface for user input and prediction.

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Training](#model-training)
4. [Saving the Model](#saving-the-model)
5. [Setting up the Flask App](#setting-up-the-flask-app)
6. [Creating the HTML Template](#creating-the-html-template)
7. [Running the Application](#running-the-application)
8. [Usage](#usage)
9. [Tech Stack](#tech-stack)
10. [Contact](#contact)

## Overview

This project aims to predict house prices based on various features such as the number of bedrooms, bathrooms, living area, lot area, and other house-related attributes. The model is built using a RandomForestRegressor and is deployed using a Flask web application. Users can input house features through an HTML form, and the application will predict the house price.

## Dataset

The dataset used for this project contains various features related to house properties and their corresponding prices. The dataset should have the following columns:
- No of bedrooms
- No of bathrooms
- Living area
- Lot area
- No of floors
- Waterfront present
- No of views
- House condition
- House grade
- House area (excluding basement)
- Area of the basement
- Built Year
- Renovation Year
- Living area renovation
- Lot area renovation
- No of schools nearby
- Distance from the airport
- Price

## Model Training

1. Load the dataset and define features and target variables.
2. Split the dataset into training and testing sets.
3. Train a RandomForestRegressor model using the training set.

Example code for model training:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
df = pd.read_csv('house_prices.csv')

# Define features and target
X = df.drop(['Price'], axis=1)
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
#Saving the Model
import pickle

# Save the model to a file
with open('house_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)
#Setting up the Flask App
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('house_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form
    features = np.array([[float(form_data['No of bedrooms']),
                          float(form_data['No of bathrooms']),
                          float(form_data['living area']),
                          float(form_data['lot area']),
                          float(form_data['No of floors']),
                          float(form_data['waterfront present']),
                          float(form_data['No of views']),
                          float(form_data['house condition']),
                          float(form_data['house grade']),
                          float(form_data['house area(excluding basement)']),
                          float(form_data['Area of the basement']),
                          float(form_data['Built Year']),
                          float(form_data['Renovation Year']),
                          float(form_data['living_area_renov']),
                          float(form_data['lot_area_renov']),
                          float(form_data['No of schools nearby']),
                          float(form_data['Distance from the airport'])]])

    prediction = model.predict(features)
    return render_template('index.html', prediction_text=f'Predicted House Price: ${prediction[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
#Creating the HTML Template
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Prediction</title>
</head>
<body>
    <h1>House Price Prediction</h1>
    <form action="/predict" method="post">
        <label for="No of bedrooms">Number of Bedrooms:</label>
        <input type="text" id="No of bedrooms" name="No of bedrooms"><br><br>
        <label for="No of bathrooms">Number of Bathrooms:</label>
        <input type="text" id="No of bathrooms" name="No of bathrooms"><br><br>
        <label for="living area">Living Area (sqft):</label>
        <input type="text" id="living area" name="living area"><br><br>
        <label for="lot area">Lot Area (sqft):</label>
        <input type="text" id="lot area" name="lot area"><br><br>
        <label for="No of floors">Number of Floors:</label>
        <input type="text" id="No of floors" name="No of floors"><br><br>
        <label for="waterfront present">Waterfront Present (0 or 1):</label>
        <input type="text" id="waterfront present" name="waterfront present"><br><br>
        <label for="No of views">Number of Views:</label>
        <input type="text" id="No of views" name="No of views"><br><br>
        <label for="house condition">House Condition:</label>
        <input type="text" id="house condition" name="house condition"><br><br>
        <label for="house grade">House Grade:</label>
        <input type="text" id="house grade" name="house grade"><br><br>
        <label for="house area(excluding basement)">House Area (excluding basement):</label>
        <input type="text" id="house area(excluding basement)" name="house area(excluding basement)"><br><br>
        <label for="Area of the basement">Area of the Basement (sqft):</label>
        <input type="text" id="Area of the basement" name="Area of the basement"><br><br>
        <label for="Built Year">Built Year:</label>
        <input type="text" id="Built Year" name="Built Year"><br><br>
        <label for="Renovation Year">Renovation Year:</label>
        <input type="text" id="Renovation Year" name="Renovation Year"><br><br>
        <label for="living_area_renov">Living Area Renovation (sqft):</label>
        <input type="text" id="living_area_renov" name="living_area_renov"><br><br>
        <label for="lot_area_renov">Lot Area Renovation (sqft):</label>
        <input type="text" id="lot_area_renov" name="lot_area_renov"><br><br>
        <label for="No of schools nearby">Number of Schools Nearby:</label>
        <input type="text" id="No of schools nearby" name="No of schools nearby"><br><br>
        <label for="Distance from the airport">Distance from the Airport (miles):</label>
        <input type="text" id="Distance from the airport" name="Distance from the airport"><br><br>
        <input type="submit" value="Predict">
    </form>

    {% if prediction_text %}
        <h2>{{ prediction_text }}</h2>
    {% endif %}
</body>
</html>
#Usage
Open your web browser and go to http://127.0.0.1:5000/.
Fill in the form with the relevant house features.
Click "Predict" to get the predicted house price.

