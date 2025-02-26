from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os

# Correct file paths
model_path = os.path.abspath("../../models/price_by_sub_area_lat_lon_model.pkl")
data_path = os.path.abspath("../../data/processed/X_test.csv")

# Load trained model
model = joblib.load(model_path)

# Load dataset
X_test = pd.read_csv(data_path)

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None  # Default value
    suburbs = sorted(X_test["Suburb"].unique())  # Ensure using X_test

    # Default input values (Ensure this exists for GET requests)
    input_data = {
        "suburb": "",
        "area": "",
        "latitude": "",
        "longitude": "",
    }

    if request.method == "POST":
        try:
            # Get form values
            input_data["suburb"] = request.form["suburb"]
            input_data["area"] = request.form["area"]
            input_data["latitude"] = request.form["latitude"]
            input_data["longitude"] = request.form["longitude"]

            # Convert values to float
            area = float(input_data["area"])
            latitude = float(input_data["latitude"])
            longitude = float(input_data["longitude"])

            # Create DataFrame for model prediction
            df = pd.DataFrame(
                {
                    "Suburb": [input_data["suburb"]],
                    "BuildingArea": [area],
                    "Latitude": [latitude],
                    "Longitude": [longitude],
                }
            )

            df = df[model.feature_names_in_]  # Ensures correct feature order
            prediction = model.predict(df).round(2)[0]  # Gets prediction
            formatted_price = f"${prediction:,.2f}"  # Format price

        except Exception as e:
            print(f"Error during prediction: {e}")  # Log error in the backend
            formatted_price = "Error: Invalid Input"

    return render_template(
        "index.html", suburbs=suburbs, prediction=formatted_price, input_data=input_data
    )
