from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os

# Correct file paths
model_path = os.path.abspath("../../models/price_by_sub_area_lat_lon_model.pkl")
data_path = os.path.abspath("../../data/processed/X_train.csv")

# Load trained model
model = joblib.load(model_path)

# Load dataset
X_train = pd.read_csv(data_path)

# Initialize Flask app
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None  # Default value
    suburbs = sorted(X_train["Suburb"].unique())  # Get suburb list

    if request.method == "POST":
        # Get form values
        suburb = request.form["suburb"]
        area = float(request.form["area"])
        latitude = float(request.form["latitude"])
        longitude = float(request.form["longitude"])

        # Create DataFrame for model prediction
        df = pd.DataFrame(
            {
                "Suburb": [suburb],
                "BuildingArea": [area],
                "Latitude": [latitude],
                "Longitude": [longitude],
            }
        )
        df = df[model.feature_names_in_]  # Ensure correct feature order
        prediction = model.predict(df).round(2)[0]  # Get prediction

    return render_template("index.html", suburbs=suburbs, prediction=prediction)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get PORT from environment variable
    app.run(host="0.0.0.0", port=port, debug=False)
