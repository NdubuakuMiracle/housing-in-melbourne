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
    prediction = None
    input_suburb = ""
    input_area = ""
    input_lat = ""
    input_lon = ""
    suburbs = sorted(X_test["Suburb"].unique())

    if request.method == "POST":
        # Get form values
        input_suburb = request.form["suburb"]
        input_area = float(request.form["area"])
        input_lat = float(request.form["latitude"])
        input_lon = float(request.form["longitude"])

        # Create DataFrame
        df = pd.DataFrame(
            {
                "Suburb": [input_suburb],
                "BuildingArea": [input_area],
                "Latitude": [input_lat],
                "Longitude": [input_lon],
            }
        )
        df = df[model.feature_names_in_]
        prediction = model.predict(df).round(2)[0]
        prediction = f"{prediction:,.2f}"  # format like $1,200,000.00

    return render_template(
        "index.html",
        suburbs=suburbs,
        prediction=prediction,
        input_suburb=input_suburb,
        input_area=input_area,
        input_lat=input_lat,
        input_lon=input_lon,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get PORT from environment variable
    app.run(host="0.0.0.0", port=port, debug=False)
