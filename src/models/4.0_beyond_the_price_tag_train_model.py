# Import necessary libraries
import joblib
import pandas as pd  # add pandas to the libraries
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from ipywidgets import Dropdown, FloatSlider, IntSlider, interact
import sys

sys.path.append("../data/updated_wrangle_module.py")
# Import the DataTransformer class from the 'updated_wrangle_module.py' file
from updated_wrangle_module import DataTransformer


# ----------------------------------------------------------------------------------------------
# 1. Use the DataTransformer class to prepare the data
# ----------------------------------------------------------------------------------------------

# Instantiate the DataTransformer class
DT = DataTransformer()

# Use the 'updated_wrangle' method to process the data
df = DT.updated_wrangle("../../data/raw/Melbourne_housing_FULL.csv")
print("df shape:", df.shape)
df.head()
df.info()

# ----------------------------------------------------------------------------------------------
# 2. Split the data into features and target
# ----------------------------------------------------------------------------------------------

# Split the data into features and target
X = df.drop(columns="Price_USD")
y = df["Price_USD"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# ----------------------------------------------------------------------------------------------
# 3. Build Model Baseline
# ----------------------------------------------------------------------------------------------

y_mean = y_train.mean()
y_mean

# Create a dumb model
y_pred_baseline = [y_mean] * len(y_train)
y_pred_baseline[:5]
len(y_pred_baseline) == len(y_train)

# ----------------------------------------------------------------------------------------------
# Calculate the baseline mean absolute error
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)
print("Mean house price", round(y_mean, 2))
print("Baseline MAE:", round(mae_baseline, 2))
"""
If an apartment 'Type' 'h' is always predicted at price $746,603.11
the the predictions would be off by an average of $276,482.9. It also means that
the model needs to have mean absolute error below $276,482.9 in order to be useful.
"""

# ----------------------------------------------------------------------------------------------
# 4. Iterate on the model
# ----------------------------------------------------------------------------------------------

# Instantiate the model
model = make_pipeline(OneHotEncoder(use_cat_names=True), SimpleImputer(), Ridge())

# Fit the model
model.fit(X_train, y_train)

# Check your that model is fitted
check_is_fitted(model)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
X_train.info()
y_train.info()
len(df) * 0.8
len(df) * 0.2

# ----------------------------------------------------------------------------------------------
# 6. Evaluate the model performance on the training set
# ----------------------------------------------------------------------------------------------

# Make predictions on the training set
y_pred_training = model.predict(X_train)
y_pred_training[:5]

# Calculate the MAE for predictions in y_pred_training against the actual values in y_train
mae_training = mean_absolute_error(y_train, y_pred_training)
print("Training MAE:", round(mae_training, 2))

# ----------------------------------------------------------------------------------------------
# 5. Make predictions on the test set and evaluate the model performance on the test set
# ----------------------------------------------------------------------------------------------

# Make predictions on the test set
y_pred = pd.Series(model.predict(X_test))
y_pred[:5]

# Calculate the MAE for predictions in y_pred against the actual values in y_test
mae_testing = mean_absolute_error(y_test, y_pred)
print("Testing MAE:", round(mae_testing, 2))

# ----------------------------------------------------------------------------------------------
# 7.  Communicate the results
# ----------------------------------------------------------------------------------------------


# Create a function 'price_predictor' that returns the model's prediction for a house price.
def price_predictor(suburb, area, latitude, longitude):
    """
    Predict the price of a house based on its suburb, latitude, longitude and area.

    Parameters:
    suburb (str): The suburb where the house is located.
    area (float): The surface area of the building in square meters.
    latitude (float): The latitude coordinate of the house.
    longitude (float): The longitude coordinate of the house.

    Returns:
    str: A string indicating the predicted apartment price, rounded to two decimal places.
    """
    data = {
        "Suburb": suburb,
        "BuildingArea": area,
        "Latitude": latitude,
        "Longitude": longitude,
    }
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(df).round(2)[0]
    return f"Predicted House price: ${prediction}"


# ----------------------------------------------------------------------------------------------

# Test the 'price_predictor' function for 'Fitzory' suburb in 'Melbourne'
predicted_price = price_predictor(
    suburb="Fitzroy",
    area=120.5,
    latitude=-37.7981,
    longitude=144.9789,
)

print(predicted_price)

# ----------------------------------------------------------------------------------------------

# Interactive widget for house price prediction
interact(
    price_predictor,
    suburb=Dropdown(options=sorted(X_train["Suburb"].unique()), description="Suburb:"),
    area=IntSlider(
        min=X_train["BuildingArea"].min(),
        max=X_train["BuildingArea"].max(),
        value=X_train["BuildingArea"].mean(),
        description="BuildingArea:",
    ),
    latitude=FloatSlider(
        min=X_train["Latitude"].min(),
        max=X_train["Latitude"].max(),
        step=0.01,
        value=X_train["Latitude"].mean(),
        description="Latitude:",
    ),
    longitude=FloatSlider(
        min=X_train["Longitude"].min(),
        max=X_train["Longitude"].max(),
        step=0.01,
        value=X_train["Longitude"].mean(),
        description="Longitude:",
    ),
)


# ----------------------------------------------------------------------------------------------
# 8. Save and export the X_train features and the model
# ----------------------------------------------------------------------------------------------

# Export X_train to the 'data' folder as a CSV file
X_train.to_csv("../../data/processed/X_train.csv", index=False)

# Save the model to 'models' folder using joblib
joblib.dump(model, "../../models/price_by_sub_area_lat_lon_model.pkl")

# Load the model from the 'models' folder
model = joblib.load("../../models/price_by_sub_area_lat_lon_model.pkl")

# Check the model
model
