# Import necessary libraries
import joblib
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from category_encoders import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.utils.validation import check_is_fitted


# ----------------------------------------------------------------------------------------------
# 1. Load the transformed data
# ----------------------------------------------------------------------------------------------

# Load the transformed data
df = pd.read_csv("../../data/processed/processed_melbourne_data.csv")
df.head()
df.shape
df.info()
df["Suburb"].nunique()
df["Suburb"].value_counts().head(30)

# ----------------------------------------------------------------------------------------------
# 2. Split the data into feature matrix and target vector
# ----------------------------------------------------------------------------------------------

# Split the data into features and target
X = df[["Suburb"]]
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
print("Mean h price", round(y_mean, 2))
print("Baseline MAE:", round(mae_baseline, 2))
"""
If an apartment 'Type' 'h' is always predicted at price $752,859.19
the the predictions would be off by an average of $241,499.33. It also means that
the model needs to have mean absolute error below $241,499.33 in order to be useful.
"""

# ----------------------------------------------------------------------------------------------
# 4. Iterate on LinearRegression model and Ridge model
# ----------------------------------------------------------------------------------------------

# Instantiate the model
model_linear = make_pipeline(OneHotEncoder(use_cat_names=True), LinearRegression())

model_ridge = make_pipeline(OneHotEncoder(use_cat_names=True), Ridge())
# Fit the model_linear
model_linear.fit(X_train, y_train)
# Fit the model_ridge
model_ridge.fit(X_train, y_train)

# Updated model using 'Random Forest'
model_rf = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    RandomForestRegressor(n_estimators=100, random_state=42),
)
# Fit the model_rf
model_rf.fit(X_train, y_train)


# Check your that model is fitted
check_is_fitted(model_linear)
check_is_fitted(model_ridge)
check_is_fitted(model_rf)


# ----------------------------------------------------------------------------------------------
# 5. Evaluate the model performance on the training sets
# ----------------------------------------------------------------------------------------------

# Make predictions on the two training sets
y_pred_training_linear = model_linear.predict(X_train)
y_pred_training_linear[:5]
y_pred_training_ridge = model_ridge.predict(X_train)
y_pred_training_ridge[:5]

# Visualise the models performance prediction against actual value
plt.plot(y_train, label="Actutal")
plt.plot(y_pred_training_linear, label="Linear Regression")
plt.plot(y_pred_training_ridge, label="Ridge Regression")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Predicted vs Actual Values")
plt.legend()
plt.show()

# Calculate the MAE for predictions in y_pred_training_linear against the actual values in y_train
mae_training_linear = mean_absolute_error(y_train, y_pred_training_linear)
print("Training Linear MAE:", round(mae_training_linear, 2))

# Calculate the MAE for predictions in y_pred_training_ridge against the actual values in y_train
mae_training_ridge = mean_absolute_error(y_train, y_pred_training_ridge)
print("Training Ridge MAE:", round(mae_training_ridge, 2))


# Make predictions on the model_rf training sets
y_pred_training_rf = model_rf.predict(X_train)
y_pred_training_rf[:5]
# Calculate the MAE for predictions in y_pred_training_rf against the actual values in y_train
mae_training_rf = mean_absolute_error(y_train, y_pred_training_rf)
print("Training RandomForest MAE:", round(mae_training_rf, 2))

# ----------------------------------------------------------------------------------------------
# RandomForest training model visualisation
# Scatter Plot: Actual training vs. Predicted training
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=y_train, y=y_pred_training_rf, alpha=0.6
)  # Alpha for better visibility
plt.plot(
    [y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--", lw=2
)  # Diagonal Line (X and y coordinates for deal Predictions)
plt.xlabel("Actual Training Price (USD)")
plt.ylabel("Predicted Training Price (USD)")
plt.title("Random Forest: Actual vs. Predicted Training Prices")
plt.show()

# Residuals Histogram indicating where the residuals are zero (perfect predictions)
residuals = y_train - y_pred_training_rf
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.axvline(
    0, color="r", linestyle="--", linewidth=2
)  # Adds a red dashed vertical line
plt.xlabel("Prediction Error (Residuals)")
plt.ylabel("Frequency")
plt.title("Residual Training Distribution")
plt.show()

# ----------------------------------------------------------------------------------------------

"""
## Model Selection for Predicting `Price_USD` from `Suburb`

From the plot and training MAE, it is clear that both **Linear** and **Ridge Regression**  
are **not ideal** for this task. These models assume a continuous, linear relationship  
between `Suburb` and `Price_USD`, which is incorrect because `Suburb` is a **categorical** variable  
with **314 unique values**.

### **Issue with Numerical Encoding**
Even if `Suburb` is numerically encoded as I have done, the model will interpret these values  
as having an inherent order (e.g., `Suburb_A = 1, Suburb_B = 2`), which is **incorrect**  
because suburb names do not have a meaningful numerical relationship.

### **Why Regularization Won't Help**
Ridge Regression applies **L2 regularization** to reduce overfitting, but it does **not**  
solve the **core issue**: `Suburb` is categorical, not numerical.

### **Better Alternatives**
Instead of linear models, a tree-based model like:
- **Random Forest**
- **Gradient Boosting (XGBoost, CatBoost, LightGBM)**  

would be much better at handling categorical features like `Suburb`,  
as they can split on different suburbs without assuming a linear relationship.

"""
# Revert back to 'No. 4' and updte the model using 'Random Forest'

# ----------------------------------------------------------------------------------------------
# 6. Creat a function to make predictions and evaluate the models performances on the test set
# ----------------------------------------------------------------------------------------------


def evaluate_models(models, X_test, y_test):
    """
    Evaluates multiple regression models on the test set.

    Parameters:
    models (dict): Dictionary of model names and trained model instances.
    X_test (array-like): Test set features.
    y_test (array-like): Actual target values.

    Returns:
    DataFrame: Evaluation metrics for each model.
    """
    # Placeholder to store results for each model
    results = []

    # Loop through each model to make predictions and evaluate
    for name, model in models.items():
        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate the MAE for predictions against actual values in y_test
        mae = mean_absolute_error(y_test, y_pred)

        # Store results
        results.append({"Model": name, "MAE": mae})

    # Convert evaluation results into a DataFrame for better visualization
    return pd.DataFrame(results).sort_values(by="MAE")


# ----------------------------------------------------------------------------------------------
# Use evaluate_models to make predication
# Define the models in a dictionary
models = {
    "Linear Regression": model_linear,
    "Ridge Regression": model_ridge,
    "Random Forest": model_rf,
}

# Call the function and display results
evaluate_models(models, X_test, y_test)

# ----------------------------------------------------------------------------------------------
# 7.  Communicate the results
# ----------------------------------------------------------------------------------------------

# Get feature names from training data
feature_names = model_rf.named_steps["onehotencoder"].get_feature_names()
print("features len:", len(feature_names))
print(feature_names[:5])  # First five feature names

# Extract the model importances parameters
importances = model_rf.named_steps["randomforestregressor"].feature_importances_
print(importances)

# Create a series with feature names and importances
feat_imp = pd.Series(importances, index=feature_names).sort_values()
print(feat_imp)

# Plot 10 most important features
feat_imp.tail(10).plot(kind="barh")
plt.xlabel("Importance [USD]")
plt.ylabel("Feature")
plt.title(
    "Feature Importance for Housing Types: House, Cottage, Villa, Semi-Detached, Terrace"
)

# ----------------------------------------------------------------------------------------------
# Extract the model_rf predictions for test performance visualisation
y_pred_rf = pd.Series(model_rf.predict(X_test))
y_pred_rf

# RandomForest final model visualisation
# Scatter Plot: Actual vs. Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.6)  # Alpha for better visibility
plt.plot(
    [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2
)  # Diagonal Line (X and y coordinates for deal Predictions)
plt.xlabel("Actual Price (USD)")
plt.ylabel("Predicted Price (USD)")
plt.title("Random Forest: Actual vs. Predicted Prices")
plt.show()

# ----------------------------------------------------------------------------------------------

# Residuals Histogram indicating where the residuals are zero (perfect predictions)
residuals = y_test - y_pred_rf
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.axvline(
    0, color="r", linestyle="--", linewidth=2
)  # Adds a red dashed vertical line
plt.xlabel("Prediction Error (Residuals)")
plt.ylabel("Frequency")
plt.title("Residual Training Distribution")
plt.show()

# ----------------------------------------------------------------------------------------------

# Check accuracy score of the actual vs the predicted model
acc_train = model_rf.score(X_train, y_train)
acc_test = model_rf.score(X_test, y_test)
acc_train
acc_test

# ----------------------------------------------------------------------------------------------
"""
## Actual vs. Predicted Prices

This scatter plot compares the actual property prices with the prices predicted by the Random Forest model.

- The **red dashed diagonal line** represents perfect predictions: for any point on this line,
  the predicted price equals the actual price.
- The scatter of points around this line suggests that while some predictions closely match actual values,
  there is a notable degree of dispersion, particularly for higher prices. This indicates that while the model
  captures some trends in the data, it may not be fully accurate across all price ranges.

### Overall Insights
- Both visualizations indicate that the **Random Forest model** performs reasonably well, with residuals centered around zero.
- The scatter plot shows room for improvement, especially in predicting higher property prices,
  suggesting that further tuning or alternative modeling strategies could enhance performance.

### Follow-up Project
A follow-up project will aim to improve the model's accuracy by incorporating additional features
  such as **building area**, **location**, and **suburb**, with the goal of better predicting property prices.
"""

# ----------------------------------------------------------------------------------------------
# 8. Save the model and Reload the model
# ----------------------------------------------------------------------------------------------

# Save the model to 'models' folder using joblib
joblib.dump(model_rf, "../../models/price_by_suburb_model.pkl")

# Load the model from the 'models' folder
model = joblib.load("../../models/price_by_suburb_model.pkl")

# Check the model
model
