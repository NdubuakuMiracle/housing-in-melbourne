# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
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
df["Price_USD"].mean()

# ----------------------------------------------------------------------------------------------
# 2. Split the data into features and target
# ----------------------------------------------------------------------------------------------

# Split the data into features and target
X = df[["Longitude", "Latitude"]]
y = df["Price_USD"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# # Reshape the data
# X_train = X_train.values.reshape(-1, 1)
# X_test = X_test.values.reshape(-1, 1)

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
# 4. Iterate on the model
# ----------------------------------------------------------------------------------------------

# Instantiate the model
model = make_pipeline(SimpleImputer(), LinearRegression())

# Fit the model
model.fit(X_train, y_train)

# Check your that model is fitted
check_is_fitted(model)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
X_train.info()
y_train.info()
df.info()
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

# Make predictions on another test set
X_test2 = pd.read_csv("../../data/interim/X_test.csv")
y_pred2 = pd.Series(model.predict(X_test2))
y_pred2[:5]

# ----------------------------------------------------------------------------------------------
# 7.  Communicate the results
# ----------------------------------------------------------------------------------------------

# Extract the model parameters
intercept = model.named_steps["linearregression"].intercept_.round()
coefficient = model.named_steps["linearregression"].coef_.round()
print("Model Intercept:", intercept)
print('Model coefficient for "Lat-Lon":', coefficient)

print(f"Location Price = {intercept} + {coefficient} * Lat-Lon")

# ----------------------------------------------------------------------------------------------

# Visualize the model on a 3D scatter plot
fig = px.scatter_3d(
    df,
    x="Longitude",
    y="Latitude",
    z="Price_USD",
    labels={
        "Longitude": "longitude",
        "Latitude": "latitude",
        "Price_USD": "price",
    },
    width=600,
    height=500,
)
# Create x and y coordinates for model representation
x_plane = np.linspace(df["Longitude"].min(), df["Longitude"].max(), 10)
y_plane = np.linspace(df["Latitude"].min(), df["Latitude"].max(), 10)
xx, yy = np.meshgrid(x_plane, y_plane)
# Use model to predict z coordinates
z_plane = model.predict(pd.DataFrame({"Longitude": x_plane, "Latitude": y_plane}))
zz = np.tile(z_plane, (10, 1))

# Add plane to figure
fig.add_trace(go.Surface(x=xx, y=yy, z=zz))

# Refine formatting
fig.update_traces(
    marker={"size": 4, "line": {"width": 2, "color": "DarkSlateGrey"}},
    selector={"mode": "markers"},
)

# Display figure
fig.show()

# ----------------------------------------------------------------------------------------------

# Density contour plot to visualize price variations based on latitude and longitude
plt.figure(figsize=(10, 6))
sns.kdeplot(
    x=df["Longitude"],
    y=df["Latitude"],
    weights=df["Price_USD"],
    cmap="coolwarm",
    fill=True,
    levels=50,
)
plt.scatter(
    df["Longitude"], df["Latitude"], c=df["Price_USD"], cmap="viridis", alpha=0.5, s=10
)
plt.colorbar(label="Price Intensity (USD)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Density Contour of Property Prices in Melbourne")

# Show the plot
plt.show()

# ----------------------------------------------------------------------------------------------

# Check correlation between Price and Latitude/Longitude
correlation = df[["Price_USD", "Latitude", "Longitude"]].corr()
correlation

# ----------------------------------------------------------------------------------------------
"""
## Based on the provided metrics:  

- **Mean house price (price_usd)**: $752,859.19  
- **Baseline MAE**: $241,499.33  
- **Testing MAE**: $217,255.91  

## Interpretation of Model Performance:  

### **Baseline MAE**:  
The baseline MAE of $241,499.33 represents the average error if the mean house price is always  
predicted for all properties.  

### **Testing MAE**:  
The testing MAE of $217,255.91 represents the average error of the linear regression model on  
the test set.  

---

### Analysis of Property Prices Based on Location:  

**Key Observations (Subsetted for Prices < $1,500,000)**:  
- **Highest Prices**: The most expensive properties in this dataset are in the **Central and Eastern regions**.  
- **Lowest Prices**: The **Western and Northern regions** have the lowest property values.  

### Geospatial Insights:  
- The **Central East** region has the highest-priced properties in this subset, likely due to its proximity to  
  premium residential areas and economic hubs.  
- The **West and North remain the most affordable**, aligning with historical price trends.  

### General Market Classification (Based on This Subset):  
- **Most Expensive Regions**: Central and Eastern suburbs.  
- **Most Affordable Regions**: Western and Northern suburbs.  

### Conclusion:  
- **For buyers looking under $1.5M**, the **Central and Eastern regions** still command higher prices.  
- The **West and North offer more budget-friendly options**, making them ideal for affordability-focused buyers.  

This analysis is specific to properties **below $1.5M**. Including all prices might shift these trends.  
"""

# ----------------------------------------------------------------------------------------------
# 8. Save the model and Reload the model
# ----------------------------------------------------------------------------------------------

# Save the model to 'models' folder using joblib
joblib.dump(model, "../../models/price_by_location_model.pkl")

# Load the model from the 'models' folder
model = joblib.load("../../models/price_by_location_model.pkl")

# Check the model
model
