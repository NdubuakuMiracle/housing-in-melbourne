# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.utils.validation import check_is_fitted

# ----------------------------------------------------------------------------------------------
# 1. Load the transformed data
# ----------------------------------------------------------------------------------------------

# Load the transformed data
df = pd.read_csv("../../data/interim/price_by_size_wrangled.csv")
df.head()
df.shape
df.info()
df["Price_USD"].mean()

# ----------------------------------------------------------------------------------------------
# 2. Split the data into features and target
# ----------------------------------------------------------------------------------------------

# Split the data into features and target
X = df[["BuildingArea"]]
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

# Visualize the baseline model
plt.plot(
    X_train.values,
    y_pred_baseline,
    color="orange",
    label="Baseline Model",
)
plt.scatter(X_train, y_train)
plt.xlabel("Building Area[metres]")
plt.ylabel("Price [USD]")
plt.title("Melbourne: Price vs. Building Area")
plt.legend()

# ----------------------------------------------------------------------------------------------
# Calculate the baseline mean absolute error
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)
mae_baseline
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
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Check your that model is fitted
check_is_fitted(model)

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

# Calculate the MAE for predictions in y_pred against the actual values in y_test
mae_testing2 = mean_absolute_error(y_test, y_pred2)
print("Testing MAE:", round(mae_testing2, 2))

X_test2.shape, y_test.shape, y_pred2.shape

# ----------------------------------------------------------------------------------------------
# 7.  Communicate the results
# ----------------------------------------------------------------------------------------------

# Extract the model parameters
intercept = model.intercept_
print("Model Intercept:", intercept)
coefficient = model.coef_[0]
print('Model coefficient for "Building Area":', coefficient)

print(f"Building Price = {intercept} + {coefficient} * Building Area")

# ----------------------------------------------------------------------------------------------
# Visualize the model
plt.plot(X_train.squeeze(), model.predict(X_train), color="red", label="Linear Model")
plt.scatter(X_train, y_train)
plt.xlabel("Building Area")
plt.ylabel("Price [USD]")
plt.legend()

# ----------------------------------------------------------------------------------------------
"""
## Based on the provided metrics:

- **Mean house price (`price_usd`)**: $752,859.19  
- **Baseline MAE**: $241,499.33  
- **Testing MAE**: $233,855.27 

## Interpretation of Model Performance:

### **Baseline MAE**
The baseline MAE of $241,499.33 represents the average error if the mean house price is always
predicted for all properties.

### **Testing MAE**
The testing MAE of $233,855.27 represents the average error of the linear regression model on
the test set.

---

## Comparison and Conclusion:
The testing MAE ($233,855.27) is lower than the baseline MAE ($241,499.33). This indicates 
that the linear regression model is performing better than the baseline model, which simply
predicts the mean price for all properties.
Although the model is better than the baseline, the MAE is still quite high. This means that
while the model captures some of the relationship between 'BuildingArea' and 'Price_USD', there may
be other important factors influencing house prices that are not included in the model.

---

## Next Steps:

### To improve the model, the following steps will be considered:

**More Features**:
Including additional features that might influence house prices, such as the number of bedrooms,
suburbs, location, etc.
"""

# ----------------------------------------------------------------------------------------------
# 8. Save the model and Reload the model
# ----------------------------------------------------------------------------------------------

# Save the model to 'models' folder using joblib
joblib.dump(model, "../../models/price_by_size_model.pkl")

# Load the model from the 'models' folder
model = joblib.load("../../models/price_by_size_model.pkl")

# Check the model
model
