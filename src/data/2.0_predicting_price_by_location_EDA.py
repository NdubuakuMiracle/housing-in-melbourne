# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sys

sys.path.append("../../../data/wrangle_module.py")
# Import the DataProcessor class from the wrangle_module.py file
from wrangle_module import DataProcessor

# ----------------------------------------------------------------------------------------------
# 1. Use the DataProcessor class to prepare the data
# ----------------------------------------------------------------------------------------------

# Instantiate the DataProcessor class
processor = DataProcessor()

# Use the wrangle method to process the data
df = processor.wrangle("../../data/raw/Melbourne_housing_FULL.csv")
print("df shape:", df.shape)
df.head()
df.info()

# ----------------------------------------------------------------------------------------------
# 2. Preview the data
# ----------------------------------------------------------------------------------------------

# check the proportion of missing values
df.isnull().mean().sort_values(ascending=False)
# check the percentage of missing values
df.isnull().mean().sort_values(ascending=False) * 100
# Check the proportion of missing values by percentage
df.isnull().sum() / len(df)

# ----------------------------------------------------------------------------------------------

# Explore the Latitude and Longitude columns
df[["Latitude", "Longitude"]].describe()

# Plot the distribution of Latitude and Longitude
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  # (rows, columns, panel number)
sns.histplot(
    df["Latitude"], kde=True, color="skyblue"
)  # kde=True adds a kernel density estimate
plt.title("Distribution of Latitude")
plt.subplot(1, 2, 2)  # (rows, columns, panel number)
sns.histplot(df["Longitude"], kde=True, color="red")
plt.title("Distribution of Longitude")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------------------

# Build mapbox scatter plot
fig = px.scatter_mapbox(
    df,  # The DataFrame
    lat="Latitude",  # Latitude
    lon="Longitude",  # Longitude
    width=600,  # Width of map
    height=600,  # Height of map
    color="Price_USD",
    hover_data=["Price_USD"],  # Display price when hovering mouse over house
)
fig.update_layout(mapbox_style="open-street-map")
fig.show()
"""
Key Insights:
The most expensive properties are concentrated in the central and northeast
inner suburbs, marked by yellow and orange hues, suggesting values often
exceeding $1 million.

This trend suggests a common urban phenomenon where proximity to the city 
center correlates with higher real estate values, often attributed to
accessibility, amenities, and demand.
"""

# Add a third dimension to the scatter plot
# 3D scatter plot
fig = px.scatter_3d(
    df,
    x="Longitude",
    y="Latitude",
    z="Price_USD",
    labels={"lon": "longitude", "lat": "latitude", "price_usd": "price"},
    width=600,
    height=500,
)
# Refine formatting
fig.update_traces(
    marker={"size": 4, "line": {"width": 2, "color": "DarkSlateGrey"}},
    selector={"mode": "markers"},
)
# Display figure
fig.show()

# ----------------------------------------------------------------------------------------------
# 3. Export the location wrangled data
# ----------------------------------------------------------------------------------------------

df.to_csv("../../data/processed/processed_melbourne_data.csv", index=False)
