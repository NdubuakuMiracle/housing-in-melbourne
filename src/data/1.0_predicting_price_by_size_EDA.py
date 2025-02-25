# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ----------------------------------------------------------------------------------------------
# 1. Create a wrangle function to prepare the data
# ----------------------------------------------------------------------------------------------


def wrangle(filepath):  # Rerun the 'wrangle' function to get the transformed data
    """
    Load and clean the housing data from a CSV file.

    Parameters:
    filepath (str): The path to the CSV file containing the housing data.

    Returns:
    pd.DataFrame: A cleaned DataFrame with the following transformations:
        - Columns 'Lattitude' and 'Longtitude' are renamed to 'Latitude' and 'Longitude', respectively.
        - A new column 'price_USD' is created by converting 'Price' to USD (assuming a conversionrate of 0.75).
        - Properties with 'price_usd' >= 1,500,000 are removed.
        - Outliers in 'BuildingArea' are removed, keeping only the properties within the 10th and 90th percentiles.
        - Properties where 'Type' is not 'h' (house, cottage, villa, semi, terrace) are removed.
    """
    # Read CSV file into DataFrame and reset index
    df = pd.read_csv(filepath)

    # Rename Lattitude to Latitude
    df.rename(columns={"Lattitude": "Latitude"}, inplace=True)
    # Rename Longtitude to Longitude
    df.rename(columns={"Longtitude": "Longitude"}, inplace=True)

    # Create "price_usd" column from AUD 'Price' column to USD in 2018
    df["Price_USD"] = round(df["Price"] * 0.75, 2)
    # Subset to properties where 'price_usd' < 1,500,000
    mask_price = df["Price_USD"] < 1_500_000
    df = df[mask_price]

    # Remove outliers by 'BuildingArea'
    low, high = df["BuildingArea"].quantile([0.1, 0.9])
    mask_area = df["BuildingArea"].between(low, high)
    df = df[mask_area]

    # Subset to properties where 'Type' is 'h' (house, cottage, villa, semi, terrace)
    mask_type = df["Type"] == "h"
    df = df[mask_type]

    return df


# ----------------------------------------------------------------------------------------------
# 2. Preview data for EDA
# ----------------------------------------------------------------------------------------------

df = wrangle("../../data/raw/Melbourne_housing_FULL.csv")
print("df shape:", df.shape)
df.head()
df.info()
# check the proportion of missing values
df.isnull().mean().sort_values(ascending=False)
# check the percentage of missing values
df.isnull().mean().sort_values(ascending=False) * 100
# Check the proportion of missing values by percentage
df.isnull().sum() / len(df)

# ----------------------------------------------------------------------------------------------

# Rename Lattitude to Latitude
df.rename(columns={"Lattitude": "Latitude"}, inplace=True)
# Rename Longtitude to Longitude
df.rename(columns={"Longtitude": "Longitude"}, inplace=True)

# Explore the price and size columns and update 'wrangle' function for subsequent EDA
# Check the distribution of the 'Price' column
df["Price"].describe()
plt.hist(df["Price"])
sns.boxplot(df["Price"])

# Check the distribution of the 'BuildingArea' column
df["Price_USD"].describe()
plt.hist(df["Price_USD"])
sns.boxplot(df["Price_USD"])

# Check the distribution of the 'BuildingArea' column and update 'wrangle' function
df["BuildingArea"].describe()  # Recheck the 'BuildingArea' column
plt.hist(df["BuildingArea"])
sns.boxplot(df["BuildingArea"])

# Plot the distribution of the 'Bedroom2' column
df["Bedroom2"].describe()
plt.hist(df["Bedroom2"])
sns.boxplot(df["Bedroom2"])
df["Bedroom2"].value_counts().plot(kind="bar")

# Plot the distribution of the 'Rooms' column
df["Rooms"].describe()
plt.hist(df["Rooms"])
sns.boxplot(df["Rooms"])
df["Rooms"].value_counts().plot(kind="bar")

# Plot the distribution of the 'Type' column
df["Type"].value_counts().plot(kind="bar")
# Update the 'Type' column in 'wrangle' function for 'h'

# Check the distribution of the 'Suburb' column
df["Suburb"].value_counts(normalize=True).head(10)
df["Suburb"].nunique()


# ----------------------------------------------------------------------------------------------

# Build scatter plot of 'BuildingArea' vs. 'Price_USD'
plt.scatter(
    data=df,
    x="BuildingArea",
    y="Price_USD",
    alpha=0.2,
    color="blue",
    marker="o",
    s=50,
    edgecolors="black",
)
# Label axes
plt.xlabel("Building Area[metres]")
plt.ylabel("Price [USD]")
# Add title
plt.title("Melbourne: Price vs. Building Area")
plt.show()

# Correlation matrixf of the 'BuildingArea' and 'Price_usd' columns
df["BuildingArea"].corr(df["Price_USD"])

# ----------------------------------------------------------------------------------------------
# 3. Export the wrangled data
# ----------------------------------------------------------------------------------------------

wrangle("../../data/raw/Melbourne_housing_FULL.csv").to_csv(
    "../../data/interim/price_by_size_wrangled.csv", index=False
)
