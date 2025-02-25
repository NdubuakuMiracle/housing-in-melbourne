# Import necessary libraries
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------
# 1. Update the wrangle function for recleaning the data
# ----------------------------------------------------------------------------------------------


def updated_wrangle(
    filepath,
):  # Rerun the 'wrangle' function to get the desired transformed data
    """
    Load and clean the housing data from a CSV file.

    Parameters:
    filepath (str): The path to the CSV file containing the housing data.

    Returns:
    pd.DataFrame: A cleaned DataFrame with the following transformations:
        - Columns 'Lattitude' and 'Longtitude' are renamed to 'Latitude' and 'Longitude', respectively.
        - A new column 'price_USD' is created by converting 'Price' to USD (assuming a conversionrate of 0.75).
        - Unwanted columns are removed.
        - Outliers are removed, keeping only the properties within the 25th and 75th percentiles.
        - All Properties 'Type' are now included except 'br'.
    """
    # Read CSV file into DataFrame and reset index
    df = pd.read_csv(filepath)

    # Rename Lattitude to Latitude and Longtitude to Longitude
    df.rename(
        columns={"Lattitude": "Latitude", "Longtitude": "Longitude"}, inplace=True
    )

    # Create "price_usd" column from AUD 'Price' column to USD in 2018
    df["Price_USD"] = round(df["Price"] * 0.75, 2)

    # Define numerical columns to apply outlier filtering selectively
    num_features = df.select_dtypes("number").drop(
        columns=["Postcode", "Price", "Latitude", "Longitude"]
    )

    # Apply IQR outlier removal
    for col in num_features:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)  # 25th percentile
            Q3 = df[col].quantile(0.75)  # 75th percentile
            IQR = Q3 - Q1
            low_bound = Q1 - 1.5 * IQR
            high_bound = Q3 + 1.5 * IQR
            # Apply filtering
            df = df[df[col].between(low_bound, high_bound)]

    # Subset to all property type except where 'Type' is 'br' bedroom(s)
    mask_type = df["Type"] != "br"
    df = df[mask_type]

    # Drop low- and high-cardinality categorical variables
    df.drop(
        columns=["Address", "SellerG", "Postcode"],
        inplace=True,
    )
    # Drop columns with multicollinearlity
    df.drop(columns=["Rooms", "Bedroom2", "Bathroom", "Landsize"], inplace=True)
    # Drop leaky and unwanted columns
    df.drop(
        columns=[
            "Price",
            "CouncilArea",
            "Regionname",
            "Distance",
            "Car",
            "Date",
            "Type",
            "Method",
            "YearBuilt",
            "Propertycount",
        ],
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)

    return df


# ----------------------------------------------------------------------------------------------
# 2. Preview the data and adjust the 'update_wrangle' funtion accordingling
# ----------------------------------------------------------------------------------------------

df = updated_wrangle("../../data/raw/Melbourne_housing_FULL.csv")
df.head()
df.shape
df.info()
# Calculate missing value proportions and data characteristics
missing_value_report = pd.DataFrame(
    {
        "Count": df.shape[0],
        "Missing Values": df.isnull().sum(),
        "Missing %": (df.isnull().sum() / len(df)) * 100,
        "Cardinality": df.nunique(),
    }
)
missing_value_report


# Describe 'Numiric' Data
df.select_dtypes("number").describe()
# Describe 'Object' Data
df.select_dtypes("object").describe()

# ----------------------------------------------------------------------------------------------

# Visualise the selected  numerical columns to identify outliers
numerical_cols = df.select_dtypes("number").drop(
    columns=["Postcode", "Price", "Latitude", "Longitude"]
)
plt.figure(figsize=(15, 12))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 4, i)
    sns.boxplot(x=df[col], color="skyblue")
    plt.title(f"{col} Box Plot")
plt.tight_layout()
plt.show()


# Remove 'Numerical' outliers and update the above general function
for col in numerical_cols:
    Q1 = numerical_cols[col].quantile(0.25)
    Q3 = numerical_cols[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    mask_num = numerical_cols[
        (numerical_cols[col] > lower_bound) & (numerical_cols[col] < upper_bound)
    ]
mask_num
mask_num.info()

# ----------------------------------------------------------------------------------------------

# Visualise the numerical columns to recheck the removed outliers
num_features = df.select_dtypes("number").drop(
    columns=["Postcode", "Price", "Latitude", "Longitude"]
)
plt.figure(figsize=(15, 12))
for i, col in enumerate(num_features, 1):
    plt.subplot(3, 4, i)
    sns.boxplot(x=df[col], color="skyblue")  # Visualise the histogram
    plt.title(f"{col} Box Plot")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------------------

df["Rooms"].value_counts()
# Visualise the relationship between 'Rooms' and 'Price_USD'
Rooms = df["Rooms"].value_counts().sort_index()
fig = px.pie(
    names=[
        "1 Room",
        "2 Rooms",
        "3 Rooms",
        "4 Rooms",
        "5 Rooms",
        "6 Rooms",
        "7 Rooms",
    ],
    values=Rooms,
    title="Number of Rooms",
    template="plotly_dark",
).update_traces(textinfo="label+percent")
fig.show()

# Visualise the average price
avg_room_price = df.groupby("Rooms")["Price_USD"].mean()
fig = px.line(
    avg_room_price,
    line_shape="spline",
    markers="*",
    template="plotly_dark",
    labels={"value": "Average Price"},
    color_discrete_sequence=["blue"],
    title="Average Price By Number of Rooms",
)
fig.show()

# ----------------------------------------------------------------------------------------------

# Visualise the relationship between 'Distance' and 'Price_USD'
avg_price_distance = df.groupby("Distance")["Price_USD"].mean()
fig = px.line(
    avg_price_distance,
    line_shape="spline",
    markers="*",
    template="plotly_dark",
    labels={"value": "Average Price"},
    color_discrete_sequence=["red"],
    title="Average Price By Distance",
)
fig.show()

# ----------------------------------------------------------------------------------------------

# Visualise the relationship between 'BuildingArea' and 'Price_USD'
avg_price_building_area = df.groupby("BuildingArea")["Price_USD"].mean()
fig = px.line(
    avg_price_building_area,
    line_shape="spline",
    markers="*",
    template="plotly_dark",
    labels={"value": "Average Price"},
    color_discrete_sequence=["blue"],
    title="Average Price By Building Area",
)
fig.show()

# ----------------------------------------------------------------------------------------------

# Visualise the relationship between 'Lat and Lon' and 'Price_USD'
fig = px.scatter_mapbox(
    df,  # The DataFrame
    lat="Latitude",  # Latitude
    lon="Longitude",  # Longitude
    width=800,  # Width of map
    height=700,  # Height of map
    color="Price_USD",
    template="plotly_dark",
    hover_data=["Price_USD"],  # Display price when hovering mouse over house
)
fig.update_layout(mapbox_style="open-street-map")
fig.show()

# ----------------------------------------------------------------------------------------------

# Visualise the distribution of 'Method'
df["Method"].unique()
Method = df["Method"].value_counts()
fig = px.pie(
    values=Method,
    names=[
        "Property Sold",
        "Property Sold Prior",
        "Vendor Bid",
        "Property Passed In",
        "Sold After Auction",
    ],
    template="plotly_dark",
    title="Sale Methods",
).update_traces(textinfo="label+percent")
fig.show()

# ----------------------------------------------------------------------------------------------

# Visualise the different property types
df["Type"].unique()
property_type = df["Type"].value_counts()
fig = px.pie(
    values=property_type,
    names=["House, Cottage, Villa, Semi, Terrace", "Unit, Duplex", "Townhouse"],
    template="plotly_dark",
    title="Property Types",
).update_traces(textinfo="label+percent")
fig.show()


# ----------------------------------------------------------------------------------------------

# Check columns for multicollinearity and update the 'updated_wrangle' function
corr = df.select_dtypes("number").drop(columns=["Price_USD"]).corr()
corr
# Visualise it for better judgement
sns.heatmap(corr)

# ----------------------------------------------------------------------------------------------
# 3. Export the location wrangled data
# ----------------------------------------------------------------------------------------------

df.to_csv("../../data/processed/final_processed_data.csv", index=False)
