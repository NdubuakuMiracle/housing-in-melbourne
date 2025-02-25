# Import necessary libraries
import pandas as pd

# ----------------------------------------------------------------------------------------------
# 1. Modularization: Save the wrangle function in a separate file for future importation
# ----------------------------------------------------------------------------------------------

"""
Save this function in a separate file (e.g., updated_wrangle_module.py) to make it reusable.

This process is called modularization. It helps in organizing code into reusable modules.
"""

# ----------------------------------------------------------------------------------------------
# 2. Create a class with the wrangle function as a method
# ----------------------------------------------------------------------------------------------


# Create a class to hold the wrangle method
class DataTransformer:
    """
    A class used to load and clean housing data from a CSV file.

    Methods
    -------
    wrangle(filepath)
        Loads and cleans the housing data from the specified CSV file.
    """

    def __init__(self):
        pass

    def updated_wrangle(
        self, filepath
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
