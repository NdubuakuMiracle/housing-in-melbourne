# Import necessary libraries
import pandas as pd

# ----------------------------------------------------------------------------------------------
# 1. Modularization: Save the wrangle function in a separate file for future importation
# ----------------------------------------------------------------------------------------------

"""
Save this function in a separate file (e.g., wrangle_module.py) to make it reusable.

This process is called modularization. It helps in organizing code into reusable modules.
"""

# ----------------------------------------------------------------------------------------------
# 2. Create a class with the wrangle function as a method
# ----------------------------------------------------------------------------------------------


# Create a class to hold the wrangle method
class DataProcessor:
    """
    A class used to load and clean housing data from a CSV file.

    Methods
    -------
    wrangle(filepath)
        Loads and cleans the housing data from the specified CSV file.
    """

    def __init__(self):
        pass

    def wrangle(self, filepath):
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

        # Subset to properties where 'Type' is 'h' (House, Cottage, Villa, Semi-Detached, Terrace)
        mask_type = df["Type"] == "h"
        df = df[mask_type]

        return df
