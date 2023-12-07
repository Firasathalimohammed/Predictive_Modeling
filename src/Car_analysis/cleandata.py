import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class CarDataCleaner:
    def __init__(self, dataframe):
        """
        Initialize the CarDataCleaner class with a pandas DataFrame.
        """
        self.data = dataframe

    def check_missing_values(self):
        """
        Check for missing values in the dataset.
        """
        return self.data.isnull().sum()

    def fill_missing_values(self, strategy="mode", columns=None):
        """
        Fill missing values in the dataset.
        - strategy: 'mean', 'median', or 'mode'
        - columns: list of columns to apply filling, if None, apply to all columns
        """
        if columns is None:
            columns = self.data.columns

        for col in columns:
            if strategy == "mean":
                fill_value = self.data[col].mean()
            elif strategy == "median":
                fill_value = self.data[col].median()
            elif strategy == "mode":
                fill_value = self.data[col].mode()[0]
            else:
                raise ValueError("Invalid strategy. Use 'mean', 'median', or 'mode'.")
            self.data[col].fillna(fill_value, inplace=True)
