# Importing required modules
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt


class DataPreprocessing:
    """Class for performing data preprocessing tasks."""

    def __init__(self, data):
        """
        Initializes the DataPreprocessing class.

        Parameters:
        - data: DataFrame (input dataset for data preprocessing)
        """
        self.df = data

    def set_option(self):
        """
        Set display options to show all columns in a dataset.
        """
        return pd.set_option("display.max_columns", None)

    def set_printoptions(self):
        """
        Set print options to suppress scientific notation.
        """
        return np.set_printoptions(suppress=True)

    def print_five_rows(self):
        """
        Display the first five rows of the dataset.
        """
        return self.df.head()

    def dataframe_shape(self):
        """
        Get the shape of the dataset.

        Returns:
        Tuple: The shape of the dataset (rows, cols).
        """
        return "The shape of the dataset (rows, cols): ", self.df.shape

    def num_columns(self):
        """
        Get the columns present in the raw dataset.

        Returns:
        Tuple: Columns present in the raw dataset.
        """
        return "Columns present in the raw dataset", self.df.columns

    def select_columns(self):
        """
        Select a subset of columns from the dataset.
        """
        self.df = self.df[
            [
                "dbn",
                "school_name",
                "borough",
                "building_code",
                "grade_span_min",
                "grade_span_max",
                "city",
                "state_code",
                "total_students",
                "school_type",
                "extracurricular_activities",
                "psal_sports_boys",
                "psal_sports_girls",
                "psal_sports_coed",
                "school_sports",
                "partner_cbo",
                "partner_hospital",
                "partner_highered",
                "partner_cultural",
                "partner_nonprofit",
                "partner_corporate",
                "partner_financial",
                "partner_other",
                "school_accessibility_description",
                "number_programs",
            ]
        ]
        return self.df

    def check_null(self):
        """
        Check the number of null values in the columns.

        Returns:
        Series: Number of null values in each column.
        """
        return self.df.isnull().sum()
