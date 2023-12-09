"""
This module have a class as DataSummary, designed for summarizing key aspects of a given dataset.

Classes:
    - DataSummary: A class for summarizing data.
"""


class DataSummary:
    """
    Class for summarizing data.
    
    Methods:
    - __init__(self, data): Initializes the DataSummary class.
    - usecase(self): Number of Use Cases: Determine the number of use cases (rows) in the dataset.
    - unique_school_name(self): Number of Unique Schools: Calculate the number of unique schools in the dataset.
    - num_attributes(self): Number of Attributes: Calculate the number of attributes (columns) in the dataset.
    - data_types(self): Data Types for Each Attribute: Display the data types of each attribute in the dataset.
    - total_students_desc(self): Descriptive Statistics for Total Students: Display descriptive statistics for 'total_students'.
    - check_null_students(self): Check Null Values for Total Students: Check the number of null values in 'total_students'.
    """

    def __init__(self, data):
        """
        Initializes the DataSummary class.

        Parameters:
        - data: DataFrame
          The input dataset for data summary.
        """
        self.df = data

    def usecase(self):
        """
        Number of Unique Schools: Calculate the number of unique schools in the dataset.

        Returns:
        Tuple: The number of unique schools.
        """
        return "Number of Use Cases: ", self.df.shape[0]

    def unique_school_name(self):
        """
        Number of Attributes: Calculate the number of attributes (columns) in the dataset.

        Returns:
        Tuple: The number of attributes.
        """
        return (
            "Number of Unique Schools present in the data: ",
            self.df.school_name.nunique(),
        )

    def num_attributes(self):
        """
        Number of Attributes: Calculate the number of attributes (columns) in the dataset.

        Returns:
        Tuple: The number of attributes.
        """
        return "Number of Attributes: ", self.df.shape[1]

    def data_types(self):
        """
        Data Types for Each Attribute: Display the data types of each attribute in the dataset.

        Returns:
        Series: Data types for each attribute.
        """
        return "Data Types for Each Attribute:", self.df.dtypes

    def total_students_desc(self):
        """
        Descriptive Statistics for Total Students: Display descriptive statistics for the 'total_students' column.

        Returns:
        DataFrame: Descriptive statistics for 'total_students'.
        """
        return self.df.total_students.describe()

    def check_null_students(self):
        """
        Check Null Values for Total Students: Check the number of null values in the 'total_students' column.

        Returns:
        int: Number of null values in 'total_students'.
        """
        return self.df.total_students.isnull().sum()
