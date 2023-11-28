class DataSummary:
    """Class for summarizing data."""

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
        Number of Use Cases: To determine the number of use cases (rows) in the dataset.
        """
        return "Number of Use Cases: ", self.df.shape[0]

    def unique_school_name(self):
        """
        Number of Unique Schools: To calculate the number of unique schools in the dataset.
        """
        return (
            "Number of Unique Schools present in the data: ",
            self.df.school_name.nunique(),
        )

    def num_attributes(self):
        """
        Number of Attributes: To calculate the number of attributes (columns) in the dataset.
        """
        return "Number of Attributes: ", self.df.shape[1]

    def data_types(self):
        """
        Data Types for Each Attribute: Display the data types of each attribute in our dataset.
        """
        return "Data Types for Each Attribute:", self.df.dtypes

    def total_students_desc(self):
        """
        Descriptive Statistics for Total Students: Display descriptive statistics for the 'total_students' column.
        """
        return self.df.total_students.describe()

    def check_null_students(self):
        """
        Check Null Values for Total Students: Check the number of null values in the 'total_students' column.
        """
        return self.df.total_students.isnull().sum()
