
class DataSummary:
    def __init__(self, data):
        """Create a object by passing in a `pandas.DataFrame` of data."""
        self.df = data
    
    # Calculate the number of use cases
    def usecase(self):
        """Number of Use Cases: To determine the number of use cases (rows) in the dataset"""
        return "Number of Use Cases: " , self.df.shape[0]
    
    def unique_school_name(self):
        return "Number of Unique Schools present in the data: ", self.df.school_name.nunique()

    def num_attributes(self):
        """Number of Attributes: To calculate the number of attributes (columns) in the dataset"""
        return "Number of Attributes: " , self.df.shape[1]
    
    
    def data_types(self):
        """Data Types for Each Attribute: To display the data types of each attribute in our dataset."""
        return "Data Types for Each Attribute:", self.df.dtypes
    
    def total_students.desc(self):
        return self.df.total_students.describe()
    
    def check_null_students():
        return self.df.total_students.isnull().sum()