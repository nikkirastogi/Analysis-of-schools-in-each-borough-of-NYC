# Importing required modules
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

class DataPreprocessing:
    def __init__(self,data):
        self.df = data
        return 
        
    def set_option(self):
        return pd.set_option('display.max_columns', None)  # display all columns in a dataset
    
    def set_printoptions(self):
        return np.set_printoptions(suppress = True)        # suppress scientific notation
    
    def print_five_rows(self):
        return self.df.head()
    
    def dataframe_shape(self):
        return "The shape of the dataset (rows, cols): ", self.df.shape
        
    def num_columns(self):
        return "Columns present in the raw dataset", self.df.columns
    
    def select_columns(self):
        self.df = self.df[['dbn', 'school_name', 'borough', 'building_code', 'grade_span_min', 'grade_span_max', 'city', 'state_code', 'total_students', 'school_type', 'extracurricular_activities', 'psal_sports_boys', 'psal_sports_girls', 'psal_sports_coed', 'school_sports', 'partner_cbo', 'partner_hospital', 'partner_highered', 'partner_cultural', 'partner_nonprofit', 'partner_corporate', 'partner_financial', 'partner_other', 'school_accessibility_description', 'number_programs']]
        return self.df
    
    # Check the number of null values in the columns
    def check_null(self):
        """Missing values in Dataset"""    
        return self.df.isnull().sum()
    
    