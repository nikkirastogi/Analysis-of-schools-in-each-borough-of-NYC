import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np

class ExploratoryAnalysis:

    def __init__(self, data):
        """
        Initialize the 'ExploratoryAnalysis' class with a DataFrame.

        Parameters:
        - data: pandas DataFrame
          The input data containing relevant information.
        """
        self.df = data

    def all_total_students(self):
        """
        Calculate and return the total number of students enrolled across all schools.

        Returns:
        - str: Formatted string indicating the total number of students.
        """
        return "Total number of students enrolled across all schools: {:,.0f}".format(self.df.total_students.sum())

    def matplot_hist_total_students(self):
        """
        Generate and display a histogram of the total number of students using Matplotlib.
        """
        plt.hist(self.df["total_students"], bins=20)
        plt.xlabel("Total Students")
        plt.ylabel("Frequency")
        plt.title("Histogram of Total Students (Using Matplotlib)")
        return plt.show()

    def seaborn_hist_total_students(self):
        """Generate and display a histogram of the total number of students using Seaborn."""
        sb.histplot(data=self.df, x="total_students", bins=20)
        plt.xlabel("Total Students")
        plt.ylabel("Frequency")
        plt.title("Histogram of Total Students (Using Seaborn)")
        return plt.show()

    def population_borough(self):
        """Calculate and return the percentage population of each borough."""
        borough = self.df["borough"].value_counts(normalize=True) * 100
        borough = pd.DataFrame({'borough': borough.index, 'values': np.round(borough.values, 2)})

    def matplot_bar_borough(self):
        """Generate and display a barplot indicating the population of each borough using Matplotlib."""
        plt.bar(borough["borough"], borough["values"])
        plt.xlabel("Borough of School")
        plt.ylabel("Percentage")
        plt.title("Barplot indicating Population of Each Borough (Using Matplotlib)")
        return plt.show()

    def seaborn_bar_borough(self):
        """Generate and display a barplot indicating the population of each borough using Seaborn."""
        ax = sb.barplot(data=borough, x="borough", y="values", palette="viridis")
        plt.xlabel("Borough of School")
        plt.ylabel("Percentage")
        plt.title("Barplot indicating Population of Each Borough (Using Seaborn)")
        for i in ax.containers:
            ax.bar_label(i, )
        return plt.show()

    def matplot_bar_grpby_borough_tot_students(self):
        """Generate and display a barplot showing the distribution of students enrolled in each borough using Matplotlib."""
        data = self.df.groupby("borough").sum("total_students").iloc[:, 4]
        plt.bar(data.index, data.values)
        plt.xlabel("Borough of School")
        plt.ylabel("Number of Students")
        plt.title("Distribution of students enrolled in each Borough (using matplotlib)")
        return plt.show()

    def seaborn_bar_borough_tot_students(self):
        """
        Generate and display a barplot showing the distribution of students enrolled in each borough using Seaborn.
        """
        ax = sb.barplot(self.df, x="borough", y="total_students", errorbar=None, estimator="sum")
        plt.xlabel("Borough of School")
        plt.ylabel("Number of Students")
        plt.title("Distribution of students enrolled in each Borough (using seaborn)")
        for i in ax.containers:
            ax.bar_label(i, )
        return plt.show()

    def unique_activities(self):
        """
        Calculate and return the number of unique Extracurricular Activities provided by all schools.

        Returns:
        - str: Formatted string indicating the number of unique extracurricular activities.
        """
        return "Number of unique Extracurricular Activities provided by all schools: {:,}".format(
            self.df['extracurricular_activities'].apply(lambda x: str(x).split(', ')).explode().nunique())

    def activities_desc(self):
        """
        Describe the field 'num_ext_act' representing the number of extracurricular activities.
        """
        self.df["num_ext_act"] = self.df['extracurricular_activities'].apply(
            lambda x: max(len(str(x).split(', ')), 0))
        return self.df["num_ext_act"].describe()

    def matplot_scatter_tot_studs_vs_activities(self):
        """
        Generate and display a scatter plot of total_students vs the number of extracurricular activities offered using Matplotlib.
        """
        plt.scatter(x=self.df["total_students"], y=self.df["num_ext_act"])
        plt.xlabel("Number of students")
        plt.ylabel("Number of Extracurricular activities offered")
        plt.title("Distribution of total_students vs Number of Extracurricular activities offered (using matplotlib)")
        return plt.show()

    def seaborn_scatter_tot_studs_vs_activities(self):
        """
        Generate and display a scatter plot of total_students vs the number of extracurricular activities offered using Seaborn.
        """
        sb.scatterplot(data=self.df, x="total_students", y="num_ext_act", color="olive")
        plt.xlabel("Number of students")
        plt.ylabel("Number of Extracurricular activities offered")
        plt.title("Distribution of total_students vs Number of Extracurricular activities offered (using seaborn)")
        return plt.show()

    def num_sports(self):
        """
        Calculate and store the number of sports-related metrics for analysis.
        """
        self.df["num_sports_boys"] = self.df['psal_sports_boys'].apply(lambda x: max(len(str(x).split(', ')), 0))
        self.df["num_sports_girls"] = self.df['psal_sports_girls'].apply(lambda x: max(len(str(x).split(', ')), 0))
        self.df["num_sports_coed"] = self.df['psal_sports_coed'].apply(lambda x: max(len(str(x).split(', ')), 0))
        self.df["num_sports_othr"] = self.df['school_sports'].apply(lambda x: max(len(str(x).split(', ')), 0))
        self.df["num_sports_tot"] = self.df[['num_sports_boys', 'num_sports_girls', 'num_sports_coed',
                                              'num_sports_othr']].sum(axis=1)

    def matplot_scatter_students_sports(self):
        """
        Generate and display scatter plots for total_students vs each sports category using Matplotlib.

        """
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

        ax[0, 0].scatter(x=self.df["total_students"], y=self.df["num_sports_tot"])
        ax[0, 0].set_xlabel("Number of Students")
        ax[0, 0].set_ylabel("Total number of sports offered")
        ax[0, 1].scatter(x=self.df["total_students"], y=self.df["num_sports_boys"])
        ax[0, 1].set_xlabel("Number of Students")
        ax[0, 1].set_ylabel("Number of sports offered for Boys")
        ax[1, 0].scatter(x=self.df["total_students"], y=self.df["num_sports_girls"])
        ax[1, 0].set_xlabel("Number of Students")
        ax[1, 0].set_ylabel("Number of sports offered for Girls")
        ax[1, 1].scatter(x=self.df["total_students"], y=self.df["num_sports_coed"])
        ax[1, 1].set_xlabel("Number of Students")
        ax[1, 1].set_ylabel("Number of sports offered for both Boys and Girls")

        plt.suptitle("Distributions of Number of students vs Sports Offered (using matplotlib)")
        return plt.show()

    def scatter_scatter_students_sports(self):
        """
        Generate and display scatter plots for total_students vs each sports category using Seaborn.
        """
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

        sb.scatterplot(data=self.df, x="total_students", y="num_sports_tot", ax=ax[0, 0], color="green")
        ax[0, 0].set_xlabel("Number of Students")
        ax[0, 0].set_ylabel("Total number of sports offered")
        sb.scatterplot(data=self.df, x="total_students", y="num_sports_boys", ax=ax[0, 1])
        ax[0, 1].set_xlabel("Number of Students")
        ax[0, 1].set_ylabel("Number of sports offered for Boys")
        sb.scatterplot(data=self.df, x="total_students", y="num_sports_girls", ax=ax[1, 0], color="plum")
        ax[1, 0].set_xlabel("Number of Students")
        ax[1, 0].set_ylabel("Number of sports offered for Girls")
        sb.scatterplot(data=self.df, x="total_students", y="num_sports_coed", ax=ax[1, 1], color="orange")
        ax[1, 0].set_xlabel("Number of Students")
        ax[1, 0].set_ylabel("Number of sports offered for both Boys and Girls")

        plt.suptitle("Distributions of Number of students vs Sports Offered (using seaborn)")
        return plt.show()

    def num_prtn(self):
        """
        Calculate and store the number of partner-related metrics for analysis.

        """
        self.df["num_prtn_cbo"] = self.df['partner_cbo'].apply(lambda x: max(len(str(x).split(', ')), 0))
        self.df["num_prtn_hpt"] = self.df['partner_hospital'].apply(lambda x: max(len(str(x).split(', ')), 0))
        self.df["num_prtn_high"] = self.df['partner_highered'].apply(lambda x: max(len(str(x).split(', ')), 0))
        self.df["num_prtn_clt"] = self.df['partner_cultural'].apply(lambda x: max(len(str(x).split(', ')), 0))
        self.df["num_prtn_nonprft"] = self.df['partner_nonprofit'].apply(lambda x: max(len(str(x).split(', ')), 0))
        self.df["num_prtn_corp"] = self.df['partner_corporate'].apply(lambda x: max(len(str(x).split(', ')), 0))
        self.df["num_prtn_fin"] = self.df['partner_financial'].apply(lambda x: max(len(str(x).split(', ')), 0))
        self.df["num_prtn_othr"] = self.df['partner_other'].apply(lambda x: max(len(str(x).split(', ')), 0))

        self.df["num_prtn_tot"] = self.df[
            ['num_prtn_cbo', 'num_prtn_hpt', 'num_prtn_high', 'num_prtn_clt', 'num_prtn_nonprft', 'num_prtn_corp',
             'num_prtn_fin', 'num_prtn_othr']].sum(axis=1)

    def matplot_scatter_students_vs_partners(self):
        """
        Generate and display a scatter plot of total_students vs the number of partner opportunities using Matplotlib.
        """
        plt.scatter(x=self.df["total_students"], y=self.df["num_prtn_tot"])
        plt.xlabel("Number of students")
        plt.ylabel("Number of Partner opportunities")
        plt.title("Distribution of total_students vs Number of Partner Opportunities available (using matplotlib)")
        return plt.show()

    def seaborn_scatter_students_vs_partners(self):
        """
        Generate and display a scatter plot of total_students vs the number of partner opportunities using Seaborn.
        """
        sb.scatterplot(data=self.df, x="total_students", y="num_prtn_tot", color="coral")
        plt.xlabel("Number of students")
        plt.ylabel("Number of Partner opportunities")
        plt.title("Distribution of total_students vs Number of Partner Opportunities available (using seaborn)")
        return plt.show()

    def matplot_bar_accessibility_vs_students(self):
        """Generate and display a barplot showing the distribution of the number of students vs School Accessibility Description using Matplotlib."""
        data = self.df.groupby("school_accessibility_description").sum("total_students").iloc[:, 4]
        plt.bar(data.index, data.values)
        plt.xlabel("School Accessibility Description")
        plt.ylabel("Number of Students")
        plt.title("Distribution of number of students vs School Accessibility Description (using matplotlib)")
        return plt.show()

    def seaborn_bar_accessibility_vs_students(self):
        """Generate and display a barplot showing the distribution of the number of students vs School Accessibility Description using Seaborn."""
        sb.barplot(schools, x="school_accessibility_description", y="total_students", errorbar=None)
        plt.xlabel("School Accessibility Description")
        plt.ylabel("Number of Students")
        plt.title("Distribution of number of students vs School Accessibility Description (using seaborn)")
        plt.show()
