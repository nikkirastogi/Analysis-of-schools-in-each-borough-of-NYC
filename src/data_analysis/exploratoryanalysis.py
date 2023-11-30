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
        return "Total number of students enrolled across all schools: {:,.0f}".format(
            self.df.total_students.sum()
        )

    def matplot_hist_total_students(self):
        """Generate and display a histogram of the total number of students using Matplotlib."""

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
        return pd.DataFrame(
            {"borough": borough.index, "values": np.round(borough.values, 2)}
        )

    def matplot_bar_borough(self):
        """Generate and display a barplot indicating the population of each borough using Matplotlib."""
        borough = self.df["borough"].value_counts(normalize=True) * 100
        borough = pd.DataFrame(
            {"borough": borough.index, "values": np.round(borough.values, 2)}
        )

        plt.bar(borough["borough"], borough["values"])
        plt.xlabel("Borough of School")
        plt.ylabel("Percentage")
        plt.title("Barplot indicating Population of Each Borough (Using Matplotlib)")
        return plt.show()

    def seaborn_bar_borough(self):
        """Generate and display a barplot indicating the population of each borough using Seaborn."""
        borough = self.df["borough"].value_counts(normalize=True) * 100
        borough = pd.DataFrame(
            {"borough": borough.index, "values": np.round(borough.values, 2)}
        )

        ax = sb.barplot(data=borough, x="borough", y="values", palette="viridis")
        plt.xlabel("Borough of School")
        plt.ylabel("Percentage")
        plt.title("Barplot indicating Population of Each Borough (Using Seaborn)")
        for i in ax.containers:
            ax.bar_label(
                i,
            )
        return plt.show()

    def matplot_bar_grpby_borough_tot_students(self):
        """
        Generate and display a bar plot showing the distribution of students enrolled in each borough using Matplotlib.
        """
        data = self.df.groupby("borough")["total_students"].sum()
        plt.bar(data.index, data.values)
        plt.xlabel("Borough of School")
        plt.ylabel("Number of Students")
        plt.title(
            "Distribution of students enrolled in each Borough (using matplotlib)"
        )
        return plt.show()

    def seaborn_bar_borough_tot_students(self):
        """Generate and display a barplot showing the distribution of students enrolled in each borough using Seaborn."""
        ax = sb.barplot(
            self.df, x="borough", y="total_students", errorbar=None, estimator="sum"
        )
        plt.xlabel("Borough of School")
        plt.ylabel("Number of Students")
        plt.title("Distribution of students enrolled in each Borough (using seaborn)")
        for i in ax.containers:
            ax.bar_label(
                i,
            )
        return plt.show()

    def unique_activities(self):
        """
        Calculate and return the number of unique Extracurricular Activities provided by all schools.

        Returns:
        - str: Formatted string indicating the number of unique extracurricular activities.
        """
        return "Number of unique Extracurricular Activities provided by all schools: {:,}".format(
            self.df["extracurricular_activities"]
            .apply(lambda x: str(x).split(", "))
            .explode()
            .nunique()
        )

    def activities_desc(self):
        self.df = self.df.copy()
        """Describe the field 'num_ext_act' representing the number of extracurricular activities."""
        self.df["num_ext_act"] = self.df["extracurricular_activities"].apply(
            lambda x: max(len(str(x).split(", ")), 0)
        )
        return self.df["num_ext_act"].describe()

    def matplot_scatter_tot_studs_vs_activities(self):
        """Generate and display a scatter plot of total_students vs the number of extracurricular activities offered using Matplotlib."""
        plt.scatter(x=self.df["total_students"], y=self.df["num_ext_act"])
        plt.xlabel("Number of students")
        plt.ylabel("Number of Extracurricular activities offered")
        plt.title(
            "Distribution of total_students vs Number of Extracurricular activities offered (using matplotlib)"
        )
        return plt.show()

    def seaborn_scatter_tot_studs_vs_activities(self):
        """Generate and display a scatter plot of total_students vs the number of extracurricular activities offered using Seaborn."""
        sb.scatterplot(data=self.df, x="total_students", y="num_ext_act", color="olive")
        plt.xlabel("Number of students")
        plt.ylabel("Number of Extracurricular activities offered")
        plt.title(
            "Distribution of total_students vs Number of Extracurricular activities offered (using seaborn)"
        )
        return plt.show()

    def num_prtn(self):
        """Calculate and store the number of partner-related metrics for analysis."""
        self.df["num_prtn_cbo"] = self.df["partner_cbo"].apply(
            lambda x: max(len(str(x).split(", ")), 0)
        )
        self.df["num_prtn_hpt"] = self.df["partner_hospital"].apply(
            lambda x: max(len(str(x).split(", ")), 0)
        )
        self.df["num_prtn_high"] = self.df["partner_highered"].apply(
            lambda x: max(len(str(x).split(", ")), 0)
        )
        self.df["num_prtn_clt"] = self.df["partner_cultural"].apply(
            lambda x: max(len(str(x).split(", ")), 0)
        )
        self.df["num_prtn_nonprft"] = self.df["partner_nonprofit"].apply(
            lambda x: max(len(str(x).split(", ")), 0)
        )
        self.df["num_prtn_corp"] = self.df["partner_corporate"].apply(
            lambda x: max(len(str(x).split(", ")), 0)
        )
        self.df["num_prtn_fin"] = self.df["partner_financial"].apply(
            lambda x: max(len(str(x).split(", ")), 0)
        )
        self.df["num_prtn_othr"] = self.df["partner_other"].apply(
            lambda x: max(len(str(x).split(", ")), 0)
        )

        self.df["num_prtn_tot"] = self.df[
            [
                "num_prtn_cbo",
                "num_prtn_hpt",
                "num_prtn_high",
                "num_prtn_clt",
                "num_prtn_nonprft",
                "num_prtn_corp",
                "num_prtn_fin",
                "num_prtn_othr",
            ]
        ].sum(axis=1)
        return self.df

    def matplot_scatter_students_vs_partners(self):
        """Generate and display a scatter plot of total_students vs the number of partner opportunities using Matplotlib."""
        plt.scatter(x=self.df["total_students"], y=self.df["num_prtn_tot"])
        plt.xlabel("Number of students")
        plt.ylabel("Number of Partner opportunities")
        plt.title(
            "Distribution of total_students vs Number of Partner Opportunities available (using matplotlib)"
        )
        return plt.show()

    def seaborn_scatter_students_vs_partners(self):
        """
        Generate and display a scatter plot of total_students vs the number of partner opportunities using Seaborn.
        """
        sb.scatterplot(
            data=self.df, x="total_students", y="num_prtn_tot", color="coral"
        )
        plt.xlabel("Number of students")
        plt.ylabel("Number of Partner opportunities")
        plt.title(
            "Distribution of total_students vs Number of Partner Opportunities available (using seaborn)"
        )
        return plt.show()
