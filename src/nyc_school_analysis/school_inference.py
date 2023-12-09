"""
This module have a class as Inferences, designed for drawing inferences and visualizing correlations within a given dataset.


Class:
    - Inferences: A class for drawing inferences and visualizing correlations.

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


class Inferences:
    """
    A class for drawing inferences and visualizing correlations.
    
    Methods:
    - __init__(self, data): Initializes the Inferences class.
    - num_sports(self): Calculate and store the number of sports-related metrics for analysis.
    - heatmap1(self): Generate and display a heatmap showing the correlation between Total Students, Number of Extracurricular Activities, Sports Offered, and Partner Opportunities.
    - heatmap2(self): Generate and display a heatmap showing the correlation between Total Students and various categories of Sports Offered.
    - heatmap3(self): Generate and display a heatmap showing the correlation between Total Students and different categories of Partner Opportunities.
    - research_ques2_scatter(self): Visualize the relationship between the availability of school sports and total student enrollment.
    - research_ques3_pairplot(self): Explore relationships between the number of partnerships, extracurricular activities, and total student population.
    - research_ques4_bar(self): Visualize the top-performing schools based on a composite score considering extracurricular activities, partnerships, sports participation, and total students.
    - research_ques4_scatter(self): Investigate the feasibility of creating composite indices representing the overall richness of a school's offerings.
    - research_ques5_heatmap(self): Investigate the distribution of extracurricular activities in different boroughs using a heatmap.
    """
    def __init__(self, data):
        """
        Initialize the 'Inferences' class with a DataFrame.

        Parameters:
        - data: pandas DataFrame
          The input data containing relevant information.
        """
        self.df = data

    def num_sports(self):
        """Calculate and store the number of sports-related metrics for analysis."""
        self.df["num_sports_boys"] = self.df["psal_sports_boys"].apply(
            lambda x: max(len(str(x).split(", ")), 0)
        )
        self.df["num_sports_girls"] = self.df["psal_sports_girls"].apply(
            lambda x: max(len(str(x).split(", ")), 0)
        )
        self.df["num_sports_coed"] = self.df["psal_sports_coed"].apply(
            lambda x: max(len(str(x).split(", ")), 0)
        )
        self.df["num_sports_othr"] = self.df["school_sports"].apply(
            lambda x: max(len(str(x).split(", ")), 0)
        )
        self.df["num_sports_tot"] = self.df[
            [
                "num_sports_boys",
                "num_sports_girls",
                "num_sports_coed",
                "num_sports_othr",
            ]
        ].sum(axis=1)
        return

    def heatmap1(self):
        """
        Generate and display a heatmap showing the correlation between Total Students,
        Number of Extracurricular Activities, Sports Offered, and Partner Opportunities.

        Steps:
        1. Select relevant columns: 'total_students', 'num_ext_act', 'num_sports_tot',
           and 'num_prtn_tot'.
        2. Calculate the correlation matrix for the selected columns.
        3. Generate a heatmap with correlation values annotated.

        Return:
        - Display the heatmap.

        """
        columns = ["total_students", "num_ext_act", "num_sports_tot", "num_prtn_tot"]
        schools_corr = self.df[columns].corr()
        sb.heatmap(schools_corr, annot=True)
        plt.title(
            "Correlation between Total Students, Number of Extracurricular Activities, Sports Offered, and Partner Opportunities"
        )
        return plt.show()

    def heatmap2(self):
        """
        Generate and display a heatmap showing the correlation between Total Students
        and various categories of Sports Offered.

        Steps:
        1. Select relevant columns: 'total_students', 'num_sports_boys',
           'num_sports_girls', 'num_sports_coed', 'num_sports_othr', and 'num_sports_tot'.
        2. Calculate the correlation matrix for the selected columns.
        3. Generate a heatmap with correlation values annotated.

        Return:
        - Display the heatmap.
        """
        schools_corr = self.df[
            [
                "total_students",
                "num_sports_boys",
                "num_sports_girls",
                "num_sports_coed",
                "num_sports_othr",
                "num_sports_tot",
            ]
        ].corr()
        sb.heatmap(schools_corr, annot=True)
        plt.title("Heatmap of Total Number of Students and Sports Offered")
        return plt.show()

    def heatmap3(self):
        """
        Generate and display a heatmap showing the correlation between Total Students
        and different categories of Partner Opportunities.

        Steps:
        1. Select relevant columns: 'total_students', 'num_prtn_cbo', 'num_prtn_hpt',
           'num_prtn_high', 'num_prtn_clt', 'num_prtn_nonprft', 'num_prtn_corp',
           'num_prtn_fin', 'num_prtn_othr', and 'num_prtn_tot'.
        2. Calculate the correlation matrix for the selected columns.
        3. Generate a heatmap with correlation values annotated.

        Return:
        - Display the heatmap.

        """
        schools_corr = self.df[
            [
                "total_students",
                "num_prtn_cbo",
                "num_prtn_hpt",
                "num_prtn_high",
                "num_prtn_clt",
                "num_prtn_nonprft",
                "num_prtn_corp",
                "num_prtn_fin",
                "num_prtn_othr",
                "num_prtn_tot",
            ]
        ].corr()
        sb.heatmap(schools_corr, annot=True)
        plt.title("Heatmap of Total Number of Students and Partner Opportunities")
        return plt.show()

    def research_ques2_scatter(self):
        """
        This function aims to visually explore the relationship between the availability
        of school sports and the total student enrollment.

        Steps:
        1. Select relevant columns: 'num_sports_tot' and 'total_students'.
        2. Drop rows with missing values in 'num_sports_tot' or 'total_students'.
        3. Create a scatter plot to visualize the correlation.

        Returns:
        - Displays the generated scatter plot.

        """
        df_subset = self.df[["num_sports_tot", "total_students"]]

        # Drop rows with missing values in 'school_sports' or 'total_students'
        df_subset = df_subset.dropna(subset=["num_sports_tot", "total_students"])
        plt.figure(figsize=(55, 10))
        plt.scatter(
            df_subset["num_sports_tot"], df_subset["total_students"], alpha=0.5, s=50
        )
        plt.title(
            "Correlation between School Sports and Student Enrollment", fontsize=40
        )
        plt.xlabel("School Sports Availability", fontsize=24)
        plt.ylabel("Total Student Enrollment", fontsize=24)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
        return plt.show()

    def research_ques3_pairplot(self):
        """
        This function aims to provide a visual exploration of the relationships between
        the number of partnerships, extracurricular activities, and the total student
        population in schools.

        Steps:
        1. Select relevant columns: 'total_students', 'num_prtn_tot', and 'num_ext_act'.
        2. Drop rows with missing values in relevant columns.
        3. Create a pairplot to visualize the relationships.

        Returns:
        - Displays the generated pairplot.

        """
        # Select relevant columns
        df_subset_rq3 = self.df[["total_students", "num_prtn_tot", "num_ext_act"]]

        # Drop rows with missing values in relevant columns
        df_subset_rq3 = df_subset_rq3.dropna(
            subset=["total_students", "num_prtn_tot", "num_ext_act"]
        )

        plt.figure(figsize=(12, 8))  # Adjust the figure size
        # Create a pairplot to visualize relationships
        sb.pairplot(df_subset_rq3)
        plt.suptitle(
            "Relationships between Partnerships, Extracurricular Activities, and Student Population",
            y=1.02,
        )
        return plt.show()

    def research_ques4_bar(self):
        """
        This function aims to visually represent the 10 top-performing schools
        based on a composite score that considers variables such as extracurricular
        activities, partnerships, sports participation, and total students.

        Steps:
        1. Select relevant columns: 'school_name', 'num_ext_act', 'num_prtn_tot',
           'num_sports_tot', and 'total_students'.
        2. Drop rows with missing values in relevant columns.
        3. Create a composite score for the richness of offerings by summing
           'num_ext_act', 'num_prtn_tot', and 'num_sports_tot'.
        4. Rank schools based on the composite score.
        5. Generate a bar plot for the top 10 schools based on the composite score.

        Returns:
        - Displays the generated bar plot.

        """
        # Select relevant columns
        df_subset = self.df[
            [
                "school_name",
                "num_ext_act",
                "num_prtn_tot",
                "num_sports_tot",
                "total_students",
            ]
        ]

        # Drop rows with missing values in relevant columns
        df_subset = df_subset.dropna(
            subset=["num_ext_act", "num_prtn_tot", "num_sports_tot", "total_students"]
        )

        # Create a composite score for the richness of offerings
        df_subset["composite_score"] = df_subset["num_ext_act"] + df_subset[
            ["num_prtn_tot", "num_sports_tot"]
        ].sum(axis=1)

        # Rank schools based on the composite score
        df_subset_ranked = df_subset.sort_values(by="composite_score", ascending=False)

        # Plotting the results
        plt.figure(figsize=(12, 8))

        # Bar plot for the top 10 schools based on the composite score
        sb.barplot(
            x="composite_score",
            y="school_name",
            data=df_subset_ranked.head(10),
            palette="viridis",
        )
        plt.title("Top 10 Schools Based on Composite Score of Offerings")
        plt.xlabel("Composite Score")
        plt.ylabel("School Name")

        return plt.show()

    def research_ques4_scatter(self):
        """
        This function investigates the feasibility of creating composite indices or scores
        that represent the overall richness of a school's offerings. It considers variables
        such as extracurricular activities, partnerships, and sports participation.

        Steps:
        1. Select relevant columns: 'school_name', 'extracurricular_activities',
           'num_ext_act', 'num_prtn_tot', and 'num_sports_tot'.
        2. Drop rows with missing values in relevant columns.
        3. Create a composite score for the richness of offerings by summing
           'num_ext_act', 'num_prtn_tot', and 'num_sports_tot'.
        4. Generate a scatter plot with 'num_ext_act' on the x-axis, 'num_prtn_tot' on
           the y-axis, and 'num_sports_tot' represented by marker size. The color of
           markers represents the composite score.
        5. Display the scatter plot with appropriate labels and legend.

        Returns:
        - Displays the generated scatter plot.

        """
        # Select relevant columns
        df_subset = self.df[
            [
                "school_name",
                "extracurricular_activities",
                "num_ext_act",
                "num_prtn_tot",
                "num_sports_tot",
            ]
        ]

        # Drop rows with missing values in relevant columns
        df_subset = df_subset.dropna(
            subset=[
                "extracurricular_activities",
                "num_ext_act",
                "num_prtn_tot",
                "num_sports_tot",
            ]
        )

        # Create a composite score for the richness of offerings
        df_subset["composite_score"] = (
            df_subset["num_ext_act"]
            + df_subset["num_prtn_tot"]
            + df_subset["num_sports_tot"]
        )

        # Plotting the results
        plt.figure(figsize=(12, 8))

        # Scatter plot of schools based on the composite score
        sb.scatterplot(
            x="num_ext_act",
            y="num_prtn_tot",
            size="num_sports_tot",
            hue="composite_score",
            data=df_subset,
            palette="viridis",
            sizes=(20, 200),
        )
        plt.title("Schools Based on Composite Score of Offerings")
        plt.xlabel("Extracurricular Activities (Length)")
        plt.ylabel("Partner Total")
        plt.legend(title="Composite Score")
        return plt.show()

    def research_ques5_heatmap(self):
        """
        This function aims to investigate the distribution of extracurricular activities
        in different boroughs, identifying patterns and trends. Insights gained can inform
        efforts to enhance opportunities and increase participation among students

        Steps:
        1. Extract relevant columns: 'school_name', 'borough', and 'num_ext_act'.
        2. Remove rows with missing values in 'borough' and 'num_ext_act'.
        3. Group the subset by 'borough' and 'num_ext_act', counting occurrences.
        4. Create a pivot table to summarize the data.
        5. Plot a heatmap to visualize the distribution.

        This function aims to investigate the distribution of extracurricular activities
        in different boroughs, identifying patterns and trends. Insights gained can inform
        efforts to enhance opportunities and increase participation among students.

        Returns:
        - Displays the generated heatmap.

        """
        df_subset_rq5 = self.df[["school_name", "borough", "num_ext_act"]]

        # Drop rows with missing values in relevant columns
        df_subset_rq5 = df_subset_rq5.dropna(subset=["borough", "num_ext_act"])

        # Group by 'borough' and 'extracurricular_activities' and count occurrences
        grouped_data = (
            df_subset_rq5.groupby(["borough", "num_ext_act"])
            .size()
            .reset_index(name="count")
        )

        # Pivot the data to create a summary table
        pivot_table = pd.pivot_table(
            grouped_data,
            values="count",
            index=["borough"],
            columns=["num_ext_act"],
            fill_value=0,
        )

        # Create a heatmap to visualize the distribution
        plt.figure(figsize=(10, 6))
        sb.heatmap(pivot_table, annot=True, cmap="viridis")
        plt.title("Distribution of Extracurricular Activities Across Boroughs")
        plt.xlabel("num_ext_act")
        plt.ylabel("Borough")
        plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
        return plt.show()
