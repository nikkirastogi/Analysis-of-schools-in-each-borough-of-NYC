import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

class Inferences:
    def __init__(self, data):
        """
        Initialize the 'Inferences' class with a DataFrame.

        Parameters:
        - data: pandas DataFrame
          The input data containing relevant information.
        """
        self.df = data

    def heatmap1(self):
        """
        Generate and display a heatmap showing the correlation between Total Students,
        Number of Extracurricular Activities, Sports Offered, and Partner Opportunities.

        """
        columns = ["total_students", "num_ext_act", "num_sports_tot", "num_prtn_tot"]
        schools_corr = self.df[columns].corr()
        sb.heatmap(schools_corr, annot=True)
        plt.title("Correlation between Total Students, Number of Extracurricular Activities, Sports Offered, and Partner Opportunities")
        return plt.show()

    def heatmap2(self):
        """
        Generate and display a heatmap showing the correlation between Total Students
        and various categories of Sports Offered.

        """
        schools_corr = self.df[["total_students", "num_sports_boys", "num_sports_girls", "num_sports_coed", "num_sports_othr", "num_sports_tot"]].corr()
        sb.heatmap(schools_corr, annot=True)
        plt.title("Heatmap of Total Number of Students and Sports Offered")
        return plt.show()

    def heatmap3(self):
        """
        Generate and display a heatmap showing the correlation between Total Students
        and different categories of Partner Opportunities.
        
        """
        schools_corr = self.df[["total_students", "num_prtn_cbo", "num_prtn_hpt", "num_prtn_high", "num_prtn_clt", "num_prtn_nonprft", "num_prtn_corp", "num_prtn_fin", "num_prtn_othr", "num_prtn_tot"]].corr()
        sb.heatmap(schools_corr, annot=True)
        plt.title("Heatmap of Total Number of Students and Partner Opportunities")
        return plt.show()
    
    def research_ques2_scatter(self):
        df_subset = self.df[['num_sports_tot', 'total_students']]

        # Drop rows with missing values in 'school_sports' or 'total_students'
        df_subset = df_subset.dropna(subset=['num_sports_tot', 'total_students'])
        plt.figure(figsize=(55, 10)) 
        plt.scatter(df_subset['num_sports_tot'], df_subset['total_students'], alpha=0.5, s=50)
        plt.title('Correlation between School Sports and Student Enrollment', fontsize=40)
        plt.xlabel('School Sports Availability', fontsize=24)
        plt.ylabel('Total Student Enrollment', fontsize=24)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
        return plt.show()

    def research_ques3_pairplot(self):
         # Select relevant columns
        df_subset_rq3 = self.df[['total_students', 'num_prtn_tot', 'num_ext_act']]

        # Drop rows with missing values in relevant columns
        df_subset_rq3 = df_subset_rq3.dropna(subset=['total_students', 'num_prtn_tot', 'num_ext_act'])
        
        plt.figure(figsize=(12, 8))  # Adjust the figure size
        # Create a pairplot to visualize relationships
        sb.pairplot(df_subset_rq3)
        plt.suptitle('Relationships between Partnerships, Extracurricular Activities, and Student Population', y=1.02)
        return plt.show()

    def research_ques4_bar(self):
        # Select relevant columns
        df_subset = self.df[['school_name', 'num_ext_act', 'num_prtn_tot', 'num_sports_tot', 'total_students']]

        # Drop rows with missing values in relevant columns
        df_subset = df_subset.dropna(subset=['num_ext_act', 'num_prtn_tot', 'num_sports_tot', 'total_students'])

        # Create a composite score for the richness of offerings
        df_subset['composite_score'] = (
        df_subset['num_ext_act'] +
        df_subset[['num_prtn_tot', 'num_sports_tot']].sum(axis=1))

        # Rank schools based on the composite score
        df_subset_ranked = df_subset.sort_values(by='composite_score', ascending=False)

        # Plotting the results
        plt.figure(figsize=(12, 8))

        # Bar plot for the top 10 schools based on the composite score
        sb.barplot(x='composite_score', y='school_name', data=df_subset_ranked.head(10), palette='viridis')
        plt.title('Top 10 Schools Based on Composite Score of Offerings')
        plt.xlabel('Composite Score')
        plt.ylabel('School Name')

        return plt.show()
        
    def research_ques4_scatter(self):
        # Select relevant columns
        df_subset = self.df[['school_name', 'extracurricular_activities','num_ext_act', 'num_prtn_tot', 'num_sports_tot']]

        # Drop rows with missing values in relevant columns
        df_subset = df_subset.dropna(subset=['extracurricular_activities','num_ext_act', 'num_prtn_tot', 'num_sports_tot'])

        # Convert 'extracurricular_activities', 'partner_total', and 'school_sports' to numeric variables
        #df_subset['extracurricular_activities'] = df_subset['extracurricular_activities'].apply(len)
        #df_subset['partner_total'] = pd.to_numeric(df_subset['partner_total'], errors='coerce')
        #df_subset['school_sports'] = pd.to_numeric(df_subset['school_sports'], errors='coerce')

        # Create a composite score for the richness of offerings
        df_subset['composite_score'] = (
            df_subset['num_ext_act'] +
            df_subset['num_prtn_tot'] +
            df_subset['num_sports_tot']
        )

        # Plotting the results
        plt.figure(figsize=(12, 8))

        # Scatter plot of schools based on the composite score
        sb.scatterplot(x='num_ext_act', y='num_prtn_tot', size='num_sports_tot', hue='composite_score', data=df_subset, palette='viridis', sizes=(20, 200))
        plt.title('Schools Based on Composite Score of Offerings')
        plt.xlabel('Extracurricular Activities (Length)')
        plt.ylabel('Partner Total')
        plt.legend(title='Composite Score')
        return plt.show()

        
        
    def research_ques5_heatmap(self):
        
        df_subset_rq5 = self.df[['school_name', 'borough', 'num_ext_act']]

        # Drop rows with missing values in relevant columns
        df_subset_rq5 = df_subset_rq5.dropna(subset=['borough', 'num_ext_act'])

        # Group by 'borough' and 'extracurricular_activities' and count occurrences
        grouped_data = df_subset_rq5.groupby(['borough', 'num_ext_act']).size().reset_index(name='count')

        # Pivot the data to create a summary table
        pivot_table = pd.pivot_table(grouped_data, values='count', index=['borough'], columns=['num_ext_act'], fill_value=0)

        # Create a heatmap to visualize the distribution
        plt.figure(figsize=(10, 6))
        sb.heatmap(pivot_table, annot=True, cmap='viridis')
        plt.title('Distribution of Extracurricular Activities Across Boroughs')
        plt.xlabel('num_ext_act')
        plt.ylabel('Borough')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
        return plt.show()
