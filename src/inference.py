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
        #schools_corr = self.df["total_students", "num_ext_act", "num_sports_tot", "num_prtn_tot"].corr()
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
