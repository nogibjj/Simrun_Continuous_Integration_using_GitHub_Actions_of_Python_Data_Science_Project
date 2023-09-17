import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def maximum(data, column):
    """
        Input:
            Data: Dataframe to calculate maximum
            Column: Target column
        Output: 
            Return the maximum of the Target column
    """

    max_value = data[column].max()

    return max_value

def minimum(data, column):
    """
        Input:
            Data: Dataframe to calculate minimum
            Column: Target column
        Output: 
            Return the minimum of the Target column
    """

    min_value = data[column].min()

    return min_value


def get_mean(data,column):
    """
        Input:
            Data: Dataframe to calculate mean
            Column: Target column
        Output: 
            Return the mean of the Target column
    """

    mean_value = data[column].mean()

    return mean_value


def get_median(data,column):
    """
        Input:
            Data: Dataframe to calculate median
            Column: Target column
        Output:
            Return the median of the Target column
    """

    median_value = data[column].median()

    return median_value


def get_std_dev(data,column):
    """
        Input:
            Data: Dataframe to calculate standard deviation
            Column: Target column
        Output:
            Return the standard deviation of the Target column
    """

    std_dev_value = data[column].std()

    return std_dev_value


def visualize_dataset(data, outcome_var: str, target_var: str, jupyter: bool=False):
    """Visualizes the data and their specified columns. Replaces NAN values for mean or median. Creates a heatmap of 
    predictor variable correlations.Created a scatter plot with a line of best fit for each predictor variable. Includes legend
     with mean,median, maximum, minimum, and standard deviation """

    # Drop the null values
    removed_NaN_data = data[['Glucose','Insulin','BMI']].replace(0,np.NaN)
    null_values_list = removed_NaN_data.isnull().sum()
    removed_NaN_data['Glucose'].fillna(removed_NaN_data['Glucose'].mean(), inplace = True)
    removed_NaN_data['Insulin'].fillna(removed_NaN_data['Insulin'].median(), inplace = True)
    removed_NaN_data['BMI'].fillna(removed_NaN_data['BMI'].median(), inplace = True)

    # Creating a countplot based on datatype
    diabetes_count_plot = sns.countplot(y=data.dtypes,data=removed_NaN_data)
    plt.xlabel("data type count of diabetes predictors")
    plt.ylabel("data types of diabetes predictors")
    plt.title("Count Plot of Diabetes Predictors Data Types")
    plt.show()

    # Glucose Scatter Plot
    glucose_fig = plt.figure(figsize=(10, 6))
    glucose_scatter = plt.scatter(
    removed_NaN_data["Glucose"],
    removed_NaN_data["Outcome"],
    alpha=0.7)
    glucose_scatter.set_title("Glucose vs. Is Diabetic?")
    glucose_scatter.set_xlabel("Glucose(mg/dL)")
    glucose_scatter.set_ylabel("Diabetes")

    # BMI Scatter Plot
    BMI_fig = plt.figure(figsize=(10, 6))
    BMI_scatter = plt.scatter(
    removed_NaN_data["BMI"],
    removed_NaN_data["Outcome"],
    alpha=0.7)
    BMI_scatter.set_title("BMI vs. Is Diabetic?")
    BMI_scatter.set_xlabel("BMI(kg/m^2)")
    BMI_scatter.set_ylabel("Diabetes")

    # Insulin Scatter Plot
    Insulin_fig = plt.figure(figsize=(10, 6))
    Insulin_scatter = plt.scatter(
    removed_NaN_data["Insulin"],
    removed_NaN_data["Outcome"],
    alpha=0.7)
    Insulin_scatter.set_title("Insulin vs. Is Diabetic?")
    Insulin_scatter.set_xlabel("Insulin(mu/mL)")
    Insulin_scatter.set_ylabel("Diabetes")

    # text on the plot for glucose
    mean_g_value = get_mean(removed_NaN_data,"Glucose")
    median_g_value = get_median(removed_NaN_data, "Glucose")
    std_dev_g_value = get_std_dev(removed_NaN_data, "Glucose")
    max_g_value = maximum(removed_NaN_data,"Glucose")
    min_g_value = minimum(removed_NaN_data, "Glucose")
    g_text = f"Mean Glucose:{mean_g_value}\nMedian Glucose: {median_g_value}\nStd Dev Glucose: {std_dev_g_value}\nMaximum Glucose: {max_g_value}\nMinimum Glucose: {min_g_value}"
    plt.g_text(
    removed_NaN_data["Glucose"].max() - 0.5,
    removed_NaN_data["Diabetes"].max() - 20,
    g_text,
    bbox=dict(facecolor="white", alpha=0.7),
    horizontalalignment="right",
    verticalalignment="top",
)

# text on plot for Insulin
    mean_i_value = get_mean(removed_NaN_data,"Insulin")
    median_i_value = get_median(removed_NaN_data, "Insulin")
    std_dev_i_value = get_std_dev(removed_NaN_data, "Insulin")
    max_i_value = maximum(removed_NaN_data,"Insulin")
    min_i_value = minimum(removed_NaN_data, "Insulin")
    i_text = f"Mean Insulin:{mean_i_value}\nMedian Insulin: {median_i_value}\nStd Dev Insulin: {std_dev_i_value}\nMaximum Insulin: {max_i_value}\nMinimum Insulin: {min_i_value}"
    plt.i_text(
    removed_NaN_data["Glucose"].max() - 0.5,
    removed_NaN_data["Diabetes"].max() - 20,
    i_text,
    bbox=dict(facecolor="white", alpha=0.7),
    horizontalalignment="right",
    verticalalignment="top",
)

#text on plot for BMI
    mean_b_value = get_mean(removed_NaN_data,"BMI")
    median_b_value = get_median(removed_NaN_data, "BMI")
    std_dev_b_value = get_std_dev(removed_NaN_data, "BMI")
    max_b_value = maximum(removed_NaN_data,"BMI")
    min_b_value = minimum(removed_NaN_data, "BMI")
    b_text = f"Mean BMI:{mean_b_value}\nMedian BMI: {median_b_value}\nStd Dev BMI: {std_dev_b_value}\nMaximum BMI: {max_b_value}\nMinimum BMI: {min_b_value}"
    plt.b_text(
    removed_NaN_data["BMI"].max() - 0.5,
    removed_NaN_data["Diabetes"].max() - 20,
    b_text,
    bbox=dict(facecolor="white", alpha=0.7),
    horizontalalignment="right",
    verticalalignment="top",
)

    if jupyter:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        plt.legend(loc='best')

    plt.show()
    visualization_path = 'output/visualization.png'

    if not jupyter:
        plt.savefig(visualization_path)  # save png

        # Save generated report
        summary_report_path = r'output/generated_report.md'
        with open(summary_report_path, "w", encoding="utf-8") as report:
            report.write(f'Mean: {round(mean, 3)} \n \n \n')
            report.write(f'Median: {round(median, 3)} \n \n \n')
            report.write(f'Standard Deviation: {round(stand_dev, 3)} \n \n \n')
            report.write("\n![Visualization](visualization.png)\n")


    if not jupyter:
        visualization_path = 'output/visualization_hist.png'
        plt.savefig(visualization_path)  # save png


if __name__ == "__main__":
    data = pd.read_csv("data/diabetes.csv")
    TARGET_COLUMN = "Insulin"

    print('Target Column: ', 'TARGET_COLUMN')
    print('Maximum Value: ', maximum(data, TARGET_COLUMN))
    print('Minimum Value: ', minumum(data, TARGET_COLUMN))
    print('Mean: ', get_mean(data, TARGET_COLUMN))
    print('Median: ', get_median(data, TARGET_COLUMN))
    print("Standard Deviation: ", get_std_dev(data, TARGET_COLUMN))

    visualize_dataset(data, "petal_width", TARGET_COLUMN, "species", jupyter=False)