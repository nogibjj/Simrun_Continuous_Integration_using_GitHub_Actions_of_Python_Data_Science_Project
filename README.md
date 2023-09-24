# Data Analysis: Glucose, Insulin, and BMI as predictor variables on the outcome variable (whether a patient is Diabetic or not)

[![Install](https://github.com/nogibjj/Simrun_Continuous_Integration_using_GitHub_Actions_of_Python_Data_Science_Project/actions/workflows/install.yml/badge.svg)](https://github.com/nogibjj/Simrun_Continuous_Integration_using_GitHub_Actions_of_Python_Data_Science_Project/actions/workflows/install.yml)[![Lint](https://github.com/nogibjj/Simrun_Continuous_Integration_using_GitHub_Actions_of_Python_Data_Science_Project/actions/workflows/lint.yml/badge.svg)](https://github.com/nogibjj/Simrun_Continuous_Integration_using_GitHub_Actions_of_Python_Data_Science_Project/actions/workflows/lint.yml)[![Format](https://github.com/nogibjj/Simrun_Continuous_Integration_using_GitHub_Actions_of_Python_Data_Science_Project/actions/workflows/format.yml/badge.svg)](https://github.com/nogibjj/Simrun_Continuous_Integration_using_GitHub_Actions_of_Python_Data_Science_Project/actions/workflows/format.yml)[![Test](https://github.com/nogibjj/Simrun_Continuous_Integration_using_GitHub_Actions_of_Python_Data_Science_Project/actions/workflows/test.yml/badge.svg)](https://github.com/nogibjj/Simrun_Continuous_Integration_using_GitHub_Actions_of_Python_Data_Science_Project/actions/workflows/test.yml)

## Data Source
This dataset contains information about female patients that are at least 21 years old and are of Pima Indian Heritage. From each patient the dataset gleans diagnostic measurements of BMI, Insulin, Pregnancies, Blood Pressure, Skin Thickness, Glucose, etc. This data was collected from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to determine if the patient has diabetes or not. The outcome variable tells with a (1-Diabetic) v. (0-Not Diabetic). I specifcally wanted to analyze Insulin, Glucose, and BMI as my predictor variables. I conduct a descriptive analysis/statistics of this dataset to deeper understand these variables relationship with Diabetes.
You can access the data from the following URL: [Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?select=diabetes.csv).

## Code Overview

### 1. Shared Common Code Script: lib.py
In this script I am using many different functions to create a holistic understanding of my statistical analysis along with visualizing the data in three different plots and tables. 
  1. get_median()
  2. get_mean()
  3. get_std_dev()
  4. minimum()
  5. maximum()
  6. visualize_dataset()
  7. display_statistics()

### 2. Jupyter Notebook and Python Script.py
The Jupyter notebook provides a visual representation of the code represented in our lib.py. The python script.py descriptive stats uses pandas and lib functions. I used nbval for pytest to ensure valdity of code. 

### 3. MakeFile and Workflows
I created four different workflows to show each step of my Makefile. Install: runs the packages indicated in my requirements.txt
Lint: Lints the code with Ruff to ensure correct python convention is held
Format: Formats code with Python black formatter
Test: Runs tests on notebook, script, and library

### 4. Test_script.py and Test_lib.py
Both these files containes tests for their respective python scripts. This ensures the code in functionally performs well.

### 5. Understanding the statistics functions
The objective is to analyze the relationship between Insulin, Glucose, BMI, and diabetes status (1 for diabetic, 0 for not diabetic). The functions defined calculate maximum, minimum, mean, median, and standard deviation for these key variables, aiding in a descriptive analysis of their association with diabetes.

### 6. Understanding the Data Visualization functions
In this code, count plots offer a glimpse into the distribution of data types, especially the prevalence of diabetic and non-diabetic cases. Understanding this distribution is vital for addressing class imbalance and potential biases in a diabetes prediction scenario. On the other hand, bar plots help visually compare predictor variables like Glucose, Insulin, and BMI against diabetes status, revealing insights into their association with the outcome. These visualizations aid in identifying patterns and trends, such as higher Glucose levels being more common among diabetic patients. Overall, count and bar plots provide valuable visual context, enabling informed decisions in diabetes research and predictive modeling.

## Results

![After executing the code, the following descriptive statistics are for Glucose, Insulin, and BMI](https://user-images.githubusercontent.com/141798228/268531816-2f9924ab-d11c-422a-b509-bb3cb042a723.jpg)

These statistics offer insights into the central tendency and spread of the diabetes patients, contributing to a better understanding of its distribution.

The visualization below illustrates the relationship between the predictor variable (Insulin, Glucose, BMI) and the Outcome variable (Whether a patient has diabetes or doesn't have diabetes)

![Countplot](https://user-images.githubusercontent.com/141798228/268531789-4ee528a7-c91f-4281-b2e6-ea15daa89a42.png)
![Barplot](https://user-images.githubusercontent.com/141798228/268531736-b20fb998-b839-4c6a-b8af-3749b5ed8f4b.png)

## Conclusion
This script serves as a valuable tool for investigating the interplay between crucial predictor variables—Glucose, Insulin, and BMI—and the diabetes status indicator within the context of the Pima Indian female patient dataset. By offering essential descriptive statistics and intuitive visualizations like count plots and bar plots, it allows for a deeper understanding of how these variables are linked to the likelihood of diabetes. These insights provide a solid foundation for conducting more advanced analyses and informed decision-making in the domain of diabetes research and prediction
