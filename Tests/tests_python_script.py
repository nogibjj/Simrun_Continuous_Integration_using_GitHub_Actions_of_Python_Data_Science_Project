import sys
sys.path.append("/workspaces/Simrun_Continuous_Integration_using_GitHub_Actions_of_Python_Data_Science_Project")
sys.path.append("/workspaces/Simrun_Continuous_Integration_using_GitHub_Actions_of_Python_Data_Science_Project/src")
from src.python_script import run_statistics
import pandas as pd


def test_descriptive_stats():
    "Test the descriptive stats function"
    data = pd.read_csv("data/diabetes.csv")
    target_column = "Insulin"

    results = run_statistics(data, target_column)

    assert 'Target Column' in results
    assert 'Maximum' in results
    assert 'Minimum' in results
    assert 'Mean' in results
    assert 'Median' in results
    assert 'Standard Deviation' in results