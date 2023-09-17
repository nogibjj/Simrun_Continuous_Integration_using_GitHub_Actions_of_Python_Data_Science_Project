"We will use pytest to test our functions from src/lib.py"
import sys
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("/workspaces/Simrun_Continuous_Integration_using_GitHub_Actions_of_Python_Data_Science_Project")
from src.lib import get_mean, get_median, get_std_dev, maximum, minimum, visualize_dataset


def test_get_mean():
    """Test function for return_25th_quantile"""
    data = pd.read_csv("data/iris_data.csv")
    target_column = 'sepal_width'

    res =  return_25th_quantile(data, target_column)

    # hand calculations
    data = data.sort_values(by=target_column)
    quan_25th = data.iloc[floor(data.shape[0] / 4)][target_column]

    assert res == quan_25th


def test_return_mean():
    """Test function for return_mean"""
    data = pd.read_csv("data/iris_data.csv")
    target_column = 'sepal_width'

    # hand calculation
    expected_mean_a = sum(data[target_column]) / len(data[target_column])

    # function calculation
    calculated_mean_a = return_mean(data, target_column)

    # Check if the calculated mean matches the expected mean
    assert round(calculated_mean_a) == round(expected_mean_a)


def test_return_std_dev():
    """Test function for return_std_dev"""
    data = {'A': [1, 2, 3, 4, 5]}
    data = pd.DataFrame(data)

    result = return_std_dev(data, 'A')

    assert isinstance(result, float)
    assert round(result, 2) == 1.58

def test_return_median():
    """Test function for return_median"""
    data = pd.read_csv("data/iris_data.csv")
    target_column = 'sepal_width'

    # hand calculation
    expected_median = data[target_column].median()

    # function calculation
    calculated_median = return_median(data, target_column)

    # Check if the calculated mean matches the expected mean
    assert round(calculated_median) == round(expected_median)

def test_visualize_dataset():
    """Testing function for visualization"""

    data = pd.read_csv("data/iris_data.csv")


    # Test if the function executes without errors
    visualize_dataset(data, jupyter=False)

    # Capture the plot output and check if it's not empty
    fig = plt.gcf()
    assert len(fig.axes) > 0


if __name__ == '__main__':
    test_get_mean()
