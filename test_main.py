from main import computation

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
    

def test_computation():
    wdi = pd.read_csv(
    "https://media.githubusercontent.com/"
    "media/nickeubank/MIDS_Data/"
    "master/World_Development_Indicators/wdi_small_tidy_2015.csv")

    wdi["Log GDP Per Capita"] = np.log(wdi["GDP per capita (constant 2010 US$)"])
    wdi.dropna(subset=["Log GDP Per Capita"], inplace=True)
    result = computation(wdi)
    expected_mean = sum(wdi["Log GDP Per Capita"])/len(wdi["Log GDP Per Capita"])
    expected_mean = round(expected_mean,10)
    sort_wdi = sorted(wdi["Log GDP Per Capita"])
    n = len(sort_wdi)
    if n % 2 == 0:
      expected_median = (sort_wdi[n //2 - 1] + sort_wdi[n //2]) / 2
      expected_median = round(expected_median,10)
    else:
      expected_median = sort_wdi[n//2]
      expected_median = round(expected_median,10)
   
    expected_standard_dev = 1.47
    
    assert result == (expected_mean, expected_median, expected_standard_dev)




if __name__ == "__main__":
    pytest.main()
    test_computation()
