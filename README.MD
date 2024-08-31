# Introduction

Hi! This is my **Python Portfolio Project** 🐉 that involves working with Python
libraries & modules such as **NumPy**, **Pandas**, **Matplotlib**, **SciPy** 
and **Statsmodels**. The dataset I am working on is a CSV file with apartment🏚️ rental information.


The project includes such steps as importing all the necessary libraries & modules, creating a 
**Pandas DataFrame** from a CSV file📅, **Cleaning & Transforming** the DataFrame, applying **Descriptive
Statistics** and checking the **Normal Distribution** of apartment prices, analyzing the **Correlation**
and predicting apartment price based on its size using **Regression Analysis**📈!

You can find the **Python** file with full code from this repository or directly by this link: [apartment_analysis.py](apartment_analysis.py).  
And also the actual **CSV** Dataset file: [german_apartment_rentals.csv](german_apartment_rentals.csv).

# Importing Modules & Creating Pandas DataFrame from CSV file

At first, I imported all necessary modules & libraries and created Pandas DataFrame from CSV file:

```
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

data_frame = pd.read_csv('german_apartment_rentals.csv')
```

I used **Pandas** to read, clean and transform the Dataset📅.  
**Matplotlib** is useful for creating charts to graphically display the data📊.  
**SciPy** and **Statsmodels** are amazing libraries that I used to apply the 
**Descriptive Statistics** & **Regression Analysis**📈

# First Look at the Dataset Structure

# Cleaning & Transforming the DataFrame

# Descriptive Statistics & Normal Distribution

# Correlation & Regression Analysis

# Conclusion