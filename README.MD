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

The next step after the initial Dataset loading from CSV file is to take a look at the datastructure:
```
print(data_frame.head())
print(data_frame.info())
```
The **_'head()'_** is a **Pandas** method to display **first 5 rows** of DataFrame just to explore it structure:
![1_first_five_rows.png](assets%2F1_first_five_rows.png)

The **_info()_** is also a **Pandas** method, it job is to display basic information about the DataFrame and 
the Columns it contains, specifically I'm interested in what datatype these columns contain:
![2_dataframe_info.png](assets%2F2_dataframe_info.png)

# Cleaning & Transforming the DataFrame

Now it's time to **Clean & Transform**✨ this DataFrame!

### 1. Fixing the Column datatype🔧
As you might notice, in the previous section, when we were checking the datatypes of columns _info()_ 
method showed that the datatype of 'Date_Built' column is 'object':  
![3_wrong_datatype.png](assets%2F3_wrong_datatype.png)

Basically, Pandas interpreted values in this column as 'string' but the correct datatype for this 
column should be **_'datetime'_**.  
To fix it I'll simply convert all values in this column to a 'datetime' datatype using **Pandas** method 
**_'to_datetime()'_**:
```
data_frame['Date_Built'] = pd.to_datetime(data_frame['Date_Built'])
```
Now if I check the datatype of this column again, using _info()_ method, It will show that the datatype 
of this column now is **_'datatime'_**, just like it should be!    
![4_fixed_datatype.png](assets%2F4_fixed_datatype.png)

### 2. Replacing values in the Columns🔁

Another thing that I want to do with this DataFrame is to replace binary '1' and '0' values to boolean 
'True' or 'False' in these columns:  
![5_int_to_boolean.png](assets%2F5_int_to_boolean.png)



# Descriptive Statistics & Normal Distribution

# Correlation & Regression Analysis

# Conclusion