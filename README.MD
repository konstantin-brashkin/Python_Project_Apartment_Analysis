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

To implement this I just converted values in this columns to boolean datatype using **Pandas** method
**_astype()_**:
```
data_frame['Has_Balcony'] = data_frame['Has_Balcony'].astype(bool)
data_frame['Has_Elevator'] = data_frame['Has_Elevator'].astype(bool)
data_frame['Has_Parking'] = data_frame['Has_Parking'].astype(bool)
```
Now values in these columns looks like this:  
![6_int_to_bolean_2.png](assets%2F6_int_to_bolean_2.png)

### 3. Setting up 'apart_id' as an Index Column🆔

The good practice while working with **Pandas DataFrame** when you have an **'id' Primary key** column
is to use this column as an Index Column, right now 'apart_id' is individual, seperated column:  
![7_apart_id.png](assets%2F7_apart_id.png)

This how I specified 'apart_id' as a Data Frame Index Column using Pandas **_set_index()_** method:
```
data_frame.set_index('apart_id', inplace=True)
```
Now 'apart_id' became an index column!  
![8_inxed_column.png](assets%2F8_inxed_column.png)

# Descriptive Statistics & Normal Distribution

After DataFrame is cleaned & transformed It is now ready to be analyzed🔎  
In this section I will apply Descriptive Statistics that is a simple and powerful way to obtain some
interesting and meaningful insights from the data!  
I will also build a **Histogram**📶 to check if prices are in **'Normal Distribution'** 
form using **Matplotlib** library!

 


### 1. Descriptive Statistics📊
I will use Pandas _**groupby()**_ and **_agg()_** methods to apply descriptive
statistics tool on our data such as _**min, max, mean, median and standard deviation**_ functions! 

```
print(data_frame.groupby(by=['City', 'District'])['Rent_Price'].agg(['min','max', 'mean', 'median', 'std']))
```
The **_agg()_** method allows to use these multiple statistical aggregation functions on a rental price column,
and then I am grouping results of these functions by 'City' and 'District' columns using **_groupby()_** method!  
Here is what results looks like:  
![9_descreptive_statistics.png](assets%2F9_descreptive_statistics.png)

### 2. Normal Distribution of a Rental Price💲

Checking the **Distribution** of a rental price can help find out whether there are abnormal outliers 
which can seriously effect _**mean**_, _**median**_ and other descriptive statistic values of data 
(even though median less sensitive to outliers than mean).  

If prices are in **_Normal Distribution_** form it means that data have no significant 
outliers, **mean** and **median** are approximately the same, and we also can say that
68% of the data falls within one standard deviation (σ) of the mean, 
95% within two standard deviations, and 99.7% within three standard deviations (It's called **'Empirical rule'**).

To check it I built up a **histogram** using **_Matplotlib_** library:
```
plt.figure(figsize=(8, 6))
plt.hist(data_frame['Rent_Price'], bins=30)
plt.title('Distribution of Rent Price')
plt.xlabel('Rent Price')
plt.ylabel('Frequency')
plt.show()
```
![10_histogram.png](assets%2F10_histogram.png)

As we can see in the histogram the data is, in general, following a bell curve and didn't have significant outliers,  
It's now safe to say that the apartment rental prices are in **_'Normal Distribution'_** form!

# Correlation & Regression Analysis

Now it's time for **Regression Analysis** and predicting apartment rental price based on its **Correlation** 
with the apartment size!

###  1. Correlation check & Building a Scatter Plot📈

The good simple way to see if there is **Correlation** between values is to build a **Scatter plot** using Matplotlib library:
```
plt.figure(figsize=(8, 6))
plt.scatter(data_frame['Apartment_Size'], data_frame['Rent_Price'])
slope, intercept, rvalue, pvalue, stderr = stats.linregress(data_frame['Apartment_Size'],data_frame['Rent_Price'])
plt.plot(data_frame['Apartment_Size'], slope*data_frame['Apartment_Size']+intercept, color='red')
plt.xlabel('Apartment Size')
plt.ylabel('Rent Price')
plt.show()
```
![11_scatter_plot.png](assets%2F11_scatter_plot.png)

As you can see on the scatter plot there is definitely a clear correlation between apartment size and its price (which is quite obvious).  
We can also check the exact correlation value by using Pandas **_corr()_** method:
```
print(data_frame['Apartment_Size'].corr(data_frame['Rent_Price']))
```
As a result, we get a correlation value of ~**0.7** which is a really nice correlation value that 
indicates a strong relationship between these variables!


### 2. Regression Analysis🔎

# Conclusion