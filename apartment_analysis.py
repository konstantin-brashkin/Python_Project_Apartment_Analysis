# Importing all necessary modules & libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats


# Creating Pandas DataFrame from CSV file
data_frame = pd.read_csv('german_apartment_rentals.csv')


# Checking first 5 rows to see the Dataset structure
print(data_frame.head())


# Checking info about Pandas data frame
print(data_frame.info())


# Converting string to datetime type
data_frame['Date_Built'] = pd.to_datetime(data_frame['Date_Built'])


# Converting ints to boolean values
data_frame['Has_Balcony'] = data_frame['Has_Balcony'].astype(bool)
data_frame['Has_Elevator'] = data_frame['Has_Elevator'].astype(bool)
data_frame['Has_Parking'] = data_frame['Has_Parking'].astype(bool)


# Setting up 'apart_id' as an index column
data_frame.set_index('apart_id', inplace=True)


# Using "describe" Pandas method to see a quick summary & descriptive statistic
# for all numeric columns
print(data_frame.describe())


# Creating a Histogram chart using Matplotlib library to check a Distribution of prices,
# in that case we can see that prices are in 'Normal Distribution' form
plt.figure(figsize=(8, 6))
plt.hist(data_frame['Rent_Price'], bins=30)
plt.title('Distribution of Rent Price')
plt.xlabel('Rent Price')
plt.ylabel('Frequency')
plt.show()


# Checking if rental price has a strong correlation with the size of apartment
print(data_frame['Apartment_Size'].corr(data_frame['Rent_Price']))


# Building a Scatter plot to graphically display
# a correlation between the rental price and the size of apartment
plt.figure(figsize=(8, 6))
plt.scatter(data_frame['Apartment_Size'], data_frame['Rent_Price'])

# Adding a red trend line in a scatter plot
slope, intercept, rvalue, pvalue, stderr = stats.linregress(data_frame['Apartment_Size'],data_frame['Rent_Price'])
plt.plot(data_frame['Apartment_Size'], slope*data_frame['Apartment_Size']+intercept, color='red')

plt.xlabel('Apartment Size')
plt.ylabel('Rent Price')
plt.show()


# Regression Analysis to predict apartments prices based on their sizes

# Setting up independent (X) and dependent (Y) variables
x = data_frame['Apartment_Size']
x = sm.add_constant(x)
y = data_frame['Rent_Price']


# Defining the sizes of apartments for which I want to predict the price
new_sizes = [130, 160, 200]
new_sizes = sm.add_constant(new_sizes)


# Creating and training a model
model = sm.OLS(y, x).fit()


# Checking regression model summary (p-value etc.) & getting the final prediction results with confidence intervals
prediction = model.get_prediction(new_sizes)
summary = prediction.summary_frame(alpha=0.05)

print(model.summary())
print(summary)




