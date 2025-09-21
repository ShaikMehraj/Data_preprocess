# Amazon_EDA
Exploring the fundamentals of Machine Learning starts with importing and preparing data, then using visualization and Exploratory Data Analysis (EDA) to understand patterns, distributions, and relationships. This foundation ensures the data is ready for effective model building.


Performing with existing Blog ; Refer : [Kaggel](https://www.kaggle.com/code/mehakiftikhar/amazon-sales-dataset-eda#Amazon-Sales-Dataset-EDA)

Packages will be using : 
1. Pandas: Data manipulation and analysis
2. Numpy: Numerical operations and calculations
3. Matplotlib: Data visualization and plotting
4. Seaborn: Enhanced data visualization and statistical graphics
5. Scipy: Scientific computing and advanced mathematical operations
6. Scikit-learn : For predective data analysis



For feeding data to a model, it is convenient to use a DataFrame, which stores data in a tabular format and allows easy modification and manipulation in memory, avoiding the need to repeatedly import the file.

for reading file code to be used : 

df = pd.read_csv(file path)
df = pd.read_excel(file path)


Depending on file format can be chosse the extebtion by [link](https://pandas.pydata.org/docs/user_guide/io.html)


df.info() provides a summary of the DataFrame, including column names, their data types, the number of non-null values, and memory usage.


df.isnull().sum() returns the number of missing (null) values in each column of the DataFrame.


# Data cleaning and data converstion :)


Changing the data type of discounted price

df['discounted_price'] = df['discounted_price'].str.replace("₹", '')   # remove ₹ symbol

df['discounted_price'] = df['discounted_price'].str.replace(",", '')  # remove commas

df['discounted_price'] = df['discounted_price'].astype('float64')     # convert to float

Input : "₹1,299"  output :  1299.0


Finding unusual string in rating column
df['rating'].value_counts()


df.describe()
By default, it summarizes only numeric columns.

It gives statistics like:

count → number of non-null values

mean → average

std → standard deviation

min → minimum value

25% → 1st quartile (25th percentile)

50% → median (50th percentile)

75% → 3rd quartile (75th percentile)

max → maximum value


df.isnull().sum().sort_values(ascending = False) , This line counts the missing values in each column of the DataFrame and sorts the columns from most to least missing values.



# Filling missing values with median value
df['rating_count'] = df.rating_count.fillna(value=df['rating_count'].median())
For more explanation and methods refer [LINK](https://medium.com/@pingsubhak/handling-missing-values-in-dataset-7-methods-that-you-need-to-know-5067d4e32b62)


# Find Duplicate 
df.duplicated().any()
This line checks if the DataFrame contains any duplicate rows and returns True if there is at least one duplicate, otherwise False.
