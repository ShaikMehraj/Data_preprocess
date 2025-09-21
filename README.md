# Amazon EDA (Exploratory Data Analysis)
Exploring the fundamentals of Machine Learning starts with importing and preparing data, then using visualization and Exploratory Data Analysis (EDA) to understand patterns, distributions, and relationships. This foundation ensures the data is ready for effective model building.


Performing with existing Blog ; Refer : [Kaggel](https://www.kaggle.com/code/mehakiftikhar/amazon-sales-dataset-eda#Amazon-Sales-Dataset-EDA)

# Packages used in the program : 
1. **Pandas:** Data manipulation and analysis
2. **Numpy:** Numerical operations and calculations
3. **Matplotlib:** Data visualization and plotting
4. **Seaborn:** Enhanced data visualization and statistical graphics
6. **Scikit-learn:** For predective data analysis
5. **Scipy:** Scientific computing and advanced mathematical operations



To feed data to a model, it is convenient to use a __DataFrame__, which stores data in a tabular format and allows easy modification and manipulation in memory, avoiding the need to repeatedly import the file.

For reading file the code below can be used : 

```python
df = pd.read_csv(file path)
df = pd.read_excel(file path)
```

Depending on file format ```pd.read_(file_type)``` can be changed, for further details related to file formats please refer to [IO tools](https://pandas.pydata.org/docs/user_guide/io.html)


```df.info()``` provides a summary of the DataFrame, including column names, their data types, the number of non-null values, and memory usage.

```df.isnull().sum()``` returns the number of missing (null) values in each column of the DataFrame.


# Data cleaning and data converstion :

Changing the data type of discounted price
```python
df['discounted_price'] = df['discounted_price'].str.replace("₹", '')   # remove ₹ symbol
df['discounted_price'] = df['discounted_price'].str.replace(",", '')  # remove commas
df['discounted_price'] = df['discounted_price'].astype('float64')     # convert to float
```

For example \
Input : "₹1,299"  output :  1299.0


Finding un-usual string in rating column

```python
df['rating'].value_counts()
```

```df.describe()``` By default, it summarizes only numeric columns.

**It gives statistics like:**
* **count** → number of non-null values
* **mean** → average
* **std** → standard deviation
* **min** → minimum value
* **25%** → 1st quartile (25th percentile)
* **50%** → median (50th percentile)
* **75%** → 3rd quartile (75th percentile)
* **max** → maximum value


```python
df.isnull().sum().sort_values(ascending = False)
```  
This line counts the missing values in each column of the DataFrame and sorts the columns from most to least missing values.


```python
# Filling missing values with median value
df['rating_count'] = df.rating_count.fillna(value=df['rating_count'].median())
```
For more explanation and methods refer [Handling missing values in dataset](https://medium.com/@pingsubhak/handling-missing-values-in-dataset-7-methods-that-you-need-to-know-5067d4e32b62)

```python
# Find Duplicate 
df.duplicated().any()
```
This line checks if the DataFrame contains any duplicate rows and returns True if there is at least one duplicate, otherwise False.


# Data vizulalization 

Plot actual_price vs. rating 
```python
plt.scatter(df['actual_price'], df['rating'])
plt.xlabel('Actual_price')
plt.ylabel('Rating')
plt.show()
```


A correlation heatmap is a visual graphic that shows how each variable in the dataset are correlated to one another. -1 signifies zero correlation, while 1 signifies a perfect correlation.
For more information on correlation read [Correlation Heatmap](https://medium.com/5-minute-eda/5-minute-eda-correlation-heatmap-b57bbb7bae14)


#LabelEncoder
```python
le_category = LabelEncoder()
```
Converts labels to numbers\
ex:\
Electronics → 0  
Clothing    → 1  
Books       → 2

```python
# Calculate mean sales by product category
grouped_df = df.groupby('category')['rating'].mean()
Print mean sales by product category
print(grouped_df)
```
For each category group, it calculates the average (mean) of the rating column.
So you get the average product rating per category.

```grouped_df``` will be a Series where:
* Index = product category
* Value = average rating of that category


The Chi-Square (χ²) Test is a statistical test used to check if there is a relationship between two categorical variables.
The test gives you a Chi-Square statistic and a p-value:

p < 0.05 → significant relationship (variables are dependent).

p ≥ 0.05 → no significant relationship (variables are independent).


Inverse transform the data
This reverses the encoding and converts the numbers back into their original string labels.
