# Exploratory Data Analysis
Exploring the fundamentals of Machine Learning starts with importing and preparing data, then using visualization and Exploratory Data Analysis (EDA) to understand patterns, distributions, and relationships. This foundation ensures the data is ready for effective model building.


Performing with existing Blog, Refer : [Amazon EDA Dataset](https://www.kaggle.com/code/mehakiftikhar/amazon-sales-dataset-eda#Amazon-Sales-Dataset-EDA)

# Packages used in the program 
1. **Pandas:** Data manipulation and analysis
2. **Numpy:** Numerical operations and calculations
3. **Matplotlib:** Data visualization and plotting
4. **Seaborn:** Enhanced data visualization and statistical graphics
6. **Scikit-learn:** For predective data analysis
5. **Scipy:** Scientific computing and advanced mathematical operations
6. **scikit-learn** The sklearn library contains a lot of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction.



To feed data to a model, it is convenient to use a __DataFrame__, which stores data in a tabular format and allows easy modification and manipulation in memory, eliminating the need to repeatedly import the file.

For reading file the code below can be used 

```python
df = pd.read_csv(file path)
df = pd.read_excel(file path)
```

Depending on file format ```pd.read_(file_type)``` can be changed, for details related to file formats please refer to [IO tools](https://pandas.pydata.org/docs/user_guide/io.html)


```df.info()``` provides a summary of the DataFrame, including column names, their data types, the number of non-null values, and memory usage.

```df.isnull().sum()``` returns the number of missing (null) values in each column of the DataFrame.


# Data cleaning and data converstion

Changing the data type of discounted price
```python
df['discounted_price'] = df['discounted_price'].str.replace("₹", '')   # remove ₹ symbol
df['discounted_price'] = df['discounted_price'].str.replace(",", '')  # remove commas
df['discounted_price'] = df['discounted_price'].astype('float64')     # convert to float
```

For example\
Input : "₹1,299"\
output :  1299.0

concating the Date of birth columns
```python
df['DOB'] = df.apply(lambda x:'%s-%s-%s' % (x['dob_day'],x['dob_month'],x['dob_year']),axis=1)
df['DOB'] = pd.to_datetime(df['DOB']) # Changing the data type
```

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
for x in df.index:
  if df.loc[x, "age"] > 80:
    df.drop(x, inplace = True)
```
Above line of code will remove the or delete the rows which has certain values , such as outliers 

concating the Date of birth columns
```python
df['DOB'] = df.apply(lambda x:'%s-%s-%s' % (x['dob_day'],x['dob_month'],x['dob_year']),axis=1)
df['DOB'] = pd.to_datetime(df['DOB']) # Changing the data type
```

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

Plot ```actual_price``` vs. ```rating ```
```python
plt.scatter(df['actual_price'], df['rating'])
plt.xlabel('Actual_price')
plt.ylabel('Rating')
plt.show()
```


# Feature Engineering

Label Encoder ```le_category = LabelEncoder()``` Converts labels to numbers\
ex:\
Electronics → 0  
Clothing    → 1  
Books       →D 2

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

Inverse transform the data
This reverses the encoding and converts the numbers back into their original string labels.


# One-Hot Encoding
While One-Hot Encoding creates a new binary column for each category, best for nominal data (where order doesn't matter), allows the model to learn separate weights for each category, leading to more nuanced decisions.

Cons: 
* Significantly increases the dimensionality of the dataset, especially with many unique categories. 
* Can lead to increased memory consumption and slower training times. 

To know more about one-hot encoding please read [Encoding Categorical Variables](https://towardsdatascience.com/encoding-categorical-variables-one-hot-vs-dummy-encoding-6d5b9c46e2db/)

Take out on above article : 
* One-Hot Encoding is best for nominal categorical data.
* Label encoding is best for Ordinal categorical data.
* Dummies encoding is best for when need to keep lighter and removes a duplicate category in each categorical variable.


With pandas package we can implement dummies or one hot encoding , ```python drop_first=True ``` is for Dummies , ```python drop_first=False ``` is for One hot encoding.
```python
dummy_df = pd.get_dummies(df, prefix={'gender':'gender'},drop_first=True, dtype=int)
```

# Binning

Binning is the process of grouping or categorizing continuous data into smaller, discrete sets called **bins** or **buckets**. This technique is widely used in data mining and machine learning to convert continuous variables into categorical ones, such as turning age into **"age ranges"**. Binning can be applied to both numerical and categorical variables, and its primary purpose is to simplify the data and make it more manageable for analysis. \
To get a good idea on binning read the article [Binning in Data mining](https://www.scaler.com/topics/binning-in-data-mining/)

# Feature scaling
Feature scaling ensures that numerical features lie within a standardized range, preventing some features from dominating the learning process due to their larger values.



# Normilazation
normilazation refers to the process of adjusting values measured on different scales to a common scale. 

## Types of normilazation
1. **Min-Max normilazation:**
With min-max normilazation, we might rescale the sizes of the houses to fit within a range of 0 to 1.is useful when you want to preserve the relative size of the values while simplifying the data.

1. **Log normilazation:**
Log normilazation is another normilazation technique. By using log normilazation, we apply a logarithmic transformation to the range of values.However, it may not work well with negative or zero values.

1. **Decimal scaling:**
is useful when you want to preserve the relative size of the values while simplifying the data.
Z-score normilazation is useful when you want to compare data points across different datasets or when you want to identify outliers. 

1. **Mean normilazation (mean-centering):**
Mean normilazation, in this context, would involve adjusting the house prices by subtracting the average price from each range value.

## What is Standardization?
While normilazation scales features to a specific range, standardization, which is also called **z-score scaling**, transforms data to have a mean of 0 and a standard deviation of 1. This process adjusts the feature values by subtracting the mean and dividing by the standard deviation. You might have heard of ‘centering and scaling’ data. Well, standardization refers to the same thing: first centering, then scaling.

**Gradient-based Algorithms:** Support Vector Machine (SVM) requires standardized data for optimal performance. While models like linear regression and logistic regression do not assume standardization

**Dimensionality Reduction:** Standardization is in dimensionality reduction techniques like PCA because PCA identifies the direction where the variance in the data is maximized. Mean normilazation alone is not sufficient because PCA considers both the mean and variance, and different feature scales would distort the analysis.

Use cases and [differnce](https://www.datacamp.com/tutorial/normilazation-vs-standardization) 
normilazation is widely used in distance-based algorithms like k-Nearest Neighbors (k-NN), where features must be on the same scale to ensure accuracy in distance calculations. 
Standardization, on the other hand, is vital for gradient-based algorithms such as Support Vector Machines (SVM) and is frequently applied in dimensionality reduction techniques like PCA, where maintaining the correct feature variance is important.

## My observation or take out:
normilazation is best when we work with single Feature , along with removing the outliers.(Better works with Image normilazations)
Standardization is best when work with multiple features and we need to scale them and applie PCA for dimentsionalty reduction .(they are less effected by outliers, but it will effect beacuse they do not represt original values)

# Feature Selection Techniques
If we have multiple fetures to train model , but we have large data set , data set is not healping train model , it will make model more effected as outlier . so thus we use Feature selection techniquies to redues the dimenstionality ,and have selected features to train model.

Refer : [Feature Selection Techniques in Machine Learning](https://www.geeksforgeeks.org/machine-learning/feature-selection-techniques-in-machine-learning/) 

## Correlation Matrix Heatmap
A correlation heatmap is a visual graphic that shows how each variable in the dataset are correlated to one another.\
-1 signifies zero correlation, while 1 signifies a perfect correlation.\
For more information on correlation read [Correlation Heatmap](https://medium.com/5-minute-eda/5-minute-eda-correlation-heatmap-b57bbb7bae14)

In order to plot , we need to convert the all the features into numeric 

```python
plt.figure(figsize=(12, 12)) 
correlation_matrix = dummy_df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()
```


The Chi-Square (χ²) Test is a statistical test used to check if there is a relationship between two categorical variables.
The test gives you a Chi-Square statistic and a p-value:
* p < 0.05 → significant relationship (variables are dependent).
* p ≥ 0.05 → no significant relationship (variables are independent).
  
This score can be used to select the n_features features with the highest values for the test chi-squared statistic from X, which must contain only non-negative integer feature values such as booleans or frequencies (e.g., term counts in document classification), relative to the classes.

# PCA
Principal component analysis is a common feature extraction method that combines and transforms a dataset’s original features to produce new features, called principal components. PCA is very effective for visualizing and exploring high-dimensional datasets, or data with many features, as it can easily identify trends, patterns, or outliers.PCA is a dimension reduction technique like linear discriminant analysis. In contrast to LDA, PCA is not limited to supervised learning tasks. For unsupervised learning tasks, this means PCA can reduce dimensions without having to consider class labels or categories. PCA is also closely related to factor analysis. They both reduce the number of dimensions or variables in a dataset while minimizing information loss.

Corelation and covarience : https://www.youtube.com/watch?v=uW0TapQ6UQU

take out from above video : even when the covarience seems have less differnce but actual corelation of variables can very low that it will not be effective in model.

# Linear discriminant analysis 
LDA is ostensibly similar to PCA in that it projects model data onto a new, lower dimensional space. While PCA produces new component variables meant to maximize data variance, LDA produces component variables primarily intended to maximize class difference in the data.


Principal Component Analysis (PCA)
Unsupervised
Maximize variance
Ignores class labels
Dimensionality reduction, exploratory analysis

Linear Discriminant Analysis (LDA)
Supervised
Class Labels
Uses class labels
Classification, feature selection for classification

# Pipeline

An ML pipeline automates and standardizes a series of steps in the machine learning workflow, from data collection and preprocessing to model training, deployment, and monitoring, creating a repeatable and scalable process

```python
from sklearn.pipeline import make_pipeline

model = make_pipeline(SimpleImputer(strategy='mean'),
                      PolynomialFeatures(degree=2),
                      LinearRegression())
```
