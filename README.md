# Amazon EDA (Exploratory Data Analysis)
Exploring the fundamentals of Machine Learning starts with importing and preparing data, then using visualization and Exploratory Data Analysis (EDA) to understand patterns, distributions, and relationships. This foundation ensures the data is ready for effective model building.


Performing with existing Blog, Refer : [Amazon EDA Dataset](https://www.kaggle.com/code/mehakiftikhar/amazon-sales-dataset-eda#Amazon-Sales-Dataset-EDA)

# Packages used in the program 
1. **Pandas:** Data manipulation and analysis
2. **Numpy:** Numerical operations and calculations
3. **Matplotlib:** Data visualization and plotting
4. **Seaborn:** Enhanced data visualization and statistical graphics
6. **Scikit-learn:** For predective data analysis
5. **Scipy:** Scientific computing and advanced mathematical operations



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

Plot ```actual_price``` vs. ```rating ```
```python
plt.scatter(df['actual_price'], df['rating'])
plt.xlabel('Actual_price')
plt.ylabel('Rating')
plt.show()
```


A correlation heatmap is a visual graphic that shows how each variable in the dataset are correlated to one another.\
-1 signifies zero correlation, while 1 signifies a perfect correlation.
For more information on correlation read [Correlation Heatmap](https://medium.com/5-minute-eda/5-minute-eda-correlation-heatmap-b57bbb7bae14)

# Feature Engineering

Feature engineering is the process of transforming the raw data into relevant information for use by machine learning models. To put it in other words, feature engineering is the process of **creating predictive model features**. Because model performance largely rests on the quality of data used during training, feature engineering is a crucial pre-processing technique that requires selecting the most relevant aspects of raw training data for both the predictive task and model type under consideration. To understand about feature engineering first we need to understand the types of features that are present in machine learning.
1. **Numerical Features**
    - As the name indicates, numerical features are those that are representing measurable quantities. These features can be continous or discrete.
    - Few examples of numerical features are age, height, mobile number etc.
2. **Categorical Features**
    - Categorical features are those that represnet the data which can be placed in to a category. These features may contain nominal or ordinal data.
    - Some known examples of categorical features are gender, country, month of birth, eye color etc.
3. **Text Features**
    - Text features are those that contain string data. These features can be further classified into unstructured and structured text data.
    - Some examples of text features are customer reviews, comments, tweets etc.
4. **Time-series Features**
    - Time-series features are those data instances that are collected over a time duration. These features are time-dependent are are mostly used in forecasting problems or trend analysis.

Although there are many feature engineering techniqies, there isn't one method universally accepted and used. The selection of feature engineering method solely depends of the dataset and the problem statement to be solved.\
Here are some of the common techniques used in feature engineering.

## One-Hot Encoding
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

Before going furter please refer 
[Corelation and covarience](https://youtu.be/uW0TapQ6UQU) from YouTube

Take out from above video :\
Even when the covarience seems have less differnce but actual corelation of variables can very low that it will not be effective in model.

# Linear discriminant analysis (LDA)
LDA is ostensibly similar to PCA in that it projects model data onto a new, lower dimensional space. While PCA produces new component variables meant to maximize data variance, LDA produces component variables primarily intended to maximize class difference in the data.


## Principal Component Analysis (PCA)
* Unsupervised
* Maximize variance
* Ignores class labels
* Dimensionality reduction, exploratory analysis

## Linear Discriminant Analysis (LDA)
* Supervised
* Class Labels
* Uses class labels
* Classification, feature selection for classification

# Feature scaling
Certain features have upper and lower bounds intrinsic to data that limits possible feature values, such as time-series data or age. While feature transformation transforms data from one type to another, feature scaling transforms data in terms of range and distribution, maintaining its original data type. Feature scaling is essential when features have different units or scales, as it ensures that no single feature dominates the learning process due to its larger values. Feature scaling is done using two primary techniques: **min-max scaling** and **z-score scaling**.

## Min-max scaling.
Min-max scaling rescales all values for a given feature so that they fall between specified minimum and maximum values, often 0 and 1. Min-max scaling is calculated using the formula
$$
\tilde{x} = \frac{x - \min(x)}{\max(x) - \min(x)}
$$

## Z-score scaling
Literature also refers to this as standardization and variance scaling. Whereas min-max scaling scales feature values to fit within designated minimum and maximum values, z-score scaling rescales features so that they have a shared standard deviation of 1 with a mean of 0.\
 Z-score scaling is represented by the following formula
$$
\tilde{x} = \frac{x - mean(x)}{\sqrt(var(x))}
$$

# Feature selection

Feature selection is the process of selecting the most relevant features of a dataset to use when building and training a machine learning model. By reducing the feature space to a selected subset, feature selection improves AI model performance while lowering its computational demands.

A "feature" refers to an individual measurable property or characteristic of a data point: a specific attribute of the data that helps describe the phenomenon being observed. A dataset about housing might have features such as “number of bedrooms” and “year of construction.” 

Feature selection is part of the feature engineering process, in which data scientists prepare data and curate a feature set for machine learning algorithms. Feature selection is the portion of feature engineering concerned with choosing the features to use for the model.

## Supervised Feature Selection Methods
Supervised feature selection methods are applied to labeled datasets, where datapoints have known target values. These methods are designed to identify features that have the strongest relationships with the target variable, thereby enhancing supervied learning models sush as classification and regression.

## Filter methods
Filter methods are a group of feature selection techniques that are solely concerned with the data itself and do not directly consider model performance optimization. Input variables are assessed independently against the target variable to determine which has the highest correlation.

### Information gain
Information gain shows how important the presence or absence of a feature is in determining the target variable by the degree of entropy reduction. To put it simply information gain helps us understand how much a particular feature contributes to making accurate predictions in a decision tree. Features with higher Information Gain are considered more informative and are preferred for splitting the dataset, as they lead to nodes with more homogenous classes.

```python
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
%matplotlib inline

importances = mutual_info_classif(X, Y)
feat_importances = pd.Series(importances, dataframe.columns[0:len(dataframe.columns)-1])
feat_importances.plot(kind='barh', color = 'teal')
plt.show()
```
Output
```md
Information Gain for each feature: [0.50192633 0.27889121 0.9924725  0.98664421]
```
### Mutual information
Mutual informaton assess the dependence between variables by measuring the information obtained about one through the other.
It is a **non-negative value** that indicates the degree of dependence between the variables: the higher the MI, the greater the dependence.
```python
from sklearn.feature_selection import mutual_info_regression
import numpy as np

# Generate sample data
np.random.seed(0)
X = np.random.rand(100, 2)
y = X[:, 0] + np.sin(6 * np.pi * X[:, 1])

# Calculate Mutual Information using mutual_info_regression
mutual_info = mutual_info_regression(X, y)
print("Mutual Information for each feature:", mutual_info)
```
Output
```md
Mutual Information for each feature: [0.42283584 0.54090791]
```

### Chi-square test
Assesses the relationship between two categorical variables by comparing observed to expected values. Chi-squared test, or χ² test, helps in determining whether these two variables are associated with each other. To aplly chi-square test the data must be in categorical format, the observations must be independent, the sample must be randomly selected and the expected frequency for each category should be at least 5.

```python
from scipy.stats import chi2_contingency

# defining the table
data = [[207, 282, 241], [234, 242, 232]]
stat, p, dof, expected = chi2_contingency(data)

# interpret p-value
alpha = 0.05
print("p value is " + str(p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')
```
```md
p value is 0.1031971404730939
Independent (H0 holds true)
```


### Fisher’s score
Fischer's socre uses derivatives to calculate the relative importance of each feature for classifying data. A higher score indicates greater influence. It works by comparing how much a feature varies between different classes versus how much it varies within the same class. Features that show big differences between classes, but are consistent within each class, are considered useful for classification
```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import load_iris
import pandas as pd

# Load dataset
data = load_iris()
X, y = data.data, data.target
feature_names = data.feature_names

# Compute Fisher scores
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X, y)
scores = selector.scores_

# Display results
fisher_scores = pd.DataFrame({'Feature': feature_names, 'Fisher Score': scores})
print(fisher_scores.sort_values(by='Fisher Score', ascending=False))
```
```md
             Feature  Fisher Score
2  petal length (cm)   1180.161182
3   petal width (cm)    960.007147
0  sepal length (cm)    119.264502
1   sepal width (cm)     49.160040
```


### Pearson’s correlation coefficient (PCC)
Correlation is the measure of linear relationship between 2 or more variables. Quantifies the relationship between two continuous variables with a score **ranging from -1 to 1.**

### Variance threshold
Removes all features that fall under a minimum degree of variance because features with more variances are likely to contain more useful information. A related method is the mean absolute difference (MAD). 
```pyton
from sklearn.feature_selection import VarianceThreshold
import numpy as np

# Sample dataset: 5 samples, 4 features
X = np.array([
    [0, 2, 0, 3],
    [0, 1, 4, 3],
    [0, 1, 1, 3],
    [0, 1, 0, 3],
    [0, 1, 3, 3] ])

# Initialize VarianceThreshold
selector = VarianceThreshold()

# Fit and transform the data
X_sele = selector.fit_transform(X)

print("Original shape:", X.shape)
print("Reduced shape:", X_sele.shape)
print(X_sele)
```
```md
Original shape: (5, 4)
Reduced shape: (5, 2)
[[2 0]
 [1 4]
 [1 1]
 [1 0]
 [1 3]]
```

# Pipeline

An ML pipeline automates and standardizes a series of steps in the machine learning workflow, from data collection and preprocessing to model training, deployment, and monitoring, creating a repeatable and scalable process

```python
from sklearn.pipeline import make_pipeline

model = make_pipeline(SimpleImputer(strategy='mean'),
                      PolynomialFeatures(degree=2),
                      LinearRegression())
```
