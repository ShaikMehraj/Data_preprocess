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

```python
import pandas as pd

data = {'Color': ['Red', 'Blue', 'Green', 'Blue', 'Orange']}
df = pd.DataFrame(data)

df_encoded = pd.get_dummies(df, columns=['Color'], prefix='color')

print(df_encoded)
```
```
   color_Blue    color_Green  color_Orange   color_Red
0       False        False         False       True
1        True        False         False      False
2       False         True         False      False
3        True        False         False      False
4       False        False          True      False
```

### Disadvantages of One-Hot Encoding: 
* This Significantly increases the dimensionality of the dataset, especially with many unique categories. 
* Using One-How Encoding can lead to increased memory consumption and slower training times.

To know more about one-hot encoding method please do read [What Is One-Hot Encoding](https://www.datacamp.com/tutorial/one-hot-encoding-python-tutorial)

## Binning

Binning is the process of grouping or categorising continuous data into smaller, discrete sets called **bins** or **buckets**. This technique is widely used in data mining and machine learning to convert continuous variables into categorical ones, such as turning age into **"age ranges"**. Binning can be applied to both numerical and categorical variables, and its primary purpose is to simplify the data and make it more manageable for analysis.

```python
import pandas as pd

data = {'Age': [23.5, 45, 18, 34, 67, 50, 21]}
df = pd.DataFrame(data)

bins = [0, 20, 40, 60, 100]
labels = ['0-20', '21-40', '41-60', '61+']

df['Age_Range'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

print(df)
```
```
    Age Age_Range
0  23.5     21-40
1  45.0     41-60
2  18.0      0-20
3  34.0     21-40
4  67.0       61+
5  50.0     41-60
6  21.0     21-40
```

To get a good idea on binning read the article [Binning in Data mining](https://www.scaler.com/topics/binning-in-data-mining/)


## Feature Splitting:
Feature spiltting is the process of beraking down a complex single feature into multiple simpler features.
```python
import pandas as pd

data = {'Full_Address': [
    '74 st, Oklahoma, 77724', '456 Oak Rd, Tennessee , 442640']}
df = pd.DataFrame(data)

df[['Street', 'City', 'Zipcode']] = df['Full_Address'].str.extract(
    r'([0-9]+\s[\w\s]+),\s([\w\s]+),\s(\d+)')

print(df)
```
```
                    Full_Address      Street        City   Zipcode
0          74 st, Oklahoma, 77724       74 st    Oklahoma   77724
1  456 Oak Rd, Tennessee , 442640  456 Oak Rd  Tennessee   442640
```