# Amazon_EDA
Exploring the fundamentals of Machine Learning starts with importing and preparing data, then using visualization and Exploratory Data Analysis (EDA) to understand patterns, distributions, and relationships. This foundation ensures the data is ready for effective model building.


Performing with existing Blog ; Refer : [Kaggel](https://www.kaggle.com/code/mehakiftikhar/amazon-sales-dataset-eda#Amazon-Sales-Dataset-EDA)

Packages will be using : 
1. Pandas: Data manipulation and analysis
2. Numpy: Numerical operations and calculations
3. Matplotlib: Data visualization and plotting
4. Seaborn: Enhanced data visualization and statistical graphics
5. Scipy: Scientific computing and advanced mathematical operations

For feeding data to a model, it is convenient to use a DataFrame, which stores data in a tabular format and allows easy modification and manipulation in memory, avoiding the need to repeatedly import the file.

for reading file code to be used : 
df = pd.read_csv(file path)
df = pd.read_excel(file path)

depending on file format can be chosse the extebtion by [link](https://pandas.pydata.org/docs/user_guide/io.html)

df.info() provides a summary of the DataFrame, including column names, their data types, the number of non-null values, and memory usage.

out put : 
RangeIndex: 1465 entries, 0 to 1464
Data columns (total 16 columns):
 #   Column               Non-Null Count  Dtype 
---  ------               --------------  ----- 
 0   product_id           1465 non-null   object
 1   product_name         1465 non-null   object
 2   category             1465 non-null   object
 3   discounted_price     1465 non-null   object
 4   actual_price         1465 non-null   object
 5   discount_percentage  1465 non-null   object
 6   rating               1465 non-null   object
 7   rating_count         1463 non-null   object
 8   about_product        1465 non-null   object
 9   user_id              1465 non-null   object
 10  user_name            1465 non-null   object
 11  review_id            1465 non-null   object
 12  review_title         1465 non-null   object
 13  review_content       1465 non-null   object
 14  img_link             1465 non-null   object
 15  product_link         1465 non-null   object
dtypes: object(16)

df.isnull().sum() returns the number of missing (null) values in each column of the DataFrame.

data cleaning and data converstion :
# Changing the data type of discounted price
df['discounted_price'] = df['discounted_price'].str.replace("₹", '')   # remove ₹ symbol
df['discounted_price'] = df['discounted_price'].str.replace(",", '')  # remove commas
df['discounted_price'] = df['discounted_price'].astype('float64')     # convert to float

Input : "₹1,299"  output :  1299.0


Finding unusual string in rating column
df['rating'].value_counts()
