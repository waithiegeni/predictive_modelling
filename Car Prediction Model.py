# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:16:40 2019

@author: X260
"""

#import pandas library
import pandas as pd

#read in auto data
df = pd.read_csv('~/Downloads/Auto.csv', header = None, encoding = 'utf-8')

#create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

# replace headers 
df.columns = headers 

# replace "?" to NaN
import numpy as np

df.replace("?", np.nan, inplace = True)

# Evaluate missing data
missing_data = df.isnull()

for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")  

# Calculate the average of the normalized losses
avg_norm_loss = df["normalized-losses"].astype('float').mean()

#replace nan with average
df['normalized-losses'].replace(np.nan, avg_norm_loss, inplace = True)

#Calculate the mean value for 'bore' column
avg_bore=df['bore'].astype('float').mean()

#replace nan with average
df['bore'].replace(np.nan, avg_bore, inplace = True)

#Calculate the mean value for 'stroke' column
avg_stroke = df['stroke'].astype('float').mean()

#replace nan with average
df['stroke'].replace(np.nan,avg_stroke, inplace = True)

# Calculate the average of horsepower
avg_horsepower = df['horsepower'].astype('float').mean()

#replace nan with average
df['horsepower'].replace(np.nan, avg_horsepower, inplace = True)

#Calculate the mean value for 'peak-rpm' column
avg_peakrpm=df['peak-rpm'].astype('float').mean()

#replace nan with average
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace = True)

#count highest frequency of number of doors
df['num-of-doors'].value_counts().idxmax()

#replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, "four", inplace=True)

# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)

# Check data types
df.dtypes

#Convert data types to proper format
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")


# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]


df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()

#bin horsepower
df["horsepower"]=df["horsepower"].astype('int')


%matplotlib inline
import matplotlib.pyplot as plt

plt.hist(df["horsepower"])
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")

# create bins
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)

#Set group names
group_names = ['Low', 'Medium', 'High']


df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )


df["horsepower-binned"].value_counts()


plt.bar(group_names, df["horsepower-binned"].value_counts())
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")

plt.hist(df["horsepower"], bins = 3)
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")

dummy_variable = pd.get_dummies(df["fuel-type"])

# merge data frame "df" and "dummy_variable" 
df = pd.concat([df, dummy_variable], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)

import seaborn as sns

# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)

#check correlation
df[["engine-size", "price"]].corr()

# highway mpg as potential predictor variable of price
sns.regplot(x="highway-mpg", y="price", data=df)

#Check correlation
df[['highway-mpg', 'price']].corr()

# Peak rpm as potential predictor variable of price
sns.regplot(x="peak-rpm", y="price", data=df)

#Check correlation
df[['peak-rpm','price']].corr()

# relationship between "body-style" and "price
sns.boxplot(x="body-style", y="price", data=df)

# relationship between "engine location" and "price
sns.boxplot(x="engine-location", y="price", data=df)

# relationship between "drive wheels" and "price
sns.boxplot(x="drive-wheels", y="price", data=df)



df_group = df[['drive-wheels','body-style','price']]

df_group = df_group.groupby(['drive-wheels', 'body-style'],as_index=False).mean()

grouped_pivot = df_group.pivot(index='drive-wheels',columns='body-style')

grouped_pivot = grouped_pivot.fillna(0)

plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)

from scipy import stats

#Calculate Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price'
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])

#Pearson Correlation Coefficient and P-value of 'horsepower' and 'price'
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])

#Pearson Correlation Coefficient and P-value of 'length' and 'price'
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])

#Pearson Correlation Coefficient and P-value of 'width' and 'price'
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])

#Pearson Correlation Coefficient and P-value of 'curb weight' and 'price'
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])

#Pearson Correlation Coefficient and P-value of 'engine size' and 'price'
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])

#Pearson Correlation Coefficient and P-value of 'bore' and 'price'
pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])

#Pearson Correlation Coefficient and P-value of 'city-mpg' and 'price'
pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])

#Pearson Correlation Coefficient and P-value of 'highway-mpg' and 'price'
pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

X = df[['highway-mpg']]
Y = df['price']

lm.fit(X,Y)

Yhat=lm.predict(X)
Yhat[0:5] 

lm.intercept_

lm.coef_

Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

lm.fit(Z, df['price'])

lm.intercept_

lm.coef_

plt.figure(figsize=(12,10))
sns.regplot(x="highway-mpg", y="price", data=df)

plt.figure(figsize=(12,10))
sns.regplot(x="peak-rpm", y="price", data=df)

plt.figure(figsize=(12,10))
sns.residplot(df['highway-mpg'], df['price'])


Y_hat = lm.predict(Z)

plt.figure(figsize=(12,10))


ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')


def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')
    
x = df['highway-mpg']
y = df['price']

# Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit(x, y, 3)
p = np.poly1d(f)

PlotPolly(p, x, y, 'highway-mpg')

np.polyfit(x, y, 3)

from sklearn.preprocessing import PolynomialFeatures

pr=PolynomialFeatures(degree=2)

Z_pr=pr.fit_transform(Z)

Z.shape

Z_pr.shape

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]

pipe=Pipeline(Input)

pipe.fit(Z,y)

ypipe=pipe.predict(Z)

#highway_mpg_fit
lm.fit(X, Y)

# Find the R^2
print('The R-square is: ', lm.score(X, Y))

Yhat=lm.predict(X)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(df['price'], Yhat)

# fit the model 
lm.fit(Z, df['price'])

# Find the R^2
lm.score(Z, df['price'])

Y_predict_multifit = lm.predict(Z)
mean_squared_error(df['price'], Y_predict_multifit)

from sklearn.metrics import r2_score

r_squared = r2_score(y, p(x))

mean_squared_error(df['price'], p(x))
