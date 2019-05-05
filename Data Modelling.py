# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:13:39 2019

@author: X260
"""

# Import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

# Read in data
df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv', encoding = 'utf-8')

# load the modules for linear regression

from sklearn.linear_model import LinearRegression

# create linear regression object

lm = LinearRegression()

# Create linear functions with Highway miles per gallon and Price

X = df[['highway-mpg']]  #predictor variable
Y = df['price']  # Response Variable

#Fit the linear model using highway-mpg

lm.fit(X,Y)

#output a prediction

Yhat=lm.predict(X)
Yhat[0:5]   

#Get value of the intercept
lm.intercept_

#Get value of the slope 
lm.coef_


#Multiple linear Regression

Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

#fit a linear model
lm.fit(Z,Y)

#Get the value of the intercept
lm.intercept_

#Get the values of the slopes
lm.coef_

#Import visualization packages
import seaborn as sns
%matplotlib inline

#visualize Horsepower as potential predictor variable of price

plt.figure(figsize = (12,10))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)

#compare with peak-rpm

plt.figure(figsize=(12,10))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)

df[["peak-rpm","highway-mpg","price"]].corr()

#Residual plot

plt.figure(figsize=(12,10))
sns.residplot(df['highway-mpg'], df['price'])
plt.show()

#prediction

Y_hat = lm.predict(Z)


plt.figure(figsize=(12,10))

ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()

#fit a polynomiol model
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

    plt.show()
    
#get variables
x = df['highway-mpg']
y = df['price']

# Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)

#Plotting the funtion
PlotPolly(p, x, y, 'highway-mpg')

np.polyfit(x, y, 3)

#perform a polynomial transform on multiple features

from sklearn.preprocessing import PolynomialFeatures

# create a PolynomialFeatures object of degree 2

pr=PolynomialFeatures(degree=2)

Z_pr=pr.fit_transform(Z)

#Original
Z.shape

#after the transformation

Z_pr.shape

#pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#Create a pipeline

Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]

# input the list as an argument to the pipeline constructor

pipe=Pipeline(Input)

#perform a transform and fit the model simultaneously

pipe.fit(Z,y)

#perform a transform and produce a prediction simultaneously
ypipe=pipe.predict(Z)
ypipe[0:4]

#Creating R^2 for Simple linear Regression

#highway_mpg_fit

lm.fit(X, Y)

# Find the R^2

print('The R-square is: ', lm.score(X, Y))

#Calculate the MSE

Yhat=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)

#creating R^2 for Multiple linear Regression

# fit the model 
lm.fit(Z, df['price'])
# Find the R^2
print('The R-square is: ', lm.score(Z, df['price']))

#calculate the MSE

Y_predict_multifit = lm.predict(Z)

#compare the predicted results with the actual results

print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))

#creating R^2 for Polynomial fit

from sklearn.metrics import r2_score

r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)

mean_squared_error(df['price'], p(x))

#Prediction

import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline 

new_input=np.arange(1, 100, 1).reshape(-1, 1)

lm.fit(X, Y)

yhat=lm.predict(new_input)
yhat[0:5]

plt.plot(new_input, yhat)
plt.show()