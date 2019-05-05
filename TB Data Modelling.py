# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:23:46 2019

@author: X260
"""

import pandas as pd

import numpy as np

all_tb = pd.read_csv('https://raw.githubusercontent.com/waithiegeni/python_analysis/master/TB_burden_countries_2019-01-11.csv', encoding = 'utf-8')

gdp = pd.read_csv('~/Downloads/Income group to TB..csv', encoding = 'utf-8')


#merge the income group column of gdp to tb data

all_tb = all_tb [['country', 'g_whoregion', 'e_inc_num','iso3', 'e_pop_num' ,'e_inc_tbhiv_num', ]]


#rename Country Code of GDP data to iso3

gdp.rename(columns = {"Country Code": "iso3"}, inplace = True )

#include income column to tb data

merged_data = pd.merge(all_tb,gdp, on = ['iso3'])

not_included = pd.merge(all_tb, merged_data, on = ['iso3'], how = "left", indicator = True)

not_included = not_included.loc[not_included["_merge"] == "left_only"]

#rename columns

merged_data.rename(columns = {'country_x': 'country' , 'e_inc_num_x' : 'tb' , 'e_pop_num_x' : 'population', 'g_whoregion' : 'region', 'e_inc_tbhiv_num' : 'hiv'  }, inplace = True)

not_included.rename(columns = {'e_inc_num' : 'tb' , 'e_pop_num' : 'population', 'g_whoregion_x' : 'region'}, inplace = True)

#concatenate
merge = pd.concat([merged_data, not_included], sort =  True, ignore_index = True)

tb =  merge [['country','region', 'IncomeGroup', 'tb', 'hiv', 'population']]

# missing values

missing_data = tb.isnull()

for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

tb.corr()

tb2 = tb.groupby('country').tb.mean()

sns.regplot(x="population", y="tb", data=tb)
plt.ylim(0,)


from scipy import stats


pearson_coef, p_value = stats.pearsonr(tb['population'], tb['tb'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 

pearson_coef, p_value = stats.pearsonr(tb['hiv'], tb['tb'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

pearson_coef, p_value = stats.pearsonr(tb['region'], tb['tb'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


from sklearn.linear_model import LinearRegression

lm = LinearRegression()

X = tb[['population']]/10000

Y = tb['tb']

lm.fit(X,Y)

Yhat=lm.predict(X)

Yhat[0:5]   

lm.intercept_

lm.coef_

plt.figure(figsize=(12,10))
sns.regplot(x="population", y="tb", data=tb)
plt.ylim(0,)


plt.figure(figsize=(12,10))
sns.regplot(x="hiv", y="tb", data=tb)
plt.ylim(0,)

z= tb[['population', 'hiv']]

Y_hat = lm.predict(z)

plt.figure(figsize=(width, height))


ax1 = sns.distplot(tb['tb'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()


reg_grouped = tb.groupby('region').tb.mean().reset_index()

inc_grouped = tb.groupby('IncomeGroup').tb.mean()

tb['region'].unique()

from scipy import stats

reg_grouped.get_group('AFR')['tb']

# ANOVA
f_val, p_val = stats.f_oneway(reg_grouped.get_group('EMR')['tb'], reg_grouped.get_group('EUR')['tb'], reg_grouped.get_group('AFR')['tb'], reg_grouped.get_group('WPR')['tb'], reg_grouped.get_group('AMR')['tb'], reg_grouped.get_group('SEA')['tb'])
f_val, p_val
