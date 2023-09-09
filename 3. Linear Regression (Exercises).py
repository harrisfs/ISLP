# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:04:19 2023

@author: harrisfs
"""

import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import os

import matplotlib.pyplot as plt
import scipy.stats as stats

import statsmodels.api as sm

from statsmodels.stats.outliers_influence \
import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm

from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
summarize ,
poly)
import seaborn as sns

# Exercise 8

Auto = load_data('Auto')

X= MS(['horsepower']).fit_transform(Auto)

y= Auto['mpg']

model = sm.OLS(y,X)
results = model.fit()
results_summary = pd.DataFrame(summarize(results))

def abline (ax,b,m,*args,**kwargs):
    #add line with slope m and intercept b to ax
    xlim = ax.get_xlim()
    ylim = [m*xlim[0]+b,m*xlim[1]+b]
    ax.plot(xlim,ylim,*args,**kwargs)
    
ax = Auto.plot.scatter('horsepower','mpg')
abline(ax,results.params[0],results.params[1],'r--',linewidth=3)

ax = subplots(figsize=(8,8))[1]
ax.scatter(results.fittedvalues , results.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--')

infl = results.get_influence()
ax = subplots(figsize=(8,8))[1]
ax.scatter(np.arange(X.shape[0]),infl.hat_matrix_diag)
ax.set_xlabel('Index')
ax.set_ylabel('Leverage')
np.argmax(infl.hat_matrix_diag)

#Exercise 9

sns.set_theme(style="ticks")
sns.pairplot(Auto, hue="Auto")

correlations = pd.DataFrame(Auto.corr())

terms = Auto.columns.drop(['name','mpg'])

X1 = MS(terms).fit_transform(Auto)
model1 = sm.OLS(y,X1)
results1 = model1.fit()
results1_summary = pd.DataFrame(summarize(results1))

anovadf = pd.DataFrame(anova_lm(results,results1))

ax = subplots(figsize=(8,8))[1]
ax.scatter(results1.fittedvalues , results1.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--')

infl = results1.get_influence()
ax = subplots(figsize=(8,8))[1]
ax.scatter(np.arange(X1.shape[0]),infl.hat_matrix_diag)
ax.set_xlabel('Index')
ax.set_ylabel('Leverage')
np.argmax(infl.hat_matrix_diag)


terms = list(Auto.columns.drop(['name','mpg']))
extra_terms =  [('cylinders','weight'),('displacement','year')]

X2 = MS(terms + extra_terms).fit_transform(Auto)
model2 = sm.OLS(y,X2)
results2 = model2.fit()
results2_summary = pd.DataFrame(summarize(results2))

#Exercise 10

Carseats = load_data('Carseats')

Xc = MS(['Price', "Urban", 'US']).fit_transform(Carseats)
yc = Carseats['Sales']

modelc = sm.OLS(yc,Xc)
resultsc= modelc.fit()
resultsc_summary = pd.DataFrame(summarize(resultsc))

# Access and print the detailed statistics
rss = resultsc.ssr  # Residual Sum of Squares
rsquared = resultsc.rsquared  # R-squared
f_statistic = resultsc.fvalue  # F-statistic

# Print the statistics
print("Residual Sum of Squares (RSS):", rss)
print("R-squared:", rsquared)
print("F-statistic:", f_statistic)

Xc1 = MS(['Price','US']).fit_transform(Carseats)
yc1 = Carseats['Sales']

modelc1 = sm.OLS(yc1,Xc1)
resultsc1= modelc1.fit()
resultsc1_summary = pd.DataFrame(summarize(resultsc1))

# Access and print the detailed statistics
rss = resultsc1.ssr  # Residual Sum of Squares
rsquared = resultsc1.rsquared  # R-squared
f_statistic = resultsc1.fvalue  # F-statistic

# Print the statistics
print("Residual Sum of Squares (RSS):", rss)
print("R-squared:", rsquared)
print("F-statistic:", f_statistic)


standardized_residuals = resultsc1.resid / resultsc1.get_influence().resid_std

# Generate the Normal Q-Q plot
sm.qqplot(standardized_residuals, line='45', fit=True)

# Customize the plot
plt.title("Normal Q-Q Plot")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")

# Display the plot
plt.show()

#Residual Plot
ax = subplots(figsize=(8,8))[1]
ax.scatter(resultsc1.fittedvalues , resultsc1.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--')

anovadfc = pd.DataFrame(anova_lm(resultsc,resultsc1))

#Exercise 11


