# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 20:53:33 2023

@author: harrisfs
"""

import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import os

import statsmodels.api as sm

from statsmodels.stats.outliers_influence \
import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm

from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
summarize ,
poly)

dir()

A = np.array([3,5,11])
dir(A)
A.sum()

Boston = pd.read_csv('C:\\Users\\harrisfs\Desktop\ISLP\Datasets\Boston.csv')
Boston = load_data("Boston")
Boston.columns

# we create a variable df with just the intercept and the low income varibale
X1 = pd.DataFrame({'intercept' : np.ones(Boston.shape[0]),
                  'lstat': Boston['lstat']})
 # we create a response df which is the median value of the neighbourhood
y= Boston['medv']

model = sm.OLS(y,X1)
results = model.fit()

summarize(results)

design = MS(['lstat'])
design = design.fit(Boston)
X = design.transform(Boston)

X = design.fit_transform(Boston)

results.params

Xpred = pd.DataFrame({'lstat':[5,10,15]})
Xpred = design.transform(Xpred)

Ypred = results.get_prediction(Xpred)
Ypred.predicted_mean
Ypred.conf_int(obs = True, alpha = 0.05)

def abline (ax,b,m,*args,**kwargs):
    #add line with slope m and intercept b to ax
    xlim = ax.get_xlim()
    ylim = [m*xlim[0]+b,m*xlim[1]+b]
    ax.plot(xlim,ylim,*args,**kwargs)
    
ax = Boston.plot.scatter('lstat','medv')
abline(ax,results.params[0],results.params[1],'r--',linewidth=3)

ax = subplots(figsize=(8,8))[1]
ax.scatter(results.fittedvalues,results.resid)
ax.set_xlabel('Fitted Value')
ax.set_ylabel('Residual')
ax.axhline (0,c='k',ls='--')

infl = results.get_influence()
ax = subplots(figsize=(8,8))[1]
ax.scatter(np.arange(X.shape[0]),infl.hat_matrix_diag)
ax.set_xlabel('Index')
ax.set_ylabel('Leverage')
np.argmax(infl.hat_matrix_diag)

X=MS(['lstat','age']).fit_transform(Boston)
model1 = sm.OLS(y,X)
results1 =  model1.fit()
summarize(results1)

terms= Boston.columns.drop('medv')

X=MS(terms).fit_transform(Boston)
model1 = sm.OLS(y,X)
results1 =  model1.fit()
summarize(results1)

vals = [VIF(X,i)
        for i in range (1,X.shape[1])]

vif = pd.DataFrame({'vif':vals},
                   index=X.columns[1:])

vif

X2 = MS(['lstat','age',('lstat','age')]).fit_transform(Boston)

model2 = sm.OLS(y,X2)
results2 = model2.fit()
summarize(results2)

X3 = MS([poly('lstat',degree=2),'age']).fit_transform(Boston)

model3 = sm.OLS(y,X3)
results3 = model3.fit()
summarize(results3)

anova_lm(results1,results2,results3)
anovadf = pd.DataFrame(anova_lm(results,results3))

ax = subplots(figsize=(8,8))[1]
ax.scatter(results3.fittedvalues , results3.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--')

Carseats = load_data('Carseats')
Carseats.columns

allvars = list(Carseats.columns.drop('Sales'))
y = Carseats['Sales']
final = allvars + [('Income','Advertising'),('Price','Age')]

X = MS(final).fit_transform(Carseats)
model = sm.OLS(y,X)
summarize(model.fit())
