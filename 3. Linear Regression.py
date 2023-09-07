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
X = pd.DataFrame({'intercept' : np.ones(Boston.shape[0]),
                  'lstat': Boston['lstat']})
 # we create a response df which is the median value of the neighbourhood
y= Boston['medv']

model = sm.OLS(y,X)
results = model.fit()

summarize(results)

