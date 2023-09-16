# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 15:39:10 2023

@author: harrisfs
"""

import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,summarize)

from ISLP import confusion_table
from ISLP.models import contrast
from sklearn.discriminant_analysis import \
(LinearDiscriminantAnalysis as LDA ,
QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

Smarket = load_data('Smarket')

Smarket.columns


Smarket.plot(y='Volume')

#-----------LOG. REG.---------------

allvars = Smarket.columns.drop(['Today','Direction','Year'])

design = MS(allvars)

X = design.fit_transform(Smarket)

y = Smarket.Direction == 'Up'

#The syntax of sm.GLM() is similar to that of sm.OLS(), except that
# we must pass in the argument family=sm.families.Binomial() in order to
# tell statsmodels to run a logistic regression rather than some other type of
# generalized linear model.

glm = sm.GLM(y,X,family = sm.families.Binomial())

results = glm.fit()

summarize(results)

results.params
results.pvalues

probs = results.predict()

labels = np.array(['Down']*1250) 
labels[probs>0.5] = 'Up'

confusion_table(labels, Smarket.Direction)

(507+145)/1250, np.mean(labels == Smarket.Direction)

#We tested using the training data and still got big error
#Now we will separate some test data

train = (Smarket.Year < 2005)

Smarket_train = Smarket.loc[train]
Smarket_test = Smarket.loc[~train]
Smarket_test.shape

X_train, X_test = X.loc[train],X.loc[~train]
y_train, y_test = y.loc[train],y.loc[~train]
glm_train = sm.GLM(y_train,X_train,family=sm.families.Binomial())

results = glm_train.fit()

probs = results.predict(exog=X_test)

D = Smarket.Direction
L_train, L_test = D.loc[train], D.loc[~train]

labels = np.array(['Down']*252)
labels[probs>0.5] = 'Up'

confusion_table(labels,L_test)

np.mean(labels == L_test) , np.mean(labels != L_test)

model = MS(['Lag1', 'Lag2']).fit(Smarket)
X = model.transform(Smarket)
X_train , X_test = X.loc[train], X.loc[~train]
glm_train = sm.GLM(y_train ,
X_train ,
family=sm.families.Binomial())
results = glm_train.fit()
probs = results.predict(exog=X_test)
labels = np.array(['Down']*252)
labels[probs >0.5] = 'Up'
confusion_table(labels , L_test)

newdata = pd.DataFrame({'Lag1':[1.2, 1.5],
'Lag2':[1.1, -0.8]})

newX = model.transform(newdata)

results.predict(newX)


#-------------LDA----------------

lda = LDA(store_covariance=True)

X_train, X_test = [M.drop(columns = ['intercept']) for M in [X_train,X_test]]

lda.fit(X_train, L_train)

lda.means_
lda.classes_
lda.priors_
lda.scalings_

lda_pred = lda.predict(X_test)

confusion_table(lda_pred, L_test)

lda_prob = lda.predict_proba(X_test)
np.all(
np.where(lda_prob[:,1] >= 0.5, 'Up','Down') == lda_pred
)

#-------------QDA--------------

qda = QDA(store_covariance=True)
qda.fit(X_train,L_train)

qda.means_, qda.priors_
qda.covariance_

qda_pred = qda.predict(X_test)
confusion_table(qda_pred, L_test)

np.mean(qda_pred ==L_test)

#---------Naive Bayes------------

NB = GaussianNB()
NB.fit(X_train, L_train)

NB.classes_
NB.class_prior_
NB.theta_
NB.var_

X_train[L_train == 'Down'].mean()

nb_labels = NB.predict(X_test)

confusion_table(nb_labels, L_test)

#-----------KNN-------------------

