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

knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train, L_train)
knn1_pred = knn1.predict(X_test)
confusion_table(knn1_pred,L_test)
np.mean(knn1_pred == L_test)

knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train, L_train)
knn3_pred = knn3.predict(X_test)
confusion_table(knn3_pred,L_test)
np.mean(knn3_pred == L_test)

Caravan = load_data("Caravan")
Purchase = Caravan.Purchase
Purchase.value_counts()

feature_df = Caravan.drop(columns=['Purchase'])

scaler = StandardScaler(with_mean=True,with_std=True,copy=True)

scaler.fit(feature_df)
X_std = scaler.transform(feature_df)

feature_std = pd.DataFrame(X_std,columns=feature_df.columns)
feature_std.std()

(X_train, X_test, y_train, y_test) = train_test_split(feature_std,Purchase,test_size=1000,random_state=0)

knn1 = KNeighborsClassifier(n_neighbors=1)
knn1_pred = knn1.fit(X_train, y_train).predict(X_test.values)
np.mean(y_test != knn1_pred) , np.mean(y_test != "No")

confusion_table(knn1_pred , y_test)

for K in range(1,6):
    knn = KNeighborsClassifier(n_neighbors=K)
    knn_pred = knn.fit(X_train , y_train).predict(X_test.values)
    C = confusion_table(knn_pred, y_test)
    templ = ('K={0:d}: # predicted to rent: {1:>2},' +
    ' # who did rent {2:d}, accuracy {3:.1%}')
    pred = C.loc['Yes'].sum()
    did_rent = C.loc['Yes','Yes']
    print(templ.format(
    K,
    pred ,
    did_rent ,
    did_rent / pred))
    
#-------Logit for comparison with KNN---------

logit = LogisticRegression(C=1e10,solver='liblinear')
logit.fit(X_train, y_train)
logit_pred = logit.predict_proba(X_test)
logit_labels = np.where(logit_pred[:,1] > 5, 'Yes', 'No')
confusion_table(logit_labels , y_test)

logit_labels = np.where(logit_pred[:,1]>0.25, 'Yes', 'No')
confusion_table(logit_labels , y_test)

#-------Linear & Poisson regression------
#-------Linear------------

Bike = load_data('Bikeshare')

X = MS(['mnth','hr','workingday','temp','weathersit']).fit_transform(Bike)

Y = Bike['bikers']

M_lm = sm.OLS(Y,X).fit()

summarize(M_lm)

hr_encode = contrast('hr', 'sum')
mnth_encode = contrast('mnth', 'sum')

X2 = MS([mnth_encode,hr_encode,'workingday','temp','weathersit']).fit_transform(Bike)
M2_lm = sm.OLS(Y,X2).fit()

S2 = summarize(M2_lm)

S2

coef_month = S2[S2.index.str.contains('mnth')]['coef']
months = Bike['mnth'].dtype.categories
coef_month = pd.concat([
coef_month ,
pd.Series([-coef_month.sum()],
index=['mnth[Dec]'
])
])

coef_month

fig_month , ax_month = subplots(figsize=(8,8))
x_month = np.arange(coef_month.shape[0])
ax_month.plot(x_month , coef_month , marker='o', ms=10)
ax_month.set_xticks(x_month)
ax_month.set_xticklabels([l[5] for l in coef_month.index], fontsize
=20)
ax_month.set_xlabel('Month', fontsize=20)
ax_month.set_ylabel('Coefficient', fontsize=20);

coef_hr = S2[S2.index.str.contains('hr')]['coef']
coef_hr = coef_hr.reindex(['hr[{0}]'.format(h) for h in range(23)])
coef_hr = pd.concat([coef_hr ,pd.Series([-coef_hr.sum()], index=['hr[23]'])
])

fig_hr , ax_hr = subplots(figsize=(8,8))
x_hr = np.arange(coef_hr.shape[0])
ax_hr.plot(x_hr , coef_hr , marker='o', ms=10)
ax_hr.set_xticks(x_hr[::2])
ax_hr.set_xticklabels(range(24)[::2], fontsize =20)
ax_hr.set_xlabel('Hour', fontsize=20)
ax_hr.set_ylabel('Coefficient', fontsize=20);

#--------Poisson--------

M_pois = sm.GLM(Y, X2, family=sm.families.Poisson()).fit()

S_pois = summarize(M_pois)

S_pois

coef_month = S_pois[S_pois.index.str.contains('mnth')]['coef']
coef_month = pd.concat([coef_month ,
pd.Series([-coef_month.sum()],
index=['mnth[Dec]'])])
coef_hr = S_pois[S_pois.index.str.contains('hr')]['coef']
coef_hr = pd.concat([coef_hr ,
pd.Series([-coef_hr.sum()],
index=['hr[23]'])])

fig_pois , (ax_month , ax_hr) = subplots(1, 2, figsize=(16,8))
ax_month.plot(x_month , coef_month , marker='o', ms=10)
ax_month.set_xticks(x_month)
ax_month.set_xticklabels([l[5] for l in coef_month.index], fontsize
=20)
ax_month.set_xlabel('Month', fontsize=20)
ax_month.set_ylabel('Coefficient', fontsize=20)
ax_hr.plot(x_hr , coef_hr , marker='o', ms=10)
ax_hr.set_xticklabels(range(24)[::2], fontsize =20)
ax_hr.set_xlabel('Hour', fontsize=20)
ax_hr.set_ylabel('Coefficient', fontsize=20);


fig , ax = subplots(figsize=(8, 8))
ax.scatter(M2_lm.fittedvalues ,
M_pois.fittedvalues ,
s=20)
ax.set_xlabel('Linear Regression Fit', fontsize=20)
ax.set_ylabel('Poisson Regression Fit', fontsize=20)
ax.axline([0,0], c='black', linewidth=3,
linestyle='--', slope=1);


















                     