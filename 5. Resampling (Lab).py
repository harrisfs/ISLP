import numpy as np
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
summarize ,
poly)
from sklearn.model_selection import train_test_split
from functools import partial
from sklearn.model_selection import \
(cross_validate ,
KFold ,
ShuffleSplit)
from sklearn.base import clone
from ISLP.models import sklearn_sm

Auto = load_data('Auto')
Auto_train , Auto_valid = train_test_split(Auto ,test_size=196,random_state=0)

hp_mm = MS(['horsepower'])
X_train = hp_mm.fit_transform(Auto_train)
y_train = Auto_train['mpg']
model = sm.OLS(y_train , X_train)
results = model.fit()

X_valid = hp_mm.transform(Auto_valid)
y_valid = Auto_valid['mpg']
valid_pred = results.predict(X_valid)
np.mean((y_valid - valid_pred)**2)

#higher degree

def evalMSE(terms ,response ,train ,test):
    mm = MS(terms)
    X_train = mm.fit_transform(train)
    y_train = train[response]
    X_test = mm.transform(test)
    y_test = test[response]
    results = sm.OLS(y_train , X_train).fit()
    test_pred = results.predict(X_test)
    return np.mean((y_test - test_pred)**2)

MSE = np.zeros(3)
for idx , degree in enumerate(range(1, 4)):
    MSE[idx] = evalMSE([poly('horsepower', degree)],
    'mpg',
    Auto_train ,
    Auto_valid)
    
MSE

#use different split

Auto_train , Auto_valid = train_test_split(Auto ,test_size=196,random_state=3)

MSE = np.zeros(3)
for idx , degree in enumerate(range(1, 4)):
    MSE[idx] = evalMSE([poly('horsepower', degree)],
    'mpg',
    Auto_train ,
    Auto_valid)
    
MSE

#etc

#---- Cross Validation-----

hp_model = sklearn_sm(sm.OLS ,
MS(['horsepower']))

X, Y = Auto.drop(columns=['mpg']), Auto['mpg']
cv_results = cross_validate(hp_model ,
X,
Y,
cv=Auto.shape[0])
cv_err = np.mean(cv_results['test_score'])
cv_err

cv_error = np.zeros(5)
H = np.array(Auto['horsepower'])
M = sklearn_sm(sm.OLS)
for i, d in enumerate(range(1,6)):
    X = np.power.outer(H, np.arange(d+1))
    M_CV = cross_validate(M,
    X,
    Y,
    cv=Auto.shape[0])
    cv_error[i] = np.mean(M_CV['test_score'])
cv_error

cv_error = np.zeros(5)
cv = KFold(n_splits=10,
shuffle=True ,
random_state=0) # use same splits for each degree
for i, d in enumerate(range(1,6)):
    X = np.power.outer(H, np.arange(d+1))
    M_CV = cross_validate(M,
    X,
    Y,
    cv=cv)
    cv_error[i] = np.mean(M_CV['test_score'])
cv_error

validation = ShuffleSplit(n_splits=1,
    test_size=196,
    random_state=0)
results = cross_validate(hp_model ,
    Auto.drop(['mpg'], axis=1),
    Auto['mpg'],
    cv=validation);
results['test_score']

validation = ShuffleSplit(n_splits=10,
    test_size=196,
    random_state=0)
results = cross_validate(hp_model ,
    Auto.drop(['mpg'], axis=1),
    Auto['mpg'],
    cv=validation)
results['test_score'].mean(), results['test_score'].std()

#--------Bootstrap----

Portfolio = load_data('Portfolio')

def alpha_func(D, idx):
    cov_ = np.cov(D[['X','Y']].loc[idx], rowvar=False)
    return ((cov_[1,1] - cov_[0,1])/(cov_[0,0]+cov_[1,1]-2*cov_[0,1]))

alpha_func(Portfolio , range(100))

rng = np.random.default_rng(0)
alpha_func(Portfolio ,
    rng.choice(100,
    100,
    replace=True))

def boot_SE(func ,D,n=None ,B=1000,seed=0):
    rng = np.random.default_rng(seed)
    first_ , second_ = 0, 0
    n = n or D.shape[0]
    for _ in range(B):
        idx = rng.choice(D.index ,
        n,
        replace=True)
        value = func(D, idx)
        first_ += value
        second_ += value**2
    return np.sqrt(second_ / B - (first_ / B)**2)

alpha_SE = boot_SE(alpha_func ,
    Portfolio ,
    B=1000,
    seed=0)
alpha_SE


def boot_OLS(model_matrix , response , D, idx):
    D_ = D.loc[idx]
    Y_ = D_[response]
    X_ = clone(model_matrix).fit_transform(D_)
    return sm.OLS(Y_, X_).fit().params

hp_func = partial(boot_OLS , MS(['horsepower']), 'mpg')

rng = np.random.default_rng(0)
np.array([hp_func(Auto ,
rng.choice(392,
392,
replace=True)) for _ in range(10)])

hp_se = boot_SE(hp_func ,
Auto ,
B=1000,
seed=10)
hp_se