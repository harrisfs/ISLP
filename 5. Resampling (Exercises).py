import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from ISLP import load_data

#---------Exercise 5---------

Default = load_data('Default')

x01 = sm.add_constant(Default.iloc[:,3:5])
y01 = np.where(Default['default']=='No', 0, 1) 
glm1 = sm.Logit(y01, x01)
print(glm1.fit().summary())

x = pd.DataFrame(Default.iloc[:, 3:5])
y = np.array(Default['default'])
np.random.seed(1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

glm2 = LogisticRegression()
glm2.pred = glm2.fit(x_train, y_train).predict(x_test)

error1 = 1-accuracy_score(y_test, glm2.pred) # 3.1%
print(error1)

error2 = np.zeros(3)
for i in range(2, 5, 1):
    np.random.seed(i)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    
    glm2        = LogisticRegression()
    glm2.pred   = glm2.fit(x_train, y_train).predict(x_test)
    error2[i-2] = 1-accuracy_score(y_test, glm2.pred)
    
error2

Default['student01'] = np.where(Default['student'] == 'No', 0, 1)
x = pd.DataFrame(Default.iloc[:, 3:6])
y = np.array(Default['default'])

error3 = np.zeros(4)
for i in range(1, 5, 1):
    np.random.seed(i)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    
    glm3        = LogisticRegression()
    glm3.pred   = glm3.fit(x_train, y_train).predict(x_test)
    error3[i-1] = 1-accuracy_score(y_test, glm3.pred)
    
error3

#---------Exercise 6---------

x01  = sm.add_constant(Default.iloc[:, 3:5]) 
y01  = np.where(Default['default']=='No', 0, 1) 
glm1 = sm.Logit(y01, x01)
print(glm1.fit().bse)

def coef(x,y):
  glm = sm.Logit(y, x)  
  return glm.fit().params
coef(x01,y01)

def boot(data, n):
    x1 = np.zeros(n)
    x2 = np.zeros(n)
    
    for i in range(0, n):
        df    = data.sample(frac=1, replace=True)
        x     = sm.add_constant(df.iloc[:, 3:5])
        y     = np.where(df['default']=='No', 0, 1) 
        x1[i] = coef(x,y)[1] 
        x2[i] = coef(x,y)[2]
        
    res1 = np.std(x1)
    res2 = np.std(x2)
    print('balance se: %.8f; income se: %.8f' %(res1, res2))
    
boot(Default, 50)

# Bootstrap standard errors are close to glm estimates. 

#---------Exercise 7---------

Weekly = load_data('Weekly')

x01  = sm.add_constant(Weekly.iloc[:, 1:3]) 
y01  = np.where(Weekly['Direction']=='Up', 1, 0) 
glm1 = sm.Logit(y01, x01)
print(glm1.fit().summary())

x       = pd.DataFrame(Weekly.iloc[:, 2:4])
y       = np.array(Weekly['Direction'])
x_train = x.iloc[1:,:]
y_train = y[1:]
x_test  = np.array(x.iloc[0,:]).reshape(1,-1)

glm2      = LogisticRegression()
glm2.pred = glm2.fit(x_train, y_train).predict(x_test)
print('actual: [\'%s\']; predicted: %s' %(y[0], glm2.pred))
# incorrectly classified

n          = len(Weekly)
error_made = np.zeros(n)

for i in range(0, n):
    # (i)-(iii)
    x_train   = x.drop([i])
    y_train   = np.delete(y,i)
    x_test    = np.array(x.iloc[i,:]).reshape(1, -1)
    glm2      = LogisticRegression()
    glm2.pred = glm2.fit(x_train, y_train).predict(x_test)
    
    # (iv)
    if glm2.pred != y[i]:
        error_made[i] = 1
        
np.mean(error_made)   

#---------Exercise 8---------

rng = np.random.default_rng(1)
x = rng.normal(size=100)
y = x - 2 * x**2 + rng.normal(size=100)
    
plt.scatter(x, y) # y is hump-shaped in x.

df = pd.DataFrame({'x': x, 'y': y})
x  = df['x'].values.reshape(-1,1)
y  = df['y'].values.reshape(-1,1)

lm   = LinearRegression()
mse1 = np.zeros(4)
for i in range(1,5):
    poly      = PolynomialFeatures(degree=i)
    x_poly    = poly.fit_transform(x)
    loocv     = KFold(n_splits=100, random_state=1, shuffle = True)
    lm_fit    = lm.fit(x_poly, y)
    scores    = cross_val_score(lm_fit, x_poly, y, scoring="neg_mean_squared_error", cv=loocv)
    mse1[i-1] = np.mean(np.abs(scores))

mse1

lm   = LinearRegression()
mse2 = np.zeros(4)
for i in range(1,5):
    poly      = PolynomialFeatures(degree=i)
    x_poly    = poly.fit_transform(x)
    loocv     = KFold(n_splits=100, random_state=2,shuffle = True)
    lm_fit    = lm.fit(x_poly, y)
    scores    = cross_val_score(lm_fit, x_poly, y, scoring="neg_mean_squared_error", cv=loocv)
    mse2[i-1] = np.mean(np.abs(scores))

mse2

# Yes, results are exactly the same.
# Because LOOCV predicts every observation using the rest (aka no randomness).

np.argmin(mse1) 

poly   = PolynomialFeatures(degree=4)
x_poly = poly.fit_transform(x)  
x_poly = sm.add_constant(x_poly)   
lm     = sm.OLS(y, x_poly)
print(lm.fit().summary())

# x, x^2, and x^4 are statistically significant, consistent with CV results.

#---------Exercise 9---------

Boston = load_data('Boston')

mu_hat = np.mean(Boston.medv) 

mu_hat_se = np.std(Boston.medv)/np.sqrt(len(Boston)) 

def boot(var, n):
    m = np.zeros(n)
    for i in range(0, n):
        v    = var.sample(frac=1, replace=True)
        m[i] = np.mean(v)
    res1 = np.mean(m)
    res2 = np.std(m)
    print('mu: %.2f; se: %.2f' %(res1, res2))
    return(res1, res2)

result = boot(Boston.medv, 50) 

print('lowerbd:%.2f' %(result[0] - 2*result[1]))
print('upperbd:%.2f' %(result[0] + 2*result[1])) 

from scipy import stats
stats.t.interval(0.95,               # confidence level
                 df = len(Boston)-1, # degrees of freedom
                 loc = mu_hat,       # sample mean
                 scale= mu_hat_se)   # sample std dev

# (e)
mu_med_hat = np.median(Boston.medv)
print(mu_med_hat) # 21.2

def boot(var, n):
    m = np.zeros(n)
    for i in range(0, n):
        v     = var.sample(frac=1, replace=True)
        m[i]  = np.median(v)
    r = np.std(m) 
    print(r)

result = boot(Boston.medv, 50)

mu_10_hat = Boston['medv'].quantile(q=0.1)
print(mu_10_hat) # 12.75

def boot(var, n):
    m = np.zeros(n)
    for i in range(0, n):
        v     = var.sample(frac=1, replace=True)
        m[i]  = v.quantile(q=0.1)
    r = np.std(m) 
    print(r)

result = boot(Boston.medv, 50)