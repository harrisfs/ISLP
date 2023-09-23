import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from ISLP import load_data
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence \
import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
from ISLP import confusion_table
from sklearn.discriminant_analysis import \
(LinearDiscriminantAnalysis as LDA ,
QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Exercise 13

#Load data and summarize

Weekly = load_data('Weekly')

Weekly.describe()
pd.crosstab(index=Weekly["Direction"], columns="count")
Weekly.corr() # Volume increases in year.

X = Weekly.drop(columns={'Direction'})

vals = [VIF(X,i)
        for i in range (1,X.shape[1])]

vif = pd.DataFrame({'vif':vals},
                   index=X.columns[1:])

vif


#Logistic Regression Complete

x01 = Weekly.iloc[:, 1:7]
y01 = Weekly.Direction == 'Up'
glm01 = sm.GLM(y01,x01,family = sm.families.Binomial())
results01 = glm01.fit()
print(results01.summary())

probs = results01.predict()
labels = np.array(['Down']*1089) 
labels[probs>0.5] = 'Up'
confusion_table(labels, Weekly.Direction)
accuracy_score(Weekly.Direction, labels)

##Logistic Regression Complete w/ intercept

x02 = sm.add_constant(Weekly.iloc[:, 1:7])
y02 = Weekly.Direction == 'Up'
glm02 = sm.GLM(y02,x02,family = sm.families.Binomial())
results02 = glm02.fit()
print(results02.summary())

probs = results02.predict()
labels = np.array(['Down']*1089) 
labels[probs>0.5] = 'Up'
confusion_table(labels, Weekly.Direction)
accuracy_score(Weekly.Direction, labels)

#Logistic Regression Complete only Lag2 and split data

x03_train = Weekly[Weekly['Year']<2009].Lag2
x03_test = Weekly[Weekly['Year']>=2009].Lag2
y03_train = Weekly[Weekly['Year']<2009].Direction == 'Up'
y03_test = Weekly[Weekly['Year']>=2009].Direction

glm03 = sm.GLM(y03_train,x03_train,family = sm.families.Binomial())
results03 = glm03.fit()
print(results03.summary())

probs03 = results03.predict(exog=x03_test)
labels03 = np.array(['Down']*len(y03_test))
labels03[probs03>0.5] = 'Up'

confusion_table(labels03, y03_test)
accuracy_score(Weekly.Direction, labels)

# same using LDA

lda = LDA(store_covariance=True)
x04_train = Weekly[Weekly['Year']<2009].Lag2
x04_test = Weekly[Weekly['Year']>=2009].Lag2
y04_train = Weekly[Weekly['Year']<2009].Direction == 'Up'
y04_test = Weekly[Weekly['Year']>=2009].Direction

D = Weekly.Direction
train = Weekly.Year < 2009
L_train, L_test = D.loc[train], D.loc[~train]

lda.fit(pd.DataFrame(x04_train), L_train)

lda.means_
lda.classes_
lda.priors_
lda.scalings_

lda_pred = lda.predict(pd.DataFrame(x04_test))
confusion_table(lda_pred, L_test)

lda_prob = lda.predict_proba(pd.DataFrame(x04_test))
np.all(
np.where(lda_prob[:,1] >= 0.5, 'Up','Down') == lda_pred
)

#-----QDA----------------

qda = QDA(store_covariance=True)
qda.fit(pd.DataFrame(x04_train), L_train)

qda.means_, qda.priors_
qda.covariance_

qda_pred = qda.predict(pd.DataFrame(x04_test))
confusion_table(qda_pred, L_test)

np.mean(qda_pred ==L_test)

#---------Naive Bayes-------

NB = GaussianNB()
NB.fit(pd.DataFrame(x04_train), L_train)

NB.classes_
NB.class_prior_
NB.theta_
NB.var_

x04_train[L_train == 'Down'].mean()

nb_labels = NB.predict(pd.DataFrame(x04_test))

confusion_table(nb_labels, L_test)

#-----------KNN-------------------

knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(pd.DataFrame(x04_train), L_train)
knn1_pred = knn1.predict(pd.DataFrame(x04_test))
confusion_table(knn1_pred,L_test)
np.mean(knn1_pred == L_test)

knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(pd.DataFrame(x04_train), L_train)
knn3_pred = knn3.predict(pd.DataFrame(x04_test))
confusion_table(knn3_pred,L_test)
np.mean(knn3_pred == L_test)

# Exercise 14------------

Auto = load_data('Auto')
Auto['mpg01'] = np.where(Auto['mpg'] > Auto['mpg'].median(), 1, 0) 

#SCATTER MATRIX PLOT
pd.plotting.scatter_matrix(Auto.iloc[:,0:10], figsize=(10,10))
# select: displacement, horsepower, weight, acceleration

x_name = ['displacement', 'horsepower', 'weight', 'acceleration']
x      = pd.DataFrame(Auto, columns=x_name)
y      = np.array(Auto['mpg01'])

np.random.seed(1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# (d) LDA
lda      = LinearDiscriminantAnalysis() 
lda.pred = lda.fit(x_train, y_train).predict(x_test)
print(pd.DataFrame(confusion_matrix(y_test, lda.pred), index=['y=0', 'y=1'], columns=['y_pred=0', 'y_pred=1']))
print('error rate: ', 1-accuracy_score(y_test, lda.pred)) # 7.6%

# (e) QDA
qda      = QuadraticDiscriminantAnalysis()
qda.pred = qda.fit(x_train, y_train).predict(x_test)
print(pd.DataFrame(confusion_matrix(y_test, qda.pred), index=['y=0', 'y=1'], columns=['y_pred=0', 'y_pred=1']))
print('error rate: ', 1-accuracy_score(y_test, qda.pred)) # 3.8%

# (f) Logit
glm      = LogisticRegression()
glm.pred = glm.fit(x_train, y_train).predict(x_test)
print(pd.DataFrame(confusion_matrix(y_test, glm.pred), index=['y=0', 'y=1'], columns=['y_pred=0', 'y_pred=1']))
print('error rate: ', 1-accuracy_score(y_test, glm.pred)) # 7.6%

# (g) KNN
error_rate = np.array([]) 
k_value    = np.array([]) 
for i in range(1, 110, 1):  
    knn        = KNeighborsClassifier(n_neighbors=i)
    knn.pred   = knn.fit(x_train, y_train).predict(x_test) 
    k_value    = np.append(k_value, i)
    error_rate = np.append(error_rate, 1-accuracy_score(y_test, knn.pred))

plt.plot(error_rate)
best_k = k_value[error_rate.argmin()]
print('KNN best when k=%i' %best_k)

#----------Exercise 15-------

def Power2(x, a):
    print(x**a)
    
print(Power2(3,5))

def Power3(x, a):
    return x**a

print(Power3(3,5))

x = np.arange(1, 11, 1)
y = Power3(x,2)

fig = plt.figure() 
fig.add_subplot(2, 2, 1)
plt.scatter(x, y)
plt.title('log(x^2) vs x')
plt.xlabel('x')
plt.ylabel('log(x^2)')

def PlotPower(x, a):
    y = Power3(x, a)
    plt.scatter(x, y)
    plt.title('x^%.0f vs x' %a)
    plt.xlabel('x')
    plt.ylabel('x^%.0f' %a)

PlotPower(np.arange(1,11,1), 3)

#-------Exercise 16---------

Boston = load_data('Boston')

Boston['crim01'] = np.where(Boston['crim'] > Boston['crim'].median(), 1, 0) 

pd.plotting.scatter_matrix(Boston.iloc[:,2:17]) # nox, rm, dis, tax, black, lstat, medv

x_name = ['indus', 'nox', 'dis', 'tax', 'lstat']
x      = pd.DataFrame(Boston, columns=x_name)
y      = np.array(Boston['crim01'])

np.random.seed(1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Logit
glm      = LogisticRegression()
glm.pred = glm.fit(x_train, y_train).predict(x_test)
print(pd.DataFrame(confusion_matrix(y_test, glm.pred), index=['y=0', 'y=1'], columns=['y_pred=0', 'y_pred=1']))
print('error rate: ', 1-accuracy_score(y_test, glm.pred)) # 21.7%

# LDA
lda      = LinearDiscriminantAnalysis() 
lda.pred = lda.fit(x_train, y_train).predict(x_test)
print(pd.DataFrame(confusion_matrix(y_test, lda.pred), index=['y=0', 'y=1'], columns=['y_pred=0', 'y_pred=1']))
print('error rate: ', 1-accuracy_score(y_test, lda.pred)) # 17.1%

# QDA
qda      = QuadraticDiscriminantAnalysis()
qda.pred = qda.fit(x_train, y_train).predict(x_test)
print(pd.DataFrame(confusion_matrix(y_test, qda.pred), index=['y=0', 'y=1'], columns=['y_pred=0', 'y_pred=1']))
print('error rate: ', 1-accuracy_score(y_test, qda.pred)) # 15.1%

# KNN
error_rate = np.array([]) 
k_value    = np.array([]) 
for i in range(1, 110, 10):  
    knn        = KNeighborsClassifier(n_neighbors=i)
    knn.pred   = knn.fit(x_train, y_train).predict(x_test) 
    k_value    = np.append(k_value, i)
    error_rate = np.append(error_rate, 1-accuracy_score(y_test, knn.pred))

best_k = k_value[error_rate.argmin()]
print('KNN best when k=%i' %best_k)
# k = 1 is the best

