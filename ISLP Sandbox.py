import numpy as np
import pandas as pd
import os

cwd1 = '/Users/harrisfs/Desktop'
os.chdir(cwd1)
print(cwd1)

import pandas as pd

Auto = pd.read_csv('Auto.data',na_values=['?'],delim_whitespace=True)
Auto

Auto ['horsepower']

np.unique(
Auto ['horsepower'])

Auto.shape

Auto['horsepower'].sum()
Auto_new = Auto.dropna()
Auto_new.shape

Auto = Auto.dropna()
Auto.shape

Auto.columns

Auto[:3]

idx_80 = Auto['year'] > 80
Auto.loc[idx_80,['name','mpg']]

Auto.iloc[[1,2],[3,5,6]]

Auto_re = Auto.set_index('name')

Auto_re.loc[lambda df: (df['year'] > 80) & (df['mpg']>30) ,['weight','origin']]

Auto_re.loc[lambda df: (df['year'] > 80) & df.index.str.contains('ford') | df.index.str.contains('datsun') ,['weight','origin']]

rng = np.random.default_rng(1)
A = rng.standard_normal((127, 5))
M = rng.choice([0, np.nan], p=[0.8,0.2], size=A.shape)
A += M
D = pd.DataFrame(A, columns=['food','bar', 'pickle', 'snack', 'popcorn'])

D[:3]

for col in D.columns:
    template = 'Column "{0}" has {1:.2%} missing values'
    print(template.format(col,np.isnan(D[col]).mean()))
    

import matplotlib
from matplotlib.pyplot import subplots 
fig, ax = subplots(figsize=(8, 8))
x = rng.standard_normal(100) 
y = rng.standard_normal(100) 
ax.plot(x, y);

from matplotlib.pyplot import subplots
fig, ax = subplots(figsize = (8,8))
ax.plot(Auto['horsepower'], Auto['mpg'], 'o')
fig.show();

ax = Auto.plot.scatter('horsepower', 'mpg')
ax.set_title('Horsepower vs MPG')
fig = ax.figure
fig.show();

fig, axes = subplots(ncols=3, figsize=(15, 5)) 
Auto.plot.scatter('horsepower', 'mpg', ax=axes[1])
fig = axes.figure
fig.show();

Auto.cylinders.dtype
Auto.cylinders = pd.Series(Auto.cylinders, dtype='category')

fig, ax = subplots(figsize=(8, 8)) 
Auto.boxplot('mpg', by='cylinders', ax=ax)
fig = ax.figure
fig.show();

fig, ax = subplots(figsize=(8, 8)) 
Auto.hist('mpg', bins=12,ax=ax)
fig = ax.figure
fig.show();

Auto[['mpg', 'weight']].describe()

import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.DataFrame({'Obs': [1, 2, 3, 4, 5, 6],
                   'X1': [0, 2, 0, 0, -1, 1],
                   'X2': [3, 0, 1, 1, 0, -1],
                   'X3': [0, 0, 3, 2, 1, 1],
                   'Y': ['Red', 'Red', 'Red', 'Green', 'Green', 'Red']})

df

def euclidian_dist(x):
    """Compute the row-wise euclidean distance
    from the origin"""
    return (np.sum(x**2, axis=1))**0.5

euc_dist = pd.DataFrame({'EuclideanDist': euclidian_dist(df[['X1', 'X2', 'X3']])})

euc_dist

df_euc = pd.concat([df, euc_dist], axis=1)
df_euc

K = 3
df_euc.nsmallest(K, 'EuclideanDist')

df_c = pd.read_csv('College.csv')
college = pd.read_csv('College.csv')

college2 = pd.read_csv('College.csv', index_col=0) 

college2

college3 = college.rename({'Unnamed: 0': 'College'},axis=1)

college3

college3 = college3.set_index('College')

college = college3

college.describe()

fig, ax = subplots(figsize=(8, 8)) 
college.boxplot('Outstate', by='Elite', ax=ax)
fig = ax.figure
fig.show();

college['Elite'] = college['Top10perc'] > 50

college['Elite'].sum()

Auto

Auto.applymap(lambda x: isinstance(x, str) and '?' in x).any().any()

Auto.head()

datatypes = {'quant': ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration'],
             'qual': ['origin', 'name']}

auto_quant = Auto[datatypes['quant']].astype(np.float_)

auto_quant.max()

auto_quant_d = auto_quant.drop(auto_quant.index[10: 85])


