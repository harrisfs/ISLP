import numpy as np, pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (summarize ,
poly ,
ModelSpec as MS)
from statsmodels.stats.anova import anova_lm

from pygam import (s as s_gam ,
l as l_gam,
f as f_gam,
LinearGAM ,
LogisticGAM)
from ISLP.transforms import (BSpline ,
NaturalSpline)
from ISLP.models import bs, ns
from ISLP.pygam import (approx_lam ,
degrees_of_freedom ,
plot as plot_gam ,
anova as anova_gam)

# Polynomial Regression and Step Functions

Wage = load_data('Wage')
y = Wage['wage']
age = Wage['age']

poly_age = MS([poly('age', degree=4)]).fit(Wage)
M = sm.OLS(y, poly_age.transform(Wage)).fit()
summarize(M)

age_grid = np.linspace(age.min(),
age.max(),
100)
age_df = pd.DataFrame({'age': age_grid})

def plot_wage_fit(age_df ,
    basis ,
    title):
    X = basis.transform(Wage)
    Xnew = basis.transform(age_df)
    M = sm.OLS(y, X).fit()
    preds = M.get_prediction(Xnew)
    bands = preds.conf_int(alpha=0.05)
    fig , ax = subplots(figsize=(8,8))
    ax.scatter(age ,
    y,
    facecolor='gray',
    alpha=0.5)
    for val , ls in zip([preds.predicted_mean ,
    bands[:,0],
    bands[:,1]],
    ['b','r--','r--']):
        ax.plot(age_df.values , val , ls, linewidth=3)
    ax.set_title(title , fontsize =20)
    ax.set_xlabel('Age', fontsize=20)
    ax.set_ylabel('Wage', fontsize=20);
    return ax

plot_wage_fit(age_df ,
poly_age ,
'Degree -4 Polynomial');

models = [MS([poly('age', degree=d)])
for d in range(1, 6)]
Xs = [model.fit_transform(Wage) for model in models]
anova_lm(*[sm.OLS(y, X_).fit()
for X_ in Xs])

models = [MS(['education', poly('age', degree=d)])
for d in range(1, 4)]
XEs = [model.fit_transform(Wage)
for model in models]
anova_lm(*[sm.OLS(y, X_).fit() for X_ in XEs])

X = poly_age.transform(Wage)
high_earn = Wage['high_earn'] = y > 250 # shorthand
glm = sm.GLM(y > 250,
X,
family=sm.families.Binomial())
B = glm.fit()
summarize(B)

newX = poly_age.transform(age_df)
preds = B.get_prediction(newX)
bands = preds.conf_int(alpha=0.05)

fig , ax = subplots(figsize=(8,8))
rng = np.random.default_rng(0)
ax.scatter(age +
0.2 * rng.uniform(size=y.shape[0]),
np.where(high_earn , 0.198, 0.002),
fc='gray',
marker='|')
for val , ls in zip([preds.predicted_mean ,
bands[:,0],
bands[:,1]],
['b','r--','r--']):
    ax.plot(age_df.values , val , ls, linewidth=3)
ax.set_title('Degree -4 Polynomial', fontsize=20)
ax.set_xlabel('Age', fontsize=20)
ax.set_ylim ([0,0.2])
ax.set_ylabel('P(Wage > 250)', fontsize=20);

cut_age = pd.qcut(age , 4)
summarize(sm.OLS(y, pd.get_dummies(cut_age)).fit())

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split


# Generate sample data
np.random.seed(0)
x = np.random.rand(50) * 10  # Random x values
y = 3 * x - 5 + np.random.randn(50) * 5  # Linear relationship with noise

# Reshape x to be a column vector
x = x.reshape(-1, 1)

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Degrees of the polynomial
degrees = [1, 2, 3, 4, 5, 6]

# Initialize lists to store models and mean squared errors
mse_values = []

for degree in degrees:
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)

    # Fit the model
    model = LinearRegression()
    model.fit(x_train_poly, y_train)

    # Predict on the test set
    y_pred = model.predict(x_test_poly)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)
    mse_values.append(mse)

# Create a DataFrame for ANOVA
data = {'Degree': degrees, 'MSE': mse_values}
df = pd.DataFrame(data)

# Fit the ANOVA model
anova_model = ols('MSE ~ C(Degree)', data=df).fit()

# Generate the ANOVA table
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print(anova_table)

# Splines

bs_ = BSpline(internal_knots=[25,40,60], intercept=True).fit(age)
bs_age = bs_.transform(age)
bs_age.shape

bs_age = MS([bs('age', internal_knots=[25,40,60])])
Xbs = bs_age.fit_transform(Wage)
M = sm.OLS(y, Xbs).fit()
summarize(M)

bs_age = MS([bs('age',
internal_knots=[25,40,60],
name='bs(age)')])
Xbs = bs_age.fit_transform(Wage)
M = sm.OLS(y, Xbs).fit()
summarize(M)

bs_age0 = MS([bs('age',
df=3,
degree=0)]).fit(Wage)
Xbs0 = bs_age0.transform(Wage)
summarize(sm.OLS(y, Xbs0).fit())

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 1, 4, 9, 16, 25])
cubic_spline = CubicSpline(x, y)
x_interpolated = np.linspace(min(x), max(x), 100)
y_interpolated = cubic_spline(x_interpolated)

plt.scatter(x, y, label='Original Data')

# Plot the interpolated curve
plt.plot(x_interpolated, y_interpolated, label='Cubic Spline Interpolation')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Generate some test x-values
test_x = np.array([1.5, 2.7, 4.2])

# Use the cubic spline to interpolate y-values for the test x-values
interpolated_y = cubic_spline(test_x)

print("Interpolated y-values for test x-values:", interpolated_y)

# Visualize the original data, the cubic spline, and the test points
plt.scatter(x, y, label='Original Data')
plt.plot(x, y, 'o')
plt.plot(x, cubic_spline(x), label='Cubic Spline Interpolation')
plt.scatter(test_x, interpolated_y, color='red', label='Interpolated Points')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Smoothing Splines

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from sklearn.model_selection import KFold

# Generate noisy sine wave data
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + np.random.normal(scale=0.3, size=len(x))

# Perform k-fold cross-validation to find the optimal smoothing parameter (s)
kf = KFold(n_splits=5)
s_values = np.logspace(-2, 2, 20)  # Range of smoothing parameters to test
mean_rss_values = []
splines = []

for s in s_values:
    rss_values = []  # Store RSS for each fold

    for train_index, val_index in kf.split(x):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        spline = UnivariateSpline(x_train, y_train, s=s)
        y_val_pred = spline(x_val)
        rss = np.sum((y_val - y_val_pred) ** 2)
        rss_values.append(rss)

    mean_rss = np.mean(rss_values)
    mean_rss_values.append(mean_rss)
    splines.append(spline)

# Plot all splines and the actual sine wave
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Noisy Data')
plt.plot(x, np.sin(x), label='True Sine Wave', color='red')

# Find the index of the minimum mean RSS
best_s_index = np.argmin(mean_rss_values)
best_s = s_values[best_s_index]

# Fit a smoothing spline with the best smoothing parameter
best_spline = UnivariateSpline(x, y, s=best_s)

# Plot the original noisy data, true sine wave, and the smoothed spline
plt.scatter(x, y, label='Noisy Data')
plt.plot(x, np.sin(x), label='True Sine Wave', color='red')
plt.plot(x, best_spline(x), label=f'Smoothed Spline (s={best_s:.2f})', color='green')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Noisy Sine Wave with Smoothed Spline (Optimal s)')
plt.show()

# GAMs

import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, f, te
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(0)
n = 300
x1 = np.random.rand(n) * 10
x2 = np.random.rand(n) * 10
y = 3 * np.sin(x1) + 2 * np.cos(x2) + np.random.normal(scale=2, size=n)

# Create a DataFrame
import pandas as pd
data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

# Split the data into training and test sets
X = data[['x1', 'x2']]
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit a GAM
gam = LinearGAM(s(0) + s(1)).fit(X_train, y_train)

# Plot the individual component functions
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

for i, ax in enumerate(axs):
    XX = gam.generate_X_grid(term=i)
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    ax.set_title('Partial Dependence of y on ' + ['x1', 'x2'][i])
    ax.set_xlabel(['x1', 'x2'][i])
    ax.set_ylabel('Partial Dependence')

plt.tight_layout()
plt.show()

# Evaluate the model on the test set
y_pred = gam.predict(X_test)

# Calculate R-squared for the model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)

import numpy as np
from pygam import LinearGAM, s, te
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
n = 300
x1 = np.random.rand(n) * 10
x2 = np.random.rand(n) * 10
y = 3 * np.sin(x1) + 2 * np.cos(x2) + np.random.normal(scale=2, size=n)

# Create a DataFrame
import pandas as pd
data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

# Split the data into training and test sets
X = data[['x1', 'x2']]
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Strategy 1: Adjust smoothness parameters
# Fit a GAM with different smoothness parameters
gam_1 = LinearGAM(s(0, lam=0.1) + s(1, lam=0.1)).fit(X_train, y_train)
gam_2 = LinearGAM(s(0, lam=1) + s(1, lam=1)).fit(X_train, y_train)

# Strategy 2: Incorporate interaction terms
# Fit a GAM with interaction between x1 and x2
gam_interact = LinearGAM(te(0, 1)).fit(X_train, y_train)

# Strategy 3: Optimize smooth terms
# Optimize the smooth terms using specialized algorithms (e.g., GCV)
gam_optimized = LinearGAM(s(0, n_splines=25, lam=0.5, penalties='auto')).gridsearch(X_train.values, y_train.values)

# Predict using the GAMs
y_pred_1 = gam_1.predict(X_test)
y_pred_2 = gam_2.predict(X_test)
y_pred_interact = gam_interact.predict(X_test)
y_pred_optimized = gam_optimized.predict(X_test)

# Evaluate models
mse_1 = mean_squared_error(y_test, y_pred_1)
mse_2 = mean_squared_error(y_test, y_pred_2)
mse_interact = mean_squared_error(y_test, y_pred_interact)
mse_optimized = mean_squared_error(y_test, y_pred_optimized)

r2_1 = r2_score(y_test, y_pred_1)
r2_2 = r2_score(y_test, y_pred_2)
r2_interact = r2_score(y_test, y_pred_interact)
r2_optimized = r2_score(y_test, y_pred_optimized)

print("Model 1 - MSE:", mse_1, "R2:", r2_1)
print("Model 2 - MSE:", mse_2, "R2:", r2_2)
print("Model with Interaction - MSE:", mse_interact, "R2:", r2_interact)
print("Optimized Model - MSE:", mse_optimized, "R2:", r2_optimized)

# Plot the true vs. predicted values for the optimized model
plt.scatter(y_test, y_pred_2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs. Predicted (Optimized Model)')
plt.show()

import numpy as np
from pygam import LinearGAM, s
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import make_scorer, mean_squared_error
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
n = 300
x1 = np.random.rand(n) * 10
x2 = np.random.rand(n) * 10
y = 3 * np.sin(x1) + 2 * np.cos(x2) + np.random.normal(scale=2, size=n)

# Create a DataFrame
import pandas as pd
data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

# Split the data into training and test sets
X = data[['x1', 'x2']]
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Perform cross-validated grid search for smoothness parameters
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Grid of smoothness parameters to search
param_grid = {'lam': np.logspace(-3, 3, 10)}

# Scorer using negative mean squared error (to be maximized by GridSearchCV)
scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Fit a LinearGAM with grid search and cross-validation
gam = LinearGAM()  # Initialize a default LinearGAM object
grid_search = GridSearchCV(gam, param_grid, scoring=scorer)
grid_search.fit(X_train, y_train)

# Get the best smoothness parameters
best_lam = grid_search.best_params_['lam']

# Fit a LinearGAM with the best smoothness parameters
best_gam = LinearGAM(lam=best_lam).fit(X_train, y_train)

# Evaluate the best model on the test set
y_pred_best = best_gam.predict(X_test)

# Calculate mean squared error and R-squared
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = best_gam.score(X_test, y_test)

print("Best Smoothness Parameter:", best_lam)
print("Mean Squared Error (Best Model):", mse_best)
print("R-squared (Best Model):", r2_best)

# Plot the true vs. predicted values for the best model
plt.scatter(y_test, y_pred_best)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs. Predicted (Best Model)')
plt.show()

# Local Regression

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(0)
x = np.random.rand(100) * 10
y = 3 * np.sin(x) + np.random.randn(100)

# Sort the data for visualization purposes
sorted_indices = np.argsort(x)
x_sorted = x[sorted_indices]
y_sorted = y[sorted_indices]

# Perform cross-validated grid search for the optimal frac
frac_values = np.linspace(0.05, 1.0, 20)  # Range of frac values to try
kf = KFold(n_splits=5, shuffle=True, random_state=0)
mean_squared_errors = []

for frac in frac_values:
    mse_for_frac = []
    for train_index, test_index in kf.split(x_sorted):
        x_train, x_test = x_sorted[train_index], x_sorted[test_index]
        y_train, y_test = y_sorted[train_index], y_sorted[test_index]

        lowess = sm.nonparametric.lowess(y_train, x_train, frac=frac)
        smoothed_y = np.interp(x_test, lowess[:, 0], lowess[:, 1])
        mse_for_fold = mean_squared_error(y_test, smoothed_y)
        mse_for_frac.append(mse_for_fold)

    mean_mse_for_frac = np.mean(mse_for_frac)
    mean_squared_errors.append(mean_mse_for_frac)

# Find the optimal frac that minimizes mean squared error
optimal_frac = frac_values[np.argmin(mean_squared_errors)]
print("Optimal frac:", optimal_frac)

# Fit the final LOESS model using the optimal frac
lowess = sm.nonparametric.lowess(y_sorted, x_sorted, frac=optimal_frac)
smoothed_y = lowess[:, 1]

# Plot the original data and the smoothed curve
plt.scatter(x_sorted, y_sorted, label='Original Data')
plt.plot(x_sorted, smoothed_y, color='red', label='Smoothed Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('LOESS with Optimal frac')
plt.show()
