"""
Simple tests to check some of the functions in prosjekt1.py
(We are not familiar with pytest)
"""
import project1_func
import project1
import numpy as np

from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.metrics      import mean_squared_error, r2_score
from random               import seed

np.random.seed(5)       # Same seed as in project 1

n_x      = 100 			# number of points
p_degree = 10  			# degree of polynomial
k        = 5            # k-folds

# Creating data values x and y
x    = np.sort(np.random.uniform(0, 1, n_x))
y    = np.sort(np.random.uniform(0, 1, n_x))
x, y = np.meshgrid(x,y)                                    # Mesh of datapoints

z 	 = project1_func.FrankeFunction(x, y)                  # true function
X    = project1_func.CreateDesignMatrix(x, y, n=p_degree)  # design matrix

#print(z.shape, X.shape)  # (100, 100) (10000, 66)

# Make data, betas and model. For OLS
data = project1_func.Create_data(x, y, z, noise=False)
betas_OLS, model_OLS = project1_func.OrdinaryLeastSquares(data, X)

#------------------------------------------------------------------------------
# We can test MSE and R2 functions against scikit learn:
#------------------------------------------------------------------------------

# Own functions from project1_func, for OLS
MSE_OLS     = project1_func.MeanSquaredError(y_data=np.ravel(z), y_model=model_OLS)
R2_OLS      = project1_func.R2_ScoreFunction(y_data=np.ravel(z), y_model=model_OLS)

# MSE and R2 from scikit learn, for OLS
MSE_sklearn = mean_squared_error(np.ravel(z), model_OLS)
R2_sklearn  = r2_score(np.ravel(z), model_OLS)

def test_MSE():
    assert MSE_OLS == MSE_sklearn, \
    print("Our own MSE function is not equal to the scikit learn MSE. MSE_own=%s, MSE_sklearn=%s" %(MSE_OLS, MSE_sklearn))
    print("Our own MSE function is equal to the scikit learn MSE")

def test_R2():
    assert R2_OLS == R2_sklearn, \
    print("Our own R2 function is not equal to the scikit learn R2. R2_own=%s, R2_sklearn=%s" %(R2_OLS, R2_sklearn))
    print("Our own R2 function is equal to the scikit learn R2")

#------------------------------------------------------------------------------
# We know that Ridge and Lasso for lambda=0, gives OLS:
#------------------------------------------------------------------------------

# Computing MSE for Ridge with lambda = 0
beta_ridge   = project1_func.beta(np.ravel(z), X, method='Ridge', l=0)
model_ridge  = np.dot(X, beta_ridge)
MSE_Ridge    = project1_func.MeanSquaredError(np.ravel(z), model_ridge)

def test_ridge():
    assert MSE_Ridge == MSE_OLS, \
    print('Ridge with lambda=0 is not equal to OLS. MSE_Ridge=%s, MSE_OLS=%s' %(MSE_Ridge, MSE_Lasso))
    print('The MSE for Ridge with lambda=0 is equal to MSE for OLS')

"""
# Computing MSE for Lasso with lambda = 0

#   alpha = 0 is equivalent to an ordinary least square, solved by the LinearRegression object.
#   For numerical reasons, using alpha = 0 with the Lasso object is not advised.

lasso = Lasso(max_iter = 1e4, tol=0.00001, normalize = True, fit_intercept=False)
#lasso.set_params(alpha=0)
#lasso = Lasso(alpha=0, fit_intercept=False)
betas_lasso = lasso.fit(X, np.ravel(z)).coef_                 # Noe galt her! Dims
model_lasso = np.dot(X, betas)
#model_lasso = lasso.predict(np.ravel(z))

MSE_Lasso   = project1_func.MeanSquaredError(np.ravel(z), model_lasso)

def test_lasso():
    assert MSE_Lasso == MSE_OLS, \
    print('Lasso with lambda=0 is not equal to OLS. MSE_Lasso=%s, MSE_OLS=%s' %(MSE_Lasso, MSE_OLS))
    print('The MSE for Lasso with lambda=0 is equal to MSE for OLS')
"""
