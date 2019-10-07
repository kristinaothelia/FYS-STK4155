import project1_func
import project1
import numpy as np
from sklearn.linear_model    import Lasso, LassoCV, LinearRegression

n_x      = 100 			# number of points 
p_degree = 10  			# degree of polynomial 
k        = 5 

# Creating data values x and y 
x    = np.sort(np.random.uniform(0, 1, n_x))
y    = np.sort(np.random.uniform(0, 1, n_x))

x, y = np.meshgrid(x,y)
z 	 = project1_func.FrankeFunction(x,y)  # true function 


X = project1_func.CreateDesignMatrix(x,y, n=p_degree)

data = project1_func.Create_data(x, y, z, noise=False)
betas_OLS, model_OLS = project1_func.OrdinaryLeastSquares(data, X)

# Computing MSE for OLS 
MSE_OLS = project1_func.MeanSquaredError(np.ravel(z), model_OLS)

# Computing MSE for Ridge with lambda = 0
beta_ridge = project1_func.beta(np.ravel(z), X, method='Ridge', l=0)
model_ridge  = X @ beta_ridge
MSE_Ridge = project1_func.MeanSquaredError(np.ravel(z), model_ridge)


# Lage test som sjekker vaar mot sklern MSE og R2. 


def test_ridge():
    assert MSE_Ridge == MSE_OLS, \
    print('Ridge with lambda=0 is not equal to OLS. MSE_Ridge=%s, MSE_OLS=%s' %(MSE_OLS, MSE_Ridge))
    print('The MSE for Ridge with lambda=0 is equal to MSE for OLS')



'''
# Computing MSE for Lasso with lambda = 0
lasso = Lasso(max_iter = 1e3, tol=0.0001, normalize = True)
lasso.set_params(alpha=0)
lasso.fit(X, z)
betas_lasso = lasso.coef_
model_lasso = lasso.predict(z)

def test_lasso():
	pass
'''