"""
FYS-STK4155 - Function codes for project 1
"""
import sys
import numpy                 as np
import matplotlib.pyplot     as plt
import sklearn.linear_model  as skl

from mpl_toolkits.mplot3d    import Axes3D
from matplotlib.ticker       import LinearLocator, FormatStrFormatter
from matplotlib              import cm
from random                  import randrange, seed
from sklearn.metrics         import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import Lasso, LassoCV, LinearRegression
from sklearn.preprocessing   import normalize
from scipy.interpolate       import spline

from sklearn.preprocessing   import StandardScaler


def FrankeFunction(x, y):
	"""
	Function to create data with Franke function
	Input x, y		| Data points in x- and y direction
	"""
	term1 =  0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 =  0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 =  0.50*np.exp(-(9*x-7)**2/4.0    - 0.25*((9*y-3)**2))
	term4 = -0.20*np.exp(-(9*x-4)**2        - (9*y-7)**2)
	return term1 + term2 + term3 + term4

def Create_data(x, y, z, noise=False):
	"""
	Function that adds noise to data if wanted, and returns data (z)
	as np.ravel(z)
	"""
	#Transform from matrices to vectors
	x_vec=np.ravel(x)
	y_vec=np.ravel(y)

	if noise==True:
		n          = (len(x_vec))
		noise_norm = np.random.normal(0,1,size=n)  # mean 'center' of distribution
		data       = np.ravel(z) + noise_norm
	else:
		data       = np.ravel(z)

	return data

def CreateDesignMatrix(x, y, n):
	"""
	Function for creating a design X-matrix,
	with rows [1, x, y, x^2, xy, xy^2 , etc.]
	Input x, y 		| Datpoints x and y after meshgrid
	Input n			| The degree of the polynomial you want to fit
	"""
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i*(i+1)/2))
		for j in range(i+1):
			X[:,q+j] = x**(i-j) * y**j

	return X

def MeanSquaredError(y_data, y_model):
	"""
	Function to calculate the mean squared error (MSE) for our model
	Input y_data	| Function array
	Input y_model	| Predicted function array
	"""
	n   = np.size(y_model)
	MSE = (1/n)*np.sum((y_data-y_model)**2)

	return MSE

def R2_ScoreFunction(y_data, y_model):
	"""
	Function to calculate the R2 score for our model
	Input y_data	| Function array
	Input y_model	| Predicted function array
	"""
	counter     = np.sum((y_data-y_model)**2)
	denominator = np.sum((y_data-np.mean(y_data))**2)
	R_2          = 1 - (counter/denominator)

	return R_2

def Bias(y_data, y_model):
	"""
	Function for calculating the Bias**2
	Input y_data	| Function array
	Input y_model	| Predicted function array
	"""
	bias = np.mean((y_data - np.mean(y_model))**2)
	return np.sqrt(bias)

# Brukes?
def RelativeError(y_data, y_model):
	"""
	Taking in:
	Returning:
	"""
	error = abs((y_data-y_model)/y_data)
	return error

def beta(data, X, method=None, l=None):
	"""
	Function for calculating OLS and Ridge Beta values, matrix inversion
	Input data 		| Input function, datapoints
	Input X			| Design matrix
	Input l 		| Lambda
	"""
	if method == None:
		print("No method given, use 'OLS' or 'Ridge'")
		sys.exit()

	if method == 'OLS':
		betas = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(data)

	elif method == 'Ridge':
		lamb   = l
		n      = np.size(X[0,:])
		lamb_I = lamb*(np.identity(n)) # Indentity matrix*lambda
		betas  = np.linalg.inv(X.T.dot(X) + lamb_I).dot(X.T).dot(data)
	return betas

# Unodvendig?
def OrdinaryLeastSquares(data, X):
	"""
	Taking in:
	Returning:
	"""
	betas = beta(data, X, method='OLS')
	OLS   = np.dot(X, betas)
	return betas, OLS


def k_fold(data, X, k, index, method=None, l=None, shuffle=False):
	"""
	Function for k-fold CV. MSE_train, MSE_test, bias, variance
	Input data 		| Input function
	Input X			| Design matrix
	Input k 		| k-folds
	Input index 	| index for makinf folds. folds = np.array_split(index, k)
	Input l 		| Lambda
	"""

	if method == None:
		print("No method given, use 'OLS', 'Ridge' or 'Lasso'")
		sys.exit()

	n         = int(len(X[:,0])/k)

	MSE_train = list()  # List for storing the MSE train values
	MSE_test  = list()  # List for storing the MSE test values
	R2_train  = list()  # List for storing the R2 train values
	R2_test   = list()  # List for storing the R2 test values
	bias      = list()  # List for storing the bias values
	variance  = list()  # List for storing the variance values

	if shuffle:
		np.random.shuffle(index)

	folds = np.array_split(index, k)

	for i in range(len(folds)):

		X_test  = np.copy(X[folds[i]])
		z_test  = np.copy(np.ravel(data)[folds[i]])

		X_train = np.delete(np.copy(X), folds[i], axis = 0)
		z_train = np.delete(np.copy(np.ravel(data)), folds[i])

		# Make beta values, ytilde and ypredict
		if method == 'OLS':
			betas    = beta(z_train, X_train, method='OLS')
			ytilde   = X_train @ betas
			ypredict = X_test @ betas

		elif method == 'Ridge':
			betas    = beta(z_train, X_train, method='Ridge', l=l)
			ytilde   = X_train @ betas
			ypredict = X_test @ betas

		elif method == 'Lasso':
			lasso    = Lasso(max_iter = 1e2, tol=0.001, normalize=True, fit_intercept=False)
			scaler   = StandardScaler()
			lasso.set_params(alpha=l)
			lasso.fit(X_train, z_train)
			betas     = lasso.coef_
			ytilde   = lasso.predict(X_train)
			ypredict = lasso.predict(X_test)

		else:
			print('No method')

		train_MSE = MeanSquaredError(z_train,ytilde)
		test_MSE  = MeanSquaredError(z_test,ypredict)
		train_R2  = R2_ScoreFunction(z_train,ytilde)
		test_R2   = R2_ScoreFunction(z_test,ypredict)

		# Append values
		MSE_train.append(train_MSE)
		MSE_test.append(test_MSE)
		R2_train.append(train_R2)
		R2_test.append(test_R2)
		bias.append(Bias(z_test, ypredict))
		variance.append(np.mean(np.var(ypredict)))

	MSE_train = np.mean(MSE_train)
	MSE_test  = np.mean(MSE_test)
	R2_train  = np.mean(R2_train)
	R2_test   = np.mean(R2_test)
	bias      = np.mean(bias)
	variance  = np.mean(variance)

	return MSE_train, MSE_test, bias, variance, R2_train, R2_test

def CI(z, X, beta, model, p, method=False, dataset=False, plot=False):
	"""
	Function for calculating the 95% confidence interval
	Input z		| Input data
	Input X		| Design matrix
	Input beta 	| Beta values
	Input model | Input function/model
	"""
	if method == False:
		print('U need to pass a method, use "OLS", "Ridge" or "Lasso"')

	if len(z.shape) > 1:
		z     = np.ravel(z)
		model = np.ravel(model)

	n             = len(z)
	# A 95% confidence interval is 1.96
	Z_score       = 1.96

	sigma_squared = sum((z - model)**2)/(n - len(beta) - 1)
	sigma         = np.sqrt(sigma_squared)

	XTX_inv       = np.linalg.inv(np.dot(X.T, X))
	var_beta      = sigma_squared*XTX_inv
	beta_i        = np.linspace(0, 1, len(beta))
	con_int       = np.zeros((len(beta), 2))  # Plot
	CI_list       = np.zeros((len(beta), 3))  # List

	if plot == True:

		for i in range(len(beta)):
			con_int[i,0] = beta[i] - Z_score*np.sqrt(XTX_inv[i,i])*sigma
			con_int[i,1] = beta[i] + Z_score*np.sqrt(XTX_inv[i,i])*sigma

		# Plot the confidence interval as error bars
		plt.errorbar(beta_i, beta, yerr=con_int[i,0]+con_int[i,1] ,fmt='.k', ecolor='grey', capsize=3)
		plt.xlabel('i', fontsize=16)
		plt.ylabel('$\\beta_i$',fontsize=16)
		plt.title('The confidence intervals of the parameters $\\beta$ \n Method: %s (%s)' %(method, dataset), fontsize=16)
		plt.savefig('Results/CI_%s_%s' %(dataset, method))

	else:

		for i in range(len(beta)):
			CI_list[i,0] = beta[i]
			CI_list[i,1] = beta[i] - Z_score*np.sqrt(XTX_inv[i,i])*sigma
			CI_list[i,2] = beta[i] + Z_score*np.sqrt(XTX_inv[i,i])*sigma
		return CI_list
