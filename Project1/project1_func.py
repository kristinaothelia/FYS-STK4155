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


def FrankeFunction(x,y):
	"""
	Taking in:
	Returning: 
	"""
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4 

def CreateDesignMatrix(x, y, n):
	"""
	Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
	Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
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

	l = l

	return X


def MeanSquaredError(y_data, y_model):
	"""
	Taking in:
	Returning:
	"""
	n = np.size(y_model)
	MSE = (1/n)*np.sum((y_data-y_model)**2)
	#MSE = np.mean((y_data-(y_model)**2))
	return MSE


def R2_ScoreFunction(y_data, y_model):
	"""
	Taking in:
	Returning:
	"""
	counter = np.sum((y_data-y_model)**2)
	denominator = np.sum((y_data-np.mean(y_data))**2)
	R_2 = 1 - (counter/denominator)
	return R_2

def RelativeError(y_data, y_model):
	"""
	Taking in:
	Returning:
	"""
	error = abs((y_data-y_model)/y_data)
	return error

def beta(data, X, method=None, l=None):

	if method == None:
		print("No method given, use OLS, ridge or lasso")
		sys.exit()

	if method == 'OLS':
		betas = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(data)
		#betas = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,data))

	elif method == "Ridge":
		lamb = l #np.zeros(1)+l  # bias
		n = np.size(X[0,:])
		lamb_I = lamb*(np.identity(n))
		betas = np.linalg.inv(X.T.dot(X) + lamb_I).dot(X.T).dot(data)
	return betas

def ConfidenceInterval():
	pass

def OrdinaryLeastSquares(data, X):
	"""
	Taking in:
	Returning:
	"""
	betas = beta(data, X, method='OLS')
	OLS   = np.dot(X, betas)
	return betas, OLS
'''
def k_fold(data, X, k, method=None, l=None, shuffle=False):
	"""
	Taking in:
	Returning:
	"""

	#np.random.shuffle(X[::]) # ??????????????

	if method == None:
		print("No method given, use 'OLS', 'Ridge' or 'Lasso'")
		sys.exit()

	#np.random.shuffle(X)
	n = int(len(X[:,0])/k)    # ??????????????

	p = 0

	MSE_train = list() # list for storing the MSE train values
	MSE_test = list()  # list for storing the MSE test values

	bias = list()
	variance = list()

	MSE_test_ridge = list()
	MSE_train_ridge = list()

	indexs = np.arange(len(np.ravel(data)))

	if shuffle:
		np.random.shuffle(indexs)

	folds = np.array_split(indexs, k)

	for i in range(k):
		X_test = np.copy(X[folds[i]])
		z_test = np.copy(np.ravel(data)[folds[i]])

		X_train = np.delete(np.copy(X), folds[i], axis = 0)
		z_train = np.delete(np.copy(np.ravel(data)), folds[i])


		#p = int(p)
		#index = range(p, p+n)

		#testset_data  = data[index]                 # the index row 
		#trainset_data = np.delete(data, index)

		#testset_X  = X[index, :]
		#trainset_X = np.delete(X, index, 0)

		if method == 'OLS':
			betas = beta(z_train, X_train, method='OLS')
			ytilde  = X_train @ betas
			ypredict = X_test @ betas
		elif method == 'Ridge':
			betas = beta(z_train, X_train, method='Ridge', l=l)
			ytilde  = X_train @ betas
			ypredict = X_test @ betas
		elif method == 'Lasso':
			lasso = Lasso(max_iter = 1e3, tol=0.001, normalize = False)
			scaler = StandardScaler()
			lasso.set_params(alpha=l)
			print(type(l))
			lasso.fit(X_train, z_train)
			betas = lasso.coef_
			ytilde = lasso.predict(X_train)
			ypredict = lasso.predict(X_test)
		else: 
			print('no method')


		train_MSE = MeanSquaredError(z_train,ytilde)

		test_MSE = mean_squared_error(z_test,ypredict) #MeanSquaredError(testset_data,ypredict)

		MSE_train.append(train_MSE)
		MSE_test.append(test_MSE)

		bias.append(Bias(z_test, ypredict))
		variance.append(np.var(ypredict)+np.std(z_test))  # +np.std(testset_data)

		p += n

	MSE_train = np.mean(MSE_train)
	MSE_test = np.mean(MSE_test)
	bias = np.mean(bias)
	variance = np.mean(variance)

	return MSE_train, MSE_test, bias, variance
'''

def k_fold(data, X, k, index, method=None, l=None, shuffle=False):
	"""
	Taking in:
	Returning:
	"""

	if method == None:
		print("No method given, use 'OLS', 'Ridge' or 'Lasso'")
		sys.exit()

	#np.random.shuffle(X)
	n = int(len(X[:,0])/k)    # ??????????????

	MSE_train = list() # list for storing the MSE train values
	MSE_test = list()  # list for storing the MSE test values

	bias = list()
	variance = list()

	if shuffle:
		np.random.shuffle(index)
	folds = np.array_split(index, k)

	for i in range(len(folds)):
		X_test = np.copy(X[folds[i]])
		z_test = np.copy(np.ravel(data)[folds[i]])

		X_train = np.delete(np.copy(X), folds[i], axis = 0)
		z_train = np.delete(np.copy(np.ravel(data)), folds[i])

		if method == 'OLS':
			betas = beta(z_train, X_train, method='OLS')
			ytilde  = X_train @ betas
			ypredict = X_test @ betas
		elif method == 'Ridge':
			betas = beta(z_train, X_train, method='Ridge', l=l)
			ytilde  = X_train @ betas
			ypredict = X_test @ betas
		elif method == 'Lasso':
			lasso = Lasso(max_iter = 1e2, tol=0.001, normalize = True) #fit_intercept=False
			scaler = StandardScaler()
			lasso.set_params(alpha=l)
			#print(type(l))
			lasso.fit(X_train, z_train)
			beta = lasso.coef_
			ytilde = lasso.predict(X_train)
			ypredict = lasso.predict(X_test)
		else: 
			print('no method')


		train_MSE = MeanSquaredError(z_train,ytilde)
		test_MSE = MeanSquaredError(z_test,ypredict) #MeanSquaredError(testset_data,ypredict)
		MSE_train.append(train_MSE)
		MSE_test.append(test_MSE)

		bias.append(Bias(z_test, ypredict))
		variance.append(np.var(ypredict))   #+np.std(z_test))  # +np.std(testset_data)  #ddof=0

	MSE_train = np.mean(MSE_train)
	MSE_test = np.mean(MSE_test)
	bias = np.mean(bias)
	variance = np.mean(variance)

	return MSE_train, MSE_test, bias, variance


def Bias(y, y_model):
	bias = np.mean((y - np.mean(y_model))**2)
	return bias



def Create_data(x, y, z, noise=False):

	#Transform from matrices to vectors
	x_vec=np.ravel(x)
	y_vec=np.ravel(y)

	if noise==True:
		n=(len(x_vec))
		noise_norm = np.random.normal(0,1,size=n) # mean 'center' of distribution
		#noise_norm = np.random.randn(n)
		data= np.ravel(z) + noise_norm # noise_norm #np.random.random(n)*1  # reshape?
	else:
		data=np.ravel(z)
	
	return data


def CI(z, X, beta, model, method=False, dataset=False):

	if method == False:
		print('U need to pass a method, use "OLS", "Ridge" or "Lasso"')

	if len(z.shape) > 1:
		z = np.ravel(z)
		model = np.ravel(model)

	n = len(z)

	sigma_squared = sum((z - model)**2)/(n - len(beta) - 1)
	sigma         = np.sqrt(sigma_squared)

	XTX_inv       = np.linalg.inv(np.dot(X.T, X))
	var_beta      = sigma_squared*XTX_inv

	Z_score = 1.96

	px = np.linspace(0,1,len(beta))

	con_int = np.zeros((len(beta), 2))

	for i in range(len(beta)):
		con_int[i,0] = beta[i] - Z_score*np.sqrt(XTX_inv[i,i])*sigma
		con_int[i,1] = beta[i] + Z_score*np.sqrt(XTX_inv[i,i])*sigma

		print(beta[i], con_int[i, 0], con_int[i,1])

	#plt.plot(px, beta, 'ro')
	plt.errorbar(px, beta, yerr=con_int[i,0]+con_int[i,1] ,fmt='.k', ecolor='grey', capsize=3)
	plt.xlabel('i', fontsize=16)
	plt.ylabel('$\\beta_i$',fontsize=16)
	plt.title('The confidence intervals of the parameters $\\beta$ \n Method: %s (%s)' %(method, dataset), fontsize=16)
	plt.savefig('Results/CI_%s_%s' %(dataset, method))
	plt.show()


'''
		# uncertainty lines (95% confidence)
		plt.plot(px, con_int[i,0], c='orange',label='95% Confidence Region')
		plt.plot(px, con_int[i,1], c='orange')

	# plot the regression
	plt.plot(px, nom, c='black', label='y=a x + b')

	# prediction band (95% confidence)
	plt.plot(px, lpb, 'k--',label='95% Prediction Band')
	plt.plot(px, upb, 'k--')
	plt.ylabel('y')
	plt.xlabel('x')
	plt.legend(loc='best')

	# save and show figure
	plt.savefig('regression.png')
	plt.show()
'''