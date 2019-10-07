from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import numpy as np
from random import randrange, seed
import argparse
import sys

def FrankeFunction(x,y):
	"""
	Taking in:
	Returning: 
	"""
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	#noise = np.random.normal(size=len(x))
	return term1 + term2 + term3 + term4 #+ noise

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
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = x**(i-k) * y**k

	return X

def Create_data(x, y, z, noise=False):

	#Transform from matrices to vectors
	x_vec=np.ravel(x)
	y_vec=np.ravel(y)

	if noise==True:
		n=int(len(x_vec))
		#noise_norm = np.random.normal(size=n) # mean 'center' of distribution
		noise_norm = np.random.randn(n)
		data=np.ravel(z) + noise_norm # noise_norm #np.random.random(n)*1
	else:
		data=np.ravel(z)
	
	return data

def MeanSquaredError(y_data, y_model):
	"""
	Taking in:
	Returning:
	"""
	n = np.size(y_model)
	MSE = (1/n)*np.sum((y_data-y_model)**2)
	return MSE

def beta(data, X):

	betas = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(data)
	return betas

def OrdinaryLeastSquares(data, X):
	"""
	Taking in:
	Returning:
	"""
	betas = beta(data, X)
	OLS   = np.dot(X, betas)
	return OLS

def k_fold(data, X, k):
	"""
	Taking in:
	Returning:
	"""

	#X = np.random.shuffle(X)

	n = int(len(X[:,0])/k)
	p = 0

	MSE_train = list() # list for storing the MSE train values
	MSE_test = list()  # list for storing the MSE test values

	bias = list()
	variance = list()

	tradeoff_training = list()

	for i in range(k):
		p = int(p)
		index = range(p, p+n)

		testset_data  = data[index]                 # the index row 
		trainset_data = np.delete(data, index)

		testset_X  = X[index, :]
		trainset_X = np.delete(X, index, 0)

		betas_k = beta(trainset_data, trainset_X)

		ytilde_k  = trainset_X @ betas_k
		train_MSE = MeanSquaredError(trainset_data,ytilde_k)

		ypredict_k = testset_X @ betas_k
		test_MSE   = MeanSquaredError(testset_data,ypredict_k)

		MSE_train.append(train_MSE)
		MSE_test.append(test_MSE)


		#bias.append(Bias(testset_X, ypredict_k))
		#variance.append(np.var(ypredict_k))  # +np.std(testset_data)
		#tradeoff_training.append(MeanSquaredError(trainset_data,ytilde_k))

		bias.append(Bias(testset_data, ypredict_k))
		variance.append(np.var(ypredict_k))  # +np.std(testset_data)
		

		tradeoff_training.append(train_MSE)
		#tradeoff_testing.append(bias+variance)

		p += n

	return np.mean(MSE_train), np.mean(MSE_test), np.mean(bias), \
	np.mean(variance), np.mean(tradeoff_training)

def Bias(y, y_model):

	bias = np.mean((y - np.mean(y_model))**2)
	
	return bias



n_x = 801       # number of points 
p_degree = 10 # degree of polynomial 

# Creating data values x and y 
x = np.sort(np.random.uniform(0, 1, n_x))
y = np.sort(np.random.uniform(0, 1, n_x))

x, y = np.meshgrid(x,y)
z 	 = FrankeFunction(x,y)

X = CreateDesignMatrix(x,y, n=p_degree)

data = Create_data(x, y, z)

MSE_train = np.zeros(p_degree)
MSE_test = np.zeros(p_degree)
bias = np.zeros(p_degree)
variance = np.zeros(p_degree)
tradeoff_training = np.zeros(p_degree)
tradeoff_testing = np.zeros(p_degree)


for degree in range(0, p_degree):

	X = CreateDesignMatrix(x,y, n=degree)
	MSE_train[degree], MSE_test[degree], bias[degree], variance[degree], tradeoff_training[degree] \
	=  k_fold(data, X, 10)
	#tradeoff_testing[degree] = bias[degree]+variance[degree]
	tradeoff_testing[degree] = MSE_test[degree]

print(MSE_train)
print(MSE_test)
print(bias)
print(variance)
print(tradeoff_training)
print(tradeoff_testing)

complexity = np.arange(0,p_degree)
print(complexity)

plt.plot(complexity, (tradeoff_testing), label='testing')
plt.plot(complexity, (tradeoff_training), label='training')
plt.legend()
plt.show()

'''
x_new = np.linspace(complexity.min(),complexity.max(),1000, endpoint=True)

#smooth_test = spline(complexity,MSE_test, x_new, order=3)
#smooth_train = spline(complexity,MSE_train, x_new, order=3)
'''


#python -c "import scipy; print(scipy.__version__)"

#X_train, X_test, data_train, data_test = train_test_split(X, np.ravel(z), shuffle=True, test_size=0.2)

'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

np.random.seed(2018)

n = 40
n_boostraps = 100
maxdegree = 14


x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)

error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

n_boostraps = 10

for degree in range(maxdegree):
	model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))
	y_pred = np.empty((y_test.shape[0], n_boostraps))

	for i in range(n_boostraps):
		x_, y_ = resample(x_train, y_train)
		y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()

		polydegree[degree] = degree
		error[degree] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True))
		bias[degree] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2)
		variance[degree] = np.mean( np.var(y_pred, axis=1, keepdims=True))


	print('Polynomial degree:', degree)
	print('Error:', error[degree])
	print('Bias^2:', bias[degree])
	print('Var:', variance[degree])
	print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='bias')
plt.plot(polydegree, variance, label='Variance')
plt.legend()
plt.show()

'''