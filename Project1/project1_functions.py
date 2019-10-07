from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.interpolate import spline

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


def MeanSquaredError(y_data, y_model):
	"""
	Taking in:
	Returning:
	"""
	n = np.size(y_model)
	MSE = (1/n)*np.sum((y_data-y_model)**2)
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

	#print("-----------------------------------")
	#print(X)
	#print(X.shape)
	np.random.shuffle(X[::]) # ??????????????
	#np.random.shuffle(data)
	#print(X)
	#print(X.shape)

	n = int(len(X[:,0])/k)    # ??????????????
	#print(n)

	p = 0

	MSE_train = list() # list for storing the MSE train values
	MSE_test = list()  # list for storing the MSE test values

	bias = list()
	variance = list()

	MSE_test_ridge = list()
	MSE_train_ridge = list()

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

		bias.append(Bias(testset_data, ypredict_k))
		variance.append(np.var(ypredict_k))  # +np.std(testset_data)

		
		betas_ridge = Ridge(trainset_data, trainset_X)
		ytilde_k_ridge = trainset_X @ betas_ridge
		ypredict_ridge = testset_X @ betas_ridge
		test_MSE_ridge = MeanSquaredError(testset_data,ypredict_ridge)
		train_MSE_ridge = MeanSquaredError(trainset_data,ytilde_k_ridge)

		MSE_test_ridge.append(test_MSE_ridge)
		MSE_train_ridge.append(train_MSE_ridge)

		p += n

	return np.mean(MSE_train), np.mean(MSE_test), np.mean(bias), np.mean(variance), np.mean(MSE_test_ridge), np.mean(MSE_train_ridge)

def Bias(y, y_model):
	bias = np.mean((y - np.mean(y_model))**2)
	return bias


def Ridge(y, X):
	lamb = np.array([0.01])  # bias
	#lamb = np.linspace(1e-5, 1e-10, np.size(X[0,:]))
	n = np.size(lamb)
	lamb_I = lamb.dot(np.identity(n))
	ridge = np.linalg.inv(X.T.dot(X) + lamb_I).dot(X.T).dot(y)
	#print(lamb_I)
	#print(ridge)
	return ridge 

def Lasso():
	pass


def Plotting(x, y, z, model, p, noise=False):
	#plot_data = np.reshape(z, (len(x), len(y)))

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	plot_model = np.reshape(model, (len(x), len(y)))

	if noise==True:
		''' Plotting the model surface '''
		surf_model = ax.plot_surface(x, y, plot_model, alpha=0.5, cmap=cm.coolwarm,linewidth=0, antialiased=False)
	
	else:
		''' Plotting the model surface with the data points as dots '''
		surf_model = ax.plot_surface(x, y, plot_model, alpha=0.5, cmap=cm.coolwarm,linewidth=0, antialiased=False)
		scatter = ax.scatter(x, y, data, alpha=1, s=0.5, color='black')
		plt.gcf().text(0.0, 0.02, 'The black dots represent the data points', fontsize=14) # Adding text 
	
	# Customize the z axis.
	ax.set_zlim(-0.10, 1.40)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	# Add a color bar which maps values to colors.
	fig.colorbar(surf_model, shrink=0.5, aspect=5)
	plt.title(r'Model: OLS (polynomial of degree $p = %g $)' %p)
	plt.show()


def Create_data(x, y, z, noise=False):

	#Transform from matrices to vectors
	x_vec=np.ravel(x)
	y_vec=np.ravel(y)

	if noise==True:
		n=int(len(x_vec))
		noise_norm = np.random.normal(0,1,size=n) # mean 'center' of distribution
		#noise_norm = np.random.randn(n)
		data=np.ravel(z) + noise_norm # noise_norm #np.random.random(n)*1
	else:
		data=np.ravel(z)
	
	return data