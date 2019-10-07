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


def MeanSquaredError(y_data, y_model):
	"""
	Taking in:
	Returning:
	"""
	n = np.size(y_model)
	MSE = (1/n)*np.sum((y_data-y_model)**2)
	return MSE


def ScoreFunction(y_data, y_model):
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

def OrdinaryLeastSquares(data, X):
	"""
	Taking in:
	Returning:
	"""
	betas = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(data)
	OLS = np.dot(X, betas)
	return OLS

def k_fold(data_set, k, randomize=False):
	"""
	Taking in:
	Returning:
	"""
	
		
	return test_set, train_set



def Plotting(x, y, z, model):
	#franke_x = np.sort(np.random.uniform(0, 1, 1000))
	#franke_y = np.sort(np.random.uniform(0, 1, 1000))
	#franke_x, franke_y = np.meshgrid(franke_x, franke_y)
	#franke_z = FrankeFunction(franke_x,franke_y)

	plot_data = np.reshape(z, (len(x), len(y)))
	plot_model = np.reshape(model, (len(x), len(y)))

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	#ax.scatter(np.ravel(x), np.ravel(y), model, s=5, c='g', marker='o', alpha=1.0)
	#surf = ax.plot_surface(franke_x, franke_y, franke_z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
	#surf = ax.plot_surface(x, y, plot_data, cmap=cm.coolwarm,linewidth=0, antialiased=False)
	surf_model = ax.plot_surface(x, y, plot_model, cmap=cm.coolwarm,linewidth=0, antialiased=False)
	# Customize the z axis.
	ax.set_zlim(-0.10, 1.40)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	# Add a color bar which maps values to colors.
	#fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.show()


n_x = 101  # number of points 
m  	= 5    # degree of polynomial 

# Creating data values x and y 
x = np.sort(np.random.uniform(0, 1, n_x))
y = np.sort(np.random.uniform(0, 1, n_x))

x, y = np.meshgrid(x,y)
z 	 = FrankeFunction(x,y)

#Transform from matrices to vectors
x_vec=np.ravel(x)
y_vec=np.ravel(y)
n=int(len(x_vec))
noise_norm = np.random.normal(size=len(x_vec)) # check if 0  random.randn(len(x))
data=np.ravel(z) #+ np.random.randn(len(x_vec)) # noise_norm #np.random.random(n)*1

X = CreateDesignMatrix(x,y, n=5)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="The Franke function:\
		Regression analysis and resampling methods")

	group = parser.add_mutually_exclusive_group()
	group.add_argument('-a', '--OLS', action="store_true", help="Ordinary Least Squares")
	group.add_argument('-b', '--k_fold', action="store_true", help="K-fold cross validation")
	group.add_argument('-c', '--bias_var_trade', action="store_true", help="Bias-variance tradeoff")
	group.add_argument('-d', '--ridge', action="store_true", help="Ridge regression")
	group.add_argument('-e', '--lasso', action="store_true", help="Lasso regression")
	group.add_argument('-f', '--real_data', action="store_true", help="Introducing real data")
	group.add_argument('-g', '--best_fit', action="store_true", help="Best fit: OLS, Ridge, Lasso")

	if len(sys.argv) <= 1:
		sys.argv.append('--help')

	args = parser.parse_args()

	OLS_method = args.OLS
	k_fold_cross_validation = args.k_fold
	bias_varience_tradeoff = args.bias_var_trade
	Ridge_method = args.ridge
	Lasso_method = args.lasso
	Real_data = args.real_data
	Best_fit = args.best_fit

	if OLS_method == True:
		print('Part a: Ordinary Least Square on The Franke function with resampling')
		print('-----------------------------')
		# create the design matrix

		model = OrdinaryLeastSquares(data, X)

		MSE = MeanSquaredError(data, model)
		print("Mean Square Error: %g " %MSE)

		R_2 = ScoreFunction(data, model)
		print('R^2 Score function: %g' %R_2)

		Plotting(x, y, data, model)

	elif k_fold_cross_validation == True:
		print('Part b: k-fold cross validation')

		X_train, X_test, data_train, data_test = train_test_split(X, np.ravel(z), shuffle=True, test_size=0.2)
		betas = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(data_train)
		print("---------")
		print(X_train.shape, data_train.shape)
		print(X_test.shape, data_test.shape)
		print("---------")

		ytilde = X_train @ betas
		print("Training R2")
		print(ScoreFunction(data_train,ytilde))
		print("Training MSE")
		print(MeanSquaredError(data_train,ytilde))

		ypredict = X_test @ betas
		print("Test R2")
		print(ScoreFunction(data_test,ypredict))
		print("Test MSE")
		print(MeanSquaredError(data_test,ypredict))

		print("Values using Scikit's built in functions")
		print("Test R2")
		print(r2_score(data_test, ypredict))
		print("Test MSE")
		print(mean_squared_error(data_test, ypredict))


		'''
		seed(1)
		print("Values using own k-fold function")
		k_test_X, k_train_X = k_fold(X, 10)
		k_test_data, k_train_data = k_fold(np.ravel(z), 10)
		print(k_train_X[0].shape)    # training
		print(k_train_data[1].shape) # training
		print(k_test_X.shape)    # test
		print(k_test_data.shape) # test
		betas_k = np.linalg.inv((k_train_X[0]).T.dot(k_train_X[0])).dot(k_train_X[0].T).dot(k_train_data[0])

		ytilde_k = k_train_X @ betas_k
		print("Training R2")
		print(ScoreFunction(k_train_data,ytilde_k))
		print("Training MSE")
		print(MeanSquaredError(k_train_data,ytilde_k))

		ypredict_k = k_test_X @ betas_k
		print("Test R2")
		print(ScoreFunction(k_test_data,ypredict_k))
		print("Test MSE")
		print(MeanSquaredError(k_test_data,ypredict_k))
		'''




	elif bias_varience_tradeoff == True:
		print('Part c: bias-varience tradeoff')

	elif Ridge_method == True:
		print('Part d: Ridge Regression on The Franke function with resampling')

	elif Lasso_method == True:
		print('Part e: Lasso Regression on The Franke function with resampling')

	elif Real_data == True:
		print('Part f: Real data')

	elif Best_fit == True:
		print('Part g: Best fit')
