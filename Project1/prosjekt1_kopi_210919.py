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
	n = int(len(X[:,0])/k)
	p = 0

	MSE_train = list() # list for storing the MSE train values
	MSE_test = list() # list for storing the MSE test values

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

		p += n

	return np.mean(MSE_train), np.mean(MSE_test)


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


def Create_data(x, y, z, n_x, p, noise=False):

	#Transform from matrices to vectors
	x_vec=np.ravel(x)
	y_vec=np.ravel(y)

	if noise==True:
		n=int(len(x_vec))
		noise_norm = np.random.normal(size=n) # check if 0  random.randn(len(x))
		data=np.ravel(z) + np.random.randn(n) # noise_norm #np.random.random(n)*1
	else:
		data=np.ravel(z)
	return data



n_x = 101  # number of points 
p_degree = 5    # degree of polynomial 

# Creating data values x and y 
x = np.sort(np.random.uniform(0, 1, n_x))
y = np.sort(np.random.uniform(0, 1, n_x))

x, y = np.meshgrid(x,y)
z 	 = FrankeFunction(x,y)

X = CreateDesignMatrix(x,y, n=p_degree)


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
		print('--------------------------------------------------------------------')

		noize = False # planning to have this as an argument input 

		if noize == True:
			print('The data contains a normally distributed noise')
			print("")
			data = Create_data(x, y, z, n_x, p_degree, noise=True)
			model = OrdinaryLeastSquares(data, X)

		else: 
			print('The data does not contain noise')
			print("")
			data = Create_data(x, y, z, n_x, p_degree, noise=False)
			model = OrdinaryLeastSquares(data, X)

		MSE = MeanSquaredError(data, model)
		R_2 = ScoreFunction(data, model)

		print('Mean Square Error: %g ' %MSE)
		print('R^2 Score function: %g' %R_2)

		Plotting(x, y, data, model, p=p_degree, noise=noize)


	elif k_fold_cross_validation == True:
		print('Part b: k-fold cross validation')

		X_train, X_test, data_train, data_test = train_test_split(X, np.ravel(z), shuffle=True, test_size=0.2)
		#betas = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(data_train)
		betas = beta(data_train, X_train)
		
		print("---------")
		print(X_train.shape, data_train.shape)
		print(X_test.shape, data_test.shape)
		print("---------")

		print("Values using train_test_split and my own functions")
		print("--------------------------------------------------")
		ytilde = X_train @ betas
		print("Training MSE")
		print(MeanSquaredError(data_train,ytilde))
		print("Training R2")
		print(ScoreFunction(data_train,ytilde))
	
		ypredict = X_test @ betas
		print("Test MSE")
		print(MeanSquaredError(data_test,ypredict))
		print("Test R2")
		print(ScoreFunction(data_test,ypredict))
		print("")

		print("Values using Scikit's built in functions")
		print("----------------------------------------")
		print("Test MSE")
		print(mean_squared_error(data_test, ypredict))
		print("Test R2")
		print(r2_score(data_test, ypredict))
		print("")

		print("Values using own k-fold function")
		print("--------------------------------")

		cross_val_MSE_train, cross_val_MSE_test = k_fold(np.ravel(z), X, 10)
		print("Train MSE")
		print(cross_val_MSE_train)
		print("Test MSE")
		print(cross_val_MSE_test)



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
