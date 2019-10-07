from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import project1_func
import sys

def plot_3D(x, y, z, string=''):

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	if len(z.shape) < 2:
		z = np.reshape(z, (len(x[:,0]), len(y[0,:])))

	surf = ax.plot_surface(x, y, z, alpha=0.5, cmap=cm.coolwarm,linewidth=0, antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.title('%s' %string)


def plot_terrain(x,y,terrain,string=''):

	if len(terrain.shape) < 2:
		terrain = np.reshape(terrain, (len(x[:,0]), len(y[0,:])))

	plt.title('%s' %string)
	plt.imshow(terrain, cmap='gray', alpha=1)
	plt.colorbar(shrink=0.5, aspect=5)
	plt.xlabel('X')
	plt.ylabel('Y')


def MSE_test_train(x, y, data, k, p_degree, method='OLS', biasvariance=False):

	MSE_train = np.zeros(p_degree)
	MSE_test = np.zeros(p_degree)
	bias = np.zeros(p_degree)
	variance = np.zeros(p_degree)
	MSE_train_check = np.zeros(p_degree)
	MSE_test_check = np.zeros(p_degree)

	# shuffle !!!!!!!!!!!!!!!!!!!!!!!!!!!!

	index = np.arange(len(np.ravel(data)))

		#if shuffle:
		#	np.random.shuffle(index)

	for degree in range(0, p_degree):

		if method == 'OLS':
			X = project1_func.CreateDesignMatrix(x,y, n=degree)
			MSE_train[degree], MSE_test[degree], bias[degree], variance[degree] =  project1_func.k_fold(data, X, k, index, method='OLS')
		if method == 'Ridge':
			X = project1_func.CreateDesignMatrix(x,y, n=degree)
			MSE_train[degree], MSE_test[degree], bias[degree], variance[degree] =  project1_func.k_fold(data, X, k, index, method='Ridge', l=0.0011)
		if method == 'Lasso':
			X = project1_func.CreateDesignMatrix(x,y, n=degree)
			MSE_train[degree], MSE_test[degree], bias[degree], variance[degree] =  project1_func.k_fold(data, X, k, index, method='Lasso', l=1.7e-5)

	print("bias+variance")
	print(bias+variance)
	print("Train MSE")
	print(MSE_train)
	print("Test MSE")
	print(MSE_test)
	print("Bias")
	print(bias)
	print("Variance")
	print(variance)

	complexity = np.arange(0,p_degree)

	if biasvariance == True:
		plt.plot(complexity, MSE_test, label='testing')
		plt.plot(complexity, MSE_train, label='training')
		plt.plot(complexity, bias, label="bias")
		plt.plot(complexity, variance, label="variance")
		#plt.plot(MSE_test+MSE_train)
		#plt.plot(bias+variance, label='biasvar')
		#plt.plot((bias+variance)-np.std(MSE_test), 'ro')
		plt.title('Bias-variance tradeoff')
		plt.legend()
	else:
		plt.plot(complexity, MSE_test, label='testing')
		plt.plot(complexity, MSE_train, label='training')
		plt.title('MSE')
		plt.legend()
'''
def plot_MSE_lambda(x, y, data, k, degree_start, degree_stop, lambdas, method=None):

	if method == None:
		print('you need to pass a method')
		sys.exit()

	for degree in range(degree_start, degree_stop+1):



			train_MSE = np.zeros(len(lambdas))
			test_MSE = np.zeros(len(lambdas))
			bias = np.zeros(len(lambdas))
			variance = np.zeros(len(lambdas))

			for l in range(0, len(lambdas)):

				X = project1_func.CreateDesignMatrix(x,y, n=degree)
				if method == 'Ridge':
					train_MSE[l], test_MSE[l], bias[l], variance[l] = project1_func.k_fold(data, X, 10, method='Ridge', l=lambdas[l], shuffle=False) 
				elif method == 'Lasso':
					train_MSE[l], test_MSE[l], bias[l], variance[l] = project1_func.k_fold(data, X, 10, method='Lasso', l=lambdas[l], shuffle=False) 

			lambda_min = np.argmin(test_MSE)
			plt.plot(lambdas, test_MSE, label='p = %g' %degree)
			plt.plot(lambdas[lambda_min], test_MSE[lambda_min], 'ro')
			print('for degree=%g, best alpha=%g' %(degree, lambdas[lambda_min]))

	#plt.imshow(grid, origin='lower')
	plt.legend()
	plt.show()
'''
def plot_MSE_lambda(x, y, data, k, degree_start, degree_stop, lambdas, method=None, shuffle=False):

	if method == None:
		print('you need to pass a method')
		sys.exit()

	for degree in range(degree_start, degree_stop+1):

		index = np.arange(len(np.ravel(data)))

		if shuffle:
			np.random.shuffle(index)

		train_MSE = np.zeros(len(lambdas))
		test_MSE = np.zeros(len(lambdas))
		bias = np.zeros(len(lambdas))
		variance = np.zeros(len(lambdas))

		for l in range(0, len(lambdas)):

			X = project1_func.CreateDesignMatrix(x,y, n=degree)
			if method == 'Ridge':
				train_MSE[l], test_MSE[l], bias[l], variance[l] = project1_func.k_fold(data, X, 10, index, method='Ridge', l=lambdas[l]) 
			elif method == 'Lasso':
				train_MSE[l], test_MSE[l], bias[l], variance[l] = project1_func.k_fold(data, X, 10, index, method='Lasso', l=lambdas[l]) 

		lambda_min = np.argmin(test_MSE)
		plt.plot(lambdas, test_MSE, label='p = %g' %degree)
		plt.plot(lambdas[lambda_min], test_MSE[lambda_min], 'ro')
		print('for degree=%g, best alpha=%g' %(degree, lambdas[lambda_min]))

	plt.legend()


def Plotting(x, y, z, model, p, noise=False):
	#plot_data = np.reshape(z, (len(x), len(y)))

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	#plot_model = np.reshape(model, len(x), len(y))
	plot_model = np.reshape(model, (len(x[:,0]), len(y[0,:])))
	print(plot_model)

	if noise==True:
		''' Plotting the model surface '''
		surf_model = ax.plot_surface(x, y, plot_model, alpha=0.5, cmap=cm.coolwarm,linewidth=0, antialiased=False)
	
	else:
		''' Plotting the model surface with the data points as dots '''
		surf_model = ax.plot_surface(x, y, plot_model, alpha=0.5, cmap=cm.coolwarm,linewidth=0, antialiased=False)
		scatter = ax.scatter(x, y, z, alpha=1, s=0.5, color='black')
		plt.gcf().text(0.0, 0.02, 'The black dots represent the data points', fontsize=14) # Adding text 
	
	# Customize the z axis.
	#ax.set_zlim(-0.10, 1.40)
	#ax.zaxis.set_major_locator(LinearLocator(10))
	#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	# Add a color bar which maps values to colors.
	fig.colorbar(surf_model, shrink=0.5, aspect=5)
	plt.title(r'Model: OLS (polynomial of degree $p = %g $)' %p)
	plt.show()


def plot_both(x, y, model, string=''):

	fig = plt.figure() #(figsize=plt.figaspect(0.5))
	plt.suptitle('Model %s' %(string))

	#fig = plt.figure()

	ax = fig.add_subplot(1,2,1, projection='3d')
	plt.title('Terrain')
	plt.imshow(plot_model, cmap='gray', alpha=1)
	#plt.colorbar(shrink=0.5, aspect=5)
	ax.set_xlabel('x')
	ax.set_ylabel('y')


	if model.size < 1:
		model = np.reshape(model, (len(x), len(y)))

	
	ax = fig.add_subplot(1,2,2, projection='3d')
	surf = ax.plot_surface(x, y, model, alpha=0.5, cmap=cm.coolwarm,linewidth=0, antialiased=False)
	#fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.title('3D terrain')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	plt.show()
		