from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import project1_func
import numpy as np
import matplotlib.pyplot as plt
from random import randrange, seed
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
import argparse
import sys

np.random.seed(2019)

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

		n_x = 100      # number of points 
		p_degree = 10  # degree of polynomial

		# confidence interval on the betas

		# Creating data values x and y 
		x = np.sort(np.random.uniform(0, 1, n_x))
		y = np.sort(np.random.uniform(0, 1, n_x))

		x, y = np.meshgrid(x,y)
		z 	 = project1_func.FrankeFunction(x,y)  # true function 

		X = project1_func.CreateDesignMatrix(x,y, n=p_degree)

		noize = False  # planning to have this as an argument input 

		if noize == True:
			print('The data contains a normally distributed noise')
			print("")
			data = project1_func.Create_data(x, y, z, noise=True)
			print(data.shape, X.shape)
			model = project1_func.OrdinaryLeastSquares(data, X)

		else: 
			print('The data does not contain noise')
			print("")
			data = project1_func.Create_data(x, y, z, noise=False)
			model = project1_func.OrdinaryLeastSquares(data, X)

		# Computing the MSE and R2 score between the true data and the model
		MSE = project1_func.MeanSquaredError(np.ravel(z), model)
		R_2 = project1_func.R2_ScoreFunction(np.ravel(z), model)

		print('Mean Square Error:  ' ,MSE)
		print('R^2 Score function: ' ,R_2)

		print("")
		print("Values using Scikit's built in functions") 
		print("----------------------------------------")
		print("MSE")
		print(mean_squared_error(np.ravel(z), model))
		print("R2")
		print(r2_score(np.ravel(z), model))
		print("")

		project1_func.Plotting(x, y, data, model, p=p_degree, noise=noize)


	elif k_fold_cross_validation == True:
		print('Part b: k-fold cross validation')

		n_x = 500      # number of points 
		p_degree = 10  # degree of polynomial 

		# Creating data values x and y 
		x = np.sort(np.random.uniform(0, 1, n_x))
		y = np.sort(np.random.uniform(0, 1, n_x))

		x, y = np.meshgrid(x,y)
		z 	 = project1_func.FrankeFunction(x,y)  # true function 

		X = project1_func.CreateDesignMatrix(x,y, n=p_degree)

		noize = True # planning to have this as an argument input 

		if noize == True:
			print('The data contains a normally distributed noise')
			print("")
			data = project1_func.Create_data(x, y, z, noise=True)

		else: 
			print('The data does not contain noise')
			print("")
			data = project1_func.Create_data(x, y, z, noise=False)
		
		X_train, X_test, data_train, data_test = train_test_split(X, data, shuffle=True, test_size=0.2)
		betas = project1_func.beta(data_train, X_train, method='OLS')

		print("Values using train_test_split and my own functions")
		print("--------------------------------------------------")
		ytilde = X_train @ betas
		print("Training MSE")
		print(project1_func.MeanSquaredError(data_train,ytilde))
		print("Training R2")
		print(project1_func.R2_ScoreFunction(data_train,ytilde))
	
		ypredict = X_test @ betas
		print("Test MSE")
		print(project1_func.MeanSquaredError(data_test,ypredict))
		print("Test R2")
		print(project1_func.R2_ScoreFunction(data_test,ypredict))
		print("")

		print("Values using Scikit's built in functions") # also training??
		print("----------------------------------------")
		print("Training MSE")
		print(mean_squared_error(data_train,ytilde))
		print("Training R2")
		print(r2_score(data_train,ytilde))
		print("Test MSE")
		print(mean_squared_error(data_test, ypredict))
		print("Test R2")
		print(r2_score(data_test, ypredict))
		print("")

		print("Values using own k-fold function")
		print("--------------------------------")

		MSE_train, MSE_test, bias, variance = project1_func.k_fold(data, X, 20, method='OLS')

		print("Train MSE")
		print(MSE_train)
		print("Test MSE")
		print(MSE_test)



	elif bias_varience_tradeoff == True:
		print('Part c: bias-variance tradeoff')
		print('------------------------------')


		n_x = 500      # number of points 
		p_degree = 12  # degree of polynomial 

		# Creating data values x and y 
		x = np.sort(np.random.uniform(0, 1, n_x))
		y = np.sort(np.random.uniform(0, 1, n_x))

		x, y = np.meshgrid(x,y)
		z 	 = project1_func.FrankeFunction(x,y)  # true function 

		noize = True # planning to have this as an argument input 

		if noize == True:
			print('The data contains a normally distributed noise')
			print("")
			data = project1_func.Create_data(x, y, z, noise=True)

		else: 
			print('The data does not contain noise')
			print("")
			data = project1_func.Create_data(x, y, z, noise=False)

		MSE_train = np.zeros(p_degree)
		MSE_test = np.zeros(p_degree)
		bias = np.zeros(p_degree)
		variance = np.zeros(p_degree)
		MSE_train_check = np.zeros(p_degree)
		MSE_test_check = np.zeros(p_degree)


		for degree in range(0, p_degree):
			X = project1_func.CreateDesignMatrix(x,y, n=degree)
			MSE_train[degree], MSE_test[degree], bias[degree], variance[degree] =  project1_func.k_fold(data, X, 10, method='OLS')


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

		plt.figure(1)
		plt.plot(complexity, MSE_test, label='testing')
		plt.plot(complexity, MSE_train, label='training')
		plt.plot(complexity, bias, label="bias")
		plt.plot(complexity, variance, label="variance")
		plt.title('Bias-variance tradeoff')
		plt.legend()
		plt.show()
		

	elif Ridge_method == True:
		print('Part d: Ridge Regression on The Franke function with resampling')
		print('---------------------------------------------------------------')

		n_x = 80      # number of points 
		p_degree = 11  # degree of polynomial 

		# Creating data values x and y 
		x = np.sort(np.random.uniform(0, 1, n_x))
		y = np.sort(np.random.uniform(0, 1, n_x))

		x, y = np.meshgrid(x,y)
		z 	 = project1_func.FrankeFunction(x,y)  # true function 

		data = project1_func.Create_data(x, y, z, noise=True)

		lambdas = np.logspace(-3.8, -1.3, 100)
		print(lambdas)

		grid = np.zeros((len(lambdas), p_degree))

		best_lambda = np.zeros(p_degree)

		for degree in range(4, 7):

			train_MSE = np.zeros(len(lambdas))
			test_MSE = np.zeros(len(lambdas))
			bias = np.zeros(len(lambdas))
			variance = np.zeros(len(lambdas))

			for l in range(0, len(lambdas)):
				X = project1_func.CreateDesignMatrix(x,y, n=degree)
				train_MSE[l], test_MSE[l], bias[l], variance[l] = project1_func.k_fold(data, X, 10, method='Ridge', l=lambdas[l])

			complexity = np.arange(0,p_degree)
			plt.plot(lambdas, test_MSE, label='p = %g' %degree)

		#plt.imshow(grid, origin='lower')
		plt.legend()
		plt.show()

	elif Lasso_method == True:
		print('Part e: Lasso Regression on The Franke function with resampling')
		print('---------------------------------------------------------------')

		n_x = 100       # number of points 
		p_degree = 10  # degree of polynomial 

		# Creating data values x and y 
		x = np.sort(np.random.uniform(0, 1, n_x))
		y = np.sort(np.random.uniform(0, 1, n_x))

		x, y = np.meshgrid(x,y)
		z 	 = project1_func.FrankeFunction(x,y)  # true function 

		X = project1_func.CreateDesignMatrix(x,y, n=p_degree)

		noize = True # planning to have this as an argument input 

		#print(10**np.linspace(10,-2,100)*0.5)

		if noize == True:
			print('The data contains a normally distributed noise')
			print("")
			data = project1_func.Create_data(x, y, z, noise=True)

		else: 
			print('The data does not contain noise')
			print("")
			data = project1_func.Create_data(x, y, z, noise=False)


		#lambdas = np.logspace(-6, -1.2, 100)
		#print(lambdas)

		complexity = np.arange(0,p_degree)


		#alphas = np.logspace(-6, -4.5, 100)
		alphas = np.logspace(-4.9, -3.4, 100)


		for degree in range(5, 8):
			train_MSE = np.zeros(len(alphas))
			test_MSE = np.zeros(len(alphas))
			bias = np.zeros(len(alphas))
			variance = np.zeros(len(alphas))

			for a in range(0, len(alphas)):
				X = project1_func.CreateDesignMatrix(x,y, n=degree)
				train_MSE[a], test_MSE[a], bias[a], variance[a] = project1_func.k_fold(data, X, 10, method='lasso', a=alphas[a])

			alpha_min = np.argmin(test_MSE)
			plt.plot(np.log10(alphas), test_MSE, label='%g' %degree)
			plt.plot(np.log10(alphas)[alpha_min], test_MSE[alpha_min], 'ro')
			print('for degree=%g, best alpha=%g' %(degree, alphas[alpha_min]))

		plt.legend()
		plt.show()



	elif Real_data == True:
		print('Part f: Real data')
		print('-----------------')

		from imageio import imread
		import matplotlib.pyplot as plt
		from mpl_toolkits.mplot3d import Axes3D
		from matplotlib import cm
		# Load the terrain
		terrain_image = imread('Oslo.tif')
		# Show the terrain
		plt.figure(1)
		plt.title('Terrain over Norway 1')
		plt.imshow(terrain_image, cmap='gray')
		plt.colorbar(shrink=0.5, aspect=5)
		plt.xlabel('X')
		plt.ylabel('Y')
		#plt.show()

		p_degree = 10
		
		terrain_arr = np.array(terrain_image)
		n_x = len(terrain_arr)
		print(terrain_arr.shape)


		x = np.arange(0, terrain_arr.shape[1])/(terrain_arr.shape[1]-1)
		y = np.arange(0, terrain_arr.shape[0])/(terrain_arr.shape[0]-1)

		x,y = np.meshgrid(x,y)

		X = project1_func.CreateDesignMatrix(x,y, n=p_degree)

		terrain_arr = np.ravel(terrain_arr)
		print(terrain_arr.shape)
		
		#model = project1_func.OrdinaryLeastSquares(terrain_arr, X)
		#l_r = LinearRegression()
		#l_r.fit(X, terrain_arr)
		#model = l_r.predict(X)

		
		lasso = Lasso(max_iter = 1e3, tol=0.001, normalize = True)
		lasso.set_params(alpha=1e-4)
		lasso.fit(X, terrain_arr)
		model = lasso.predict(X)
		
		MSE = project1_func.MeanSquaredError(terrain_arr, model)
		R_2 = project1_func.R2_ScoreFunction(terrain_arr, model)

		print('MSE:', MSE)
		print('R_2:', R_2)

		plot_model = np.reshape(model, (len(x[:,0]), len(y[0,:])))

		plt.figure(2)
		plt.title('Terrain over Norway 1')
		plt.imshow(plot_model, cmap='gray', alpha=1)
		plt.colorbar(shrink=0.5, aspect=5)
		plt.xlabel('X')
		plt.ylabel('Y')
		plt.show()

		project1_func.Plotting(x, y, terrain_arr, model, p=p_degree, noise=True)
		

	elif Best_fit == True:
		print('Part g: Best fit')
		print('----------------')
