"""
FYS-STK4155 - Main code for project 1
"""
import argparse
import sys
import cv2
import numpy                 as np
import matplotlib.pyplot     as plt

from imageio                 import imread
from PIL                     import Image
from mpl_toolkits.mplot3d    import Axes3D
from matplotlib              import cm
from random                  import randrange, seed
from sklearn.metrics         import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import Lasso, LassoCV, LinearRegression
from sklearn.preprocessing   import normalize

# Import other Python codes for project 1
import project1_func
import project1_plot

np.random.seed(5)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="The Franke function:\
		Regression analysis and resampling methods")

	group = parser.add_mutually_exclusive_group()
	group.add_argument('-a', '--OLS',            action="store_true", help="Ordinary Least Squares")
	group.add_argument('-b', '--k_fold',         action="store_true", help="K-fold cross validation")
	group.add_argument('-c', '--bias_var_trade', action="store_true", help="Bias-variance tradeoff")
	group.add_argument('-d', '--ridge',          action="store_true", help="Ridge regression")
	group.add_argument('-e', '--lasso',          action="store_true", help="Lasso regression")
	group.add_argument('-f', '--real_data',      action="store_true", help="Introducing real data")
	group.add_argument('-g', '--best_fit',       action="store_true", help="Best fit: OLS, Ridge, Lasso")

	if len(sys.argv) <= 1:
		sys.argv.append('--help')

	args                    = parser.parse_args()

	OLS_method              = args.OLS
	k_fold_cross_validation = args.k_fold
	bias_varience_tradeoff  = args.bias_var_trade
	Ridge_method            = args.ridge
	Lasso_method            = args.lasso
	Real_data               = args.real_data
	Best_fit                = args.best_fit

	# Global variables
	n_x      = 200 			# number of points 
	p_degree = 10 			# degree of polynomial 
	k        = 20

	# Creating data values x and y 
	x    = np.sort(np.random.uniform(0, 1, n_x))
	y    = np.sort(np.random.uniform(0, 1, n_x))

	x, y = np.meshgrid(x,y)
	z 	 = project1_func.FrankeFunction(x,y)  # true function 


	if OLS_method == True:
		print('Part a: Ordinary Least Square on The Franke function with resampling')
		print('--------------------------------------------------------------------')

		# confidence interval on the betas

		m = 5
		X = project1_func.CreateDesignMatrix(x,y, n=m)

		noize = True

		if noize == True:
			print('The data contains a normally distributed noise')
			print("")
			data = project1_func.Create_data(x, y, z, noise=True)
			betas, model = project1_func.OrdinaryLeastSquares(data, X)

		else: 
			print('The data does not contain noise')
			print("")
			data = project1_func.Create_data(x, y, z, noise=False)
			betas, model = project1_func.OrdinaryLeastSquares(data, X)

		# Computing the MSE and R2 score between the true data and the model
		MSE = project1_func.MeanSquaredError(np.ravel(z), model)
		R_2 = project1_func.R2_ScoreFunction(np.ravel(z), model)


		# Plotting the 3D model of the Franke function
		project1_plot.Plot_3D_Franke(x, y, data, model, m, 'franke_model', func="OLS", noise=True, savefig=True)
		plt.show()

		# Plotting the true franke function
		project1_plot.Plot_3D_Franke(x, y, data, np.ravel(z), m, 'franke_true_function', func="True", noise=True, savefig=False)
		plt.show()

		project1_func.CI(data, X, betas, model, method='OLS', dataset='Franke')

		if noize == True:
			file = open("Results/MSE_R2_noise_exercise_a.txt", "w") 
			sys.stdout = file 

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

			file.close()
		else:
			file = open("Results/MSE_R2_exercise_a.txt", "w") 
			sys.stdout = file 

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

			file.close()



	elif k_fold_cross_validation == True:
		print('Part b: k-fold cross validation')

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
		
		X_train, X_test, data_train, data_test = train_test_split(X, data, shuffle=True, test_size=1./k)
		betas = project1_func.beta(data_train, X_train, method='OLS')


		# Write values to file
		file = open("Results/MSE_R2_scores.txt", "w") 
		sys.stdout = file 

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

		print("Values using Scikit's built in functions") 
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

		index = np.arange(len(np.ravel(data)))

		MSE_train, MSE_test, bias, variance = project1_func.k_fold(data, X, k, index, method='OLS', shuffle=True)

		print("Training MSE")
		print(MSE_train)
		print("Test MSE")
		print(MSE_test)

		file.close()


	elif bias_varience_tradeoff == True:
		print('Part c: bias-variance tradeoff')
		print('------------------------------')

		noize = True 

		if noize == True:
			print('The data contains a normally distributed noise')
			print("")
			data = project1_func.Create_data(x, y, z, noise=True)

		else: 
			print('The data does not contain noise')
			print("")
			data = project1_func.Create_data(x, y, z, noise=False)


		# Plotting MSE test and train, and printing values to file 
		project1_plot.MSE_BV_Franke(x, y, data, k, p_degree, method='OLS', shuffle=False, savefig=True)
		plt.show()


		# Use best p_deg from plot, make a new 3D plot with that value
		# Make new model
		p_deg_optimal  = 6 

		X_new = project1_func.CreateDesignMatrix(x, y, p_deg_optimal)
		beta_new = project1_func.beta(np.ravel(z), X_new, method='OLS')
		model_new  = X_new @ beta_new

		project1_plot.Plot_3D_Franke(x, y, np.ravel(z), model_new, p_deg_optimal, "OLS_final_model", func="OLS", scatter=True, savefig=True)
		plt.show()

	elif Ridge_method == True:
		print('Part d: Ridge Regression on The Franke function with resampling')
		print('---------------------------------------------------------------')

		data = project1_func.Create_data(x, y, z, noise=True)

		lambdas = np.logspace(-5, -3, 50)

		best_lambda = np.zeros(p_degree)

		p_min =  6
		p_max =  9

		p_deg_optimal  = 7
		lambda_optimal = 0.00148497

		#Deg, Best_lamb, Min_MSE = project1_plot.plot_MSE_lambda(x, y, data, k, p_min, p_max, lambdas, "Franke_Ridge", method='Ridge', shuffle=False, savefig=False)
		#plt.show()
		project1_plot.MSE_BV_Franke(x, y, data, k, p_max, method='Ridge', savefig=False, l=lambda_optimal)
		plt.show()

		X_new = project1_func.CreateDesignMatrix(x,y, p_deg_optimal)
		beta_new = project1_func.beta(np.ravel(z), X_new, method='Ridge', l=lambda_optimal)
		model_new  = X_new @ beta_new

		#project1_plot.plot_3D(x, y, z, 10, 'file_name', func="Ridge", savefig=False)
		#project1_plot.plot_3D(x, y, model_new, p_deg_optimal, 'file_name', func="Ridge", savefig=False)
		project1_plot.Plot_3D_Franke(x, y, z, model_new, p_deg_optimal, 'final_model_Ridge_franke', func="Ridge", scatter=True, savefig=True, l=lambda_optimal)
		plt.show()
		
		

	elif Lasso_method == True:
		print('Part e: Lasso Regression on The Franke function with resampling')
		print('---------------------------------------------------------------')

		#X = project1_func.CreateDesignMatrix(x,y, n=p_degree)

		noize = True # planning to have this as an argument input 

		if noize == True:
			print('The data contains a normally distributed noise')
			print("")
			data = project1_func.Create_data(x, y, z, noise=True)

		else: 
			print('The data does not contain noise')
			print("")
			data = project1_func.Create_data(x, y, z, noise=False)

		lambdas = np.logspace(-7, -4, 50)

		best_lambda = np.zeros(p_degree)

		p_min =  4
		p_max =  5

		p_deg_optimal  = 5
		lambda_optimal = 1e-6

		#Deg, Best_lamb, Min_MSE = project1_plot.plot_MSE_lambda(x, y, data, k, p_min, p_max, lambdas, "Franke_Lasso", method='Lasso', shuffle=False, savefig=True)
		#plt.show()
		
		project1_plot.MSE_BV_Franke(x, y, data, k, p_max, method='Lasso', savefig=True, l=lambda_optimal)
		plt.show()

		X_new = project1_func.CreateDesignMatrix(x,y, p_deg_optimal)
		lasso = Lasso(max_iter = 1e2, tol=0.001, normalize = True) #fit_intercept=False
		#scaler = StandardScaler()
		lasso.set_params(alpha=lambda_optimal)
		lasso.fit(X_new, data)
		beta = lasso.coef_
		model_new = lasso.predict(X_new)

		project1_plot.Plot_3D_Franke(x, y, z, model_new, p_deg_optimal, 'final_model_Lasso_franke', func="Lasso", scatter=True, savefig=True, l=lambda_optimal)
		plt.show()
		

	elif Real_data == True:
		print('Part f: Real data')
		print('-----------------')

		terrain_image = imread('PIA23328.tif')
		reduced_image = terrain_image[700:1100, 200:600]  # Mars
		
		terrain_arr   = np.array(terrain_image)
		n_x           = len(terrain_arr)

		x             = np.arange(0, terrain_arr.shape[1])/(terrain_arr.shape[1]-1)
		y             = np.arange(0, terrain_arr.shape[0])/(terrain_arr.shape[0]-1)
		x,y           = np.meshgrid(x,y)

		final_image = cv2.resize(reduced_image, dsize=(200, 200), interpolation=cv2.INTER_NEAREST)
		x_          = np.arange(0, final_image.shape[1])/(final_image.shape[1]-1)
		y_          = np.arange(0, final_image.shape[0])/(final_image.shape[0]-1)
		x_,y_       = np.meshgrid(x_,y_)

		print(final_image.shape)

		'''
		plt.figure(1)
		project1_plot.plot_terrain(x,y,terrain_image, "Terrain_original", func="Original", string='Utopia Planitia, Mars', savefig=True)
		plt.figure(2)
		project1_plot.plot_terrain(x,y,reduced_image, "Terrain_cropped", func="Original", string='Unnamed crater in Utopia Planitia, Mars', savefig=True)
		plt.figure(3)
		project1_plot.plot_terrain(x_,y_,final_image, "Terrain_final", func="Original", string='Unnamed crater in Utopia Planitia, Mars', savefig=True)
		plt.figure(4)
		project1_plot.plot_3D(x_, y_,final_image, p_degree, "3D_terrain_real_data", "Final_terrain", savefig=True)
		plt.show()
		'''

		

	elif Best_fit == True:
		print('Part g: Best fit of Terrain data')
		print('----------------')

		poly = 8

		terrain_image = imread('PIA23328.tif')
		reduced_image = terrain_image[700:1100, 200:600]  # Mars
		
		terrain_arr   = np.array(terrain_image)
		n_x           = len(terrain_arr)

		x             = np.arange(0, terrain_arr.shape[1])/(terrain_arr.shape[1]-1)
		y             = np.arange(0, terrain_arr.shape[0])/(terrain_arr.shape[0]-1)
		x,y           = np.meshgrid(x,y)

		final_image = cv2.resize(reduced_image, dsize=(200, 200), interpolation=cv2.INTER_NEAREST)
		x_          = np.arange(0, final_image.shape[1])/(final_image.shape[1]-1)
		y_          = np.arange(0, final_image.shape[0])/(final_image.shape[0]-1)
		x_,y_       = np.meshgrid(x_,y_)

		X = project1_func.CreateDesignMatrix(x_,y_, n=poly)
		
		z = np.ravel(final_image)

		betas, model = project1_func.OrdinaryLeastSquares(z, X)


		project1_plot.plot_3D(x_, y_,final_image, p_degree, "3D_terrain_model", "Final_terrain", savefig=False)
		plt.show()
		project1_plot.plot_3D(x_, y_, model, poly, 'file_name', func="OLS", savefig=False)

		plt.show()
		#model_plot = np.reshape(model, (len(y_), len(x_)))
		project1_plot.plot_terrain(x_,y_,model, "Terrain_model", func="Original", string='Unnamed crater in Utopia Planitia, Mars', savefig=False)
		plt.show()





		#model = project1_func.OrdinaryLeastSquares(terrain_arr, X)
		#l_r = LinearRegression()
		#l_r.fit(X, terrain_arr)
		#model = l_r.predict(X)

		'''
		lasso = Lasso(max_iter = 1e3, tol=0.001, normalize = True)
		lasso.set_params(alpha=1e-3)
		lasso.fit(X, terrain_arr)
		model = lasso.predict(X)
		
		MSE = project1_func.MeanSquaredError(terrain_arr, model)
		R_2 = project1_func.R2_ScoreFunction(terrain_arr, model)

		print('MSE:', MSE)
		print('R_2:', R_2)

		project1_plot.plot_terrain(x,y,model, string='Lasso')
		project1_plot.plot_3D(x, y,model, string='Lasso')
		'''