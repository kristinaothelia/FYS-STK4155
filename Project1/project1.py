"""
FYS-STK4155 - Main code for project 1
"""
import argparse
import sys
import cv2
import scipy
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

	parser = argparse.ArgumentParser(description="Regression analysis and resampling methods")

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

	# Creating data values x and y (Franke)
	#--------------------------------------------------------------------------

	x    = np.sort(np.random.uniform(0, 1, n_x))
	y    = np.sort(np.random.uniform(0, 1, n_x))

	x, y = np.meshgrid(x,y)
	z 	 = project1_func.FrankeFunction(x,y)  # true function

	# Variables and data values for Terrain data
	#--------------------------------------------------------------------------

	# Reading and creating data array from the .tif file
	terrain_image       = imread('PIA23328.tif')

	# Cropping the image (zooming in on the crater)
	reduced_image       = terrain_image[700:1100, 200:600]

	terrain_arr         = np.array(terrain_image)
	x_terrain           = np.arange(0, terrain_arr.shape[1])/(terrain_arr.shape[1]-1)
	y_terrain           = np.arange(0, terrain_arr.shape[0])/(terrain_arr.shape[0]-1)
	x_terrain,y_terrain = np.meshgrid(x_terrain, y_terrain)

	# Reduce internal points in the cropped image
	final_image         = cv2.resize(reduced_image, dsize=(200, 200), interpolation=cv2.INTER_NEAREST)
	x_                  = np.arange(0, final_image.shape[1])/(final_image.shape[1]-1)
	y_                  = np.arange(0, final_image.shape[0])/(final_image.shape[0]-1)
	x_,y_               = np.meshgrid(x_,y_)
	z_terrain           = np.ravel(final_image)

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
		project1_plot.Plot_3D_Franke(x, y, data, model, m, 'franke_model', func="OLS", savefig=False)
		plt.show()

		# Plotting the true franke function
		project1_plot.Plot_3D_Franke(x, y, data, np.ravel(z), m, 'franke_true_function', func="True", savefig=False)
		plt.show()

		# Calculating and plotting the confidence interval
		project1_func.CI(data, X, betas, model, method='OLS', dataset='Franke')
		plt.show()

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

		MSE_train, MSE_test, bias, variance, R2_train, R2_test = project1_func.k_fold(data, X, k, index, method='OLS', shuffle=True)

		print("MSE Train")
		print(MSE_train)
		print("MSE Test")
		print(MSE_test)
		print("R2 Train")
		print(R2_train)
		print("R2 Test")
		print(R2_test)

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
		lamb=0
		project1_plot.MSE_BV_Franke(x, y, data, k, p_degree, lamb, method='OLS', shuffle=False, savefig=True)
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
		#project1_plot.MSE_BV_Franke(x, y, data, k, p_max, lambda_optimal, method='Ridge', savefig=False)
		#plt.show()

		X_new = project1_func.CreateDesignMatrix(x,y, p_deg_optimal)
		beta_new = project1_func.beta(np.ravel(z), X_new, method='Ridge', l=lambda_optimal)
		model_new  = X_new @ beta_new

		#project1_plot.plot_3D(x, y, z, 10, 'file_name', func="Ridge", savefig=False)
		#project1_plot.plot_3D(x, y, model_new, p_deg_optimal, 'file_name', func="Ridge", savefig=False)
		#project1_plot.Plot_3D_Franke(x, y, z, model_new, p_deg_optimal, 'final_model_Ridge_franke', func="Ridge", scatter=True, savefig=True, l=lambda_optimal)
		#plt.show()

		# Printing and calculating the confidence intervals for Ridge with the best lambda
		# Need to comment out MSE_BV_Franke before running (ValueError: I/O operation on closed file.)
		project1_func.CI(z, X_new, beta_new, model_new, method='Ridge', dataset='Franke')
		plt.show()


	elif Lasso_method == True:
		print('Part e: Lasso Regression on The Franke function with resampling')
		print('---------------------------------------------------------------')

		#X = project1_func.CreateDesignMatrix(x,y, n=p_degree)

		noize = True

		if noize == True:
			print('The data contains a normally distributed noise')
			print("")
			data = project1_func.Create_data(x, y, z, noise=True)

		else:
			print('The data does not contain noise')
			print("")
			data = project1_func.Create_data(x, y, z, noise=False)

		lambdas = np.logspace(-6.8, -5, 100)

		best_lambda = np.zeros(p_degree)

		p_min =  4
		p_max =  6

		p_deg_optimal  = 4
		lambda_optimal = 1.67683e-6

		# Need to run this from 0 to p_degree to get all the values

		#'The best lambda value for each degree:'
		#'For degree=4 - Best alpha=1.67683e-06 - Min MSE=1.01693'
		#'For degree=5 - Best alpha=2.17137e-07 - Min MSE=1.01728'
		#'For degree=6 - Best alpha=1.25893e-07 - Min MSE=1.01672'

		Deg, Best_lamb, Min_MSE = project1_plot.plot_MSE_lambda(x, y, data, k, p_min, p_max, lambdas, "Franke_Lasso", method='Lasso', shuffle=False, savefig=True)
		plt.show()

		#project1_plot.MSE_BV_Franke(x, y, data, k, p_max, method='Lasso', savefig=True, l=lambda_optimal)
		#plt.show()

		X_new = project1_func.CreateDesignMatrix(x,y, p_deg_optimal)
		lasso = Lasso(max_iter = 1e2, tol=0.001, normalize = True) #fit_intercept=False
		scaler = StandardScaler()
		lasso.set_params(alpha=lambda_optimal)
		lasso.fit(X_new, data)
		beta_new = lasso.coef_
		model_new = lasso.predict(X_new)

		#project1_plot.Plot_3D_Franke(x, y, z, model_new, p_deg_optimal, 'final_model_Lasso_franke_', func="Lasso", scatter=True, savefig=True, l=lambda_optimal)
		#plt.show()


		# Printing and calculating the confidence intervals for Lasso with the best lambda
		# Need to comment out MSE_BV_Franke before running (ValueError: I/O operation on closed file.)
		#project1_func.CI(z, X_new, beta_new, model_new, method='Lasso', dataset='Franke')
		#plt.show()

		# Plotting MSE test and train for all the model in the same plot
		#project1_plot.MSE_BV_Franke(x, y, data, k, p_max, method='Lasso', savefig=False, l=1.67683e-6)
		#project1_plot.MSE_BV_Franke(x, y, data, k, p_max, method='Ridge', savefig=False, l=0.00148497)
		#project1_plot.MSE_BV_Franke(x, y, data, k, p_max, method='OLS', savefig=False)
		#plt.show()


	elif Real_data == True:
		print('Part f: Real data. Plot terrain images')
		print('-----------------')

		plt.figure(1)
		project1_plot.plot_terrain(x_terrain,y_terrain,terrain_image, "Terrain_original", func="Original", string='Utopia Planitia, Mars', savefig=True)
		plt.figure(2)
		project1_plot.plot_terrain(x_terrain,y_terrain,reduced_image, "Terrain_cropped", func="Original", string='Unnamed crater in Utopia Planitia, Mars', savefig=True)
		plt.figure(3)
		project1_plot.plot_terrain(x_,y_,final_image, "Terrain_final", func="Original", string='Unnamed crater in Utopia Planitia, Mars', savefig=True)
		plt.figure(4)
		project1_plot.plot_3D(x_, y_,final_image, p_degree, "3D_terrain_real_data", "Final_terrain", savefig=True)
		plt.show()


	elif Best_fit == True:
		"""
		OLS_p5 				| Runs a similar scenario as ex a), for polynomial degree 5.
		OLS_kfold_p5
		OLS_best			| Runs a OLS regression with k-fold CV, with the best model/p_degree. Calculates the bias-variance tradeoff
		Ridge_MSEvsLambda 	| Runs a simulation and plots MSE vs Lambda values for i-polynomial degrees. A little slow process
							  Used to find the polynomial degree and lambda value for Ridge which gives the model with the lowest/best MSE.
		Terrain_Ridge_calc  | Uses the best fit model, best p_degree and lambda value, to plot a MSE vs comlexity plot and terrain 2D and 3D images
		Lasso_MSEvsLambda 	| Runs a simulation and plots MSE vs Lambda values for i-polynomial degrees. A VERY slow process
							  Used to find the polynomial degree and lambda value for Lasso which gives the model with the lowest/best MSE.
		Lasso_best 			| Uses the best fit model, best p_degree and lambda value, to plot a MSE vs comlexity plot and terrain 2D and 3D images
		Confidence_interval | Confidence interval for the best OLS, Ridge and Lasso model. txt file
							  Note: OLS_best_BV = Terrain_Ridge_calc = Terrain_Lasso_calc = True
		"""
		print('Part g: Terrain data - Best fit calculations')
		print('----------------')

		OLS_p5              = False
		OLS_kfold_p5        = False
		OLS_best_BV         = True

		Ridge_MSEvsLambda   = False		# A little slow to run
		Terrain_Ridge_calc  = True

		Lasso_MSEvsLambda   = False		# VERY slow to run!
		Terrain_Lasso_calc  = True

		Confidence_interval = True

		#----------------------------------------------------------------------
		if OLS_p5 == True:
			print("Terrain data - Ex.a: OLS. Polynomial degree 5")
			print("--"*35)

			p_degree_a   = 5
			X_a          = project1_func.CreateDesignMatrix(x_,y_, n=p_degree_a)
			betas, model = project1_func.OrdinaryLeastSquares(z_terrain, X_a)
			project1_plot.plot_3D(x_, y_,final_image, p_degree_a, "3D_terrain_model", "Final_terrain", savefig=True)
			plt.show()
			project1_plot.plot_3D(x_, y_, model, p_degree_a, 'file_name', func="OLS", savefig=True)

			plt.show()
			project1_plot.plot_terrain(x_, y_, model, p_degree_a, 0, "OLS_p5", func="OLS", string='Unnamed crater in Utopia Planitia, Mars', savefig=True)
			plt.show()

			# Computing the MSE and R2 score between the true data and the model
			MSE = project1_func.MeanSquaredError(z_terrain, model)
			R_2 = project1_func.R2_ScoreFunction(z_terrain, model)

			print("MSE and R2 vs scikit learn, bias and variance:")
			print('Mean Square Error, Terrain:  ' , MSE)
			print('MSE sklearn, Terrain:        ' , mean_squared_error(z_terrain, model))
			print('R^2 Score function, Terrain: ' , R_2)
			print('R^2 sklearn, Terrain:        ' , r2_score(z_terrain, model))
			print('Bias:                        ' , project1_func.Bias(z_terrain, model))
			print('Variance:                    ' , np.var(model))

			# Confidence interval. Plotted and printed to terminal
			project1_func.CI(z_terrain, X_a, betas, model, p_degree_a, method='OLS', dataset='Terrain', plot=True)
			plt.show()
			project1_func.CI(z_terrain, X_a, betas, model, p_degree_a, method='OLS', dataset='Terrain', plot=False)

		if OLS_kfold_p5 == True:
			print("Terrain data - Ex.b: OLS with k-fold CV")
			print("--"*35)

			p_degree_a   = 5
			index        = np.arange(len(np.ravel(z_terrain)))
			X_a          = project1_func.CreateDesignMatrix(x_,y_, n=p_degree_a)
			betas, model = project1_func.OrdinaryLeastSquares(z_terrain, X_a)

			MSE_train, MSE_test, bias, variance, R2_train, R2_test = project1_func.k_fold(z_terrain, X_a, k, index, method='OLS', shuffle=True)
			# Print MSE, R2, bias and variance to terminal
			print('MSE Test:                    ' , MSE_test)
			print('R2 Test:                     ' , R2_test)
			print('Bias:                        ' , project1_func.Bias(z_terrain, model))
			print('Variance:                    ' , np.var(model))


		if OLS_best_BV == True:
			print("Terrain data - Ex.c: Bias-Variance Tradeoff and best model for OLS with k-fold CV")
			print("--"*35)

			p_max = 12
			lamb  = 0

			# Plotting MSE test and train, and printing values to file
			#project1_plot.MSE_BV_Terrain(x_, y_, z_terrain, k, p_max, lamb, method='OLS', shuffle=False, savefig=True)
			#plt.show()

			# Use best p_deg from plot, make a new 3D plot with that value. Make new model
			p_deg_optimal  = 8
			X_new          = project1_func.CreateDesignMatrix(x_, y_, p_deg_optimal)
			beta_new       = project1_func.beta(z_terrain, X_new, method='OLS')
			model_new      = X_new @ beta_new

			project1_plot.plot_3D(x_, y_, model_new, p_deg_optimal, "OLS_final_model_Terrain", func="OLS", savefig=False)
			plt.show()
			project1_plot.plot_terrain(x_,y_,model_new, p_deg_optimal, lamb, "Terrain_final_best_p_OLS", func="OLS", string='Unnamed crater in Utopia Planitia, Mars', savefig=False)
			plt.show()

			# Calculate the confidence interval for the best OLS model. Printed to terminal
			# Need to comment out MSE_BV_Franke before running (ValueError: I/O operation on closed file.)
			project1_func.CI(z_terrain, X_new, beta_new, model_new, p_deg_optimal, method='OLS', dataset='Terrain')
			plt.show()

			# Confidence interval. Plotted and printed to terminal
			project1_func.CI(z_terrain, X_new, beta_new, model_new, p_deg_optimal, method='OLS', dataset='Terrain', plot=True)
			plt.show()
			CI_OLS = project1_func.CI(z_terrain, X_new, beta_new, model_new, p_deg_optimal, method='OLS', dataset='Terrain', plot=False)
			#print(CI_OLS)

		if Terrain_Ridge_calc == True:
			print("Terrain data - Ex.d: Ridge regression on terrain data")
			print("--"*35)

			#best_lambda = np.zeros(p_degree)
			p_min =  9
			p_max =  14

			if Ridge_MSEvsLambda == True:

				lambdas     = np.logspace(-6.9, -4.5, 20)
				#lam_for_BV  = [1e-4, 1e-3, 0.01, 0.1, 0, 1]

				Deg, Best_lamb, Min_MSE = project1_plot.plot_MSE_lambda(x_, y_, z_terrain, k, p_min, p_max, lambdas, "Terrain_Ridge", method='Ridge', shuffle=False, savefig=True)
				plt.show()

			# Found from Ridge_MSEvsLambda
			lambda_optimal = 5.38988e-07
			p_deg_optimal  = 9

			#project1_plot.MSE_BV_Terrain(x_, y_, z_terrain, k, p_max, lambda_optimal, method='Ridge', savefig=True)
			#plt.show()

			X_new = project1_func.CreateDesignMatrix(x_, y_, p_deg_optimal)
			beta_new = project1_func.beta(z_terrain, X_new, method='Ridge', l=lambda_optimal)
			model_new  = X_new @ beta_new

			project1_plot.plot_3D(x_, y_, model_new, p_deg_optimal, "Ridge_final_model_Terrain_", func="Ridge", savefig=True)
			plt.show()
			project1_plot.plot_terrain(x_, y_, model_new, p_deg_optimal, lambda_optimal, 'Terrain_final_best_p_Ridge', func="Ridge", string='Unnamed crater in Utopia Planitia, Mars', savefig=True)
			plt.show()

			# Printing and calculating the confidence intervals for Ridge with the best lambda
			# Need to comment out MSE_BV_Franke before running (ValueError: I/O operation on closed file.)
			project1_func.CI(z_terrain, X_new, beta_new, model_new, p_deg_optimal, method='Ridge', dataset='Terrain', plot=True)
			plt.show()
			CI_Ridge = project1_func.CI(z_terrain, X_new, beta_new, model_new, p_deg_optimal, method='Ridge', dataset='Terrain', plot=False)


		if Terrain_Lasso_calc == True:
			print("Terrain data - Ex.e: Lasso regression on terrain data")
			print("--"*35)

			p_min =  9
			p_max =  12

			if Lasso_MSEvsLambda == True:

				lambdas = np.logspace(-6.9, -5.5, 20)
				#lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10, 1e2, 1e3]

				Deg, Best_lamb, Min_MSE = project1_plot.plot_MSE_lambda(x_, y_, z_terrain, k, p_min, p_max, lambdas, "Terrain_Lasso", method='Lasso', shuffle=False, savefig=False)
				plt.show()

			p_deg_optimal  = 10
			lambda_optimal = 3e-5

			#project1_plot.MSE_BV_Terrain(x_, y_, z_terrain, k, p_max, lambda_optimal, method='Lasso', savefig=True)
			#plt.show()

			X_new     = project1_func.CreateDesignMatrix(x_,y_, p_deg_optimal)
			lasso     = Lasso(max_iter = 5e3, tol=0.00001, normalize=True, fit_intercept=False)
			lasso.set_params(alpha=lambda_optimal)
			lasso.fit(X_new, z_terrain)
			beta      = lasso.coef_
			model_new = lasso.predict(X_new)

			project1_plot.plot_3D(x_, y_, model_new, p_deg_optimal, "Lasso_final_model_Terrain_", func="Lasso", savefig=True)
			plt.show()
			project1_plot.plot_terrain(x_, y_, model_new, p_deg_optimal, lambda_optimal, 'Terrain_final_best_p_Lasso', func="Lasso", string='Unnamed crater in Utopia Planitia, Mars', savefig=True)
			plt.show()

			MSE = project1_func.MeanSquaredError(z_terrain, model_new)
			R_2 = project1_func.R2_ScoreFunction(z_terrain, model_new)

			print('MSE:', MSE)
			print('R_2:', R_2)

			# Printing and calculating the confidence intervals for Ridge with the best lambda
			# Need to comment out MSE_BV_Franke before running (ValueError: I/O operation on closed file.)
			project1_func.CI(z_terrain, X_new, beta_new, model_new, p_deg_optimal, method='Lasso', dataset='Terrain', plot=True)
			plt.show()
			CI_Lasso = project1_func.CI(z_terrain, X_new, beta_new, model_new, p_deg_optimal, method='Lasso', dataset='Terrain', plot=False)



		if Confidence_interval == True:

			file_ = open("Results/Confidence_interval_Terrain_ALL.txt", "w")
			sys.stdout = file_
			print("OLS")
			print("        Beta             -                + ")
			print(CI_OLS)
			print("--"*20)
			print("Ridge")
			print("        Beta             -                + ")
			print(CI_Ridge)
			print("--"*20)
			print("Lasso")
			print("        Beta             -                + ")
			print(CI_Lasso)

			file_.close()
