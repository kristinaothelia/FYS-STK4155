import sys
import numpy              as np
import matplotlib.pyplot  as plt
import project1_func

from matplotlib.ticker    import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib           import cm


def Plot_MSE_R2_BV(x, y, data, k, p_degree, lamb, method='OLS', dataset='Franke', savefig=False, shuffle=False):
	"""
	Function that plots MSE test and MSE train vs Complexity (polynomial degree),
	R2 test and R2 train vs Complexity and Bias-Variance tradeoff.

	p_degree 		| Polynomial degree
	method 			| Method to be tested. OLS is default.
	dataset 		| Insert Franke or Terrain
	biasvariance 	| If True:  Plots a Bias-Variance tradeoff
					  If False: Plots MSE test and MSE train vs Complexity
	"""
	MSE_train       = np.zeros(p_degree)
	MSE_test        = np.zeros(p_degree)
	R2_train        = np.zeros(p_degree)
	R2_test         = np.zeros(p_degree)
	bias            = np.zeros(p_degree)
	variance        = np.zeros(p_degree)

	complexity      = np.arange(0,p_degree)		# Polynomial degrees

	for degree in range(0, p_degree):

		index = np.arange(len(np.ravel(data)))

		if shuffle:
			np.random.shuffle(index)

		if method == 'OLS':
			X = project1_func.CreateDesignMatrix(x,y, n=degree)
			MSE_train[degree], MSE_test[degree], bias[degree], variance[degree], R2_train[degree], R2_test[degree] =  project1_func.k_fold(data, X, k, index, method='OLS')
		if method == 'Ridge':
			X = project1_func.CreateDesignMatrix(x,y, n=degree)
			MSE_train[degree], MSE_test[degree], bias[degree], variance[degree], R2_train[degree], R2_test[degree] =  project1_func.k_fold(data, X, k, index, method='Ridge', l=lamb)
		if method == 'Lasso':
			X = project1_func.CreateDesignMatrix(x,y, n=degree)
			MSE_train[degree], MSE_test[degree], bias[degree], variance[degree], R2_train[degree], R2_test[degree] =  project1_func.k_fold(data, X, k, index, method='Lasso', l=lamb)

	print("Calculations for %s, %s with p: 0-%g" %(method, dataset, p_degree))
	print("--"*35)
	
	print("Train MSE:")
	print(MSE_train)
	print("Test MSE:")
	print(MSE_test)
	print("Train R2:")
	print(R2_train)
	print("Test R2:")
	print(R2_test)
	print("Bias:")
	print(bias)
	print("Variance:")
	print(variance)
	print("bias+variance:")
	print(bias+variance)

	plt.figure(1)
	plt.plot(complexity, R2_test, label='R2 Test')
	plt.plot(complexity, R2_train, label='R2 Train')
	plt.title('R2 for %s (%s)' %(method, dataset), fontsize=16)
	plt.xlabel('Complexity (polynomial degree)', fontsize=16)
	plt.ylabel('Predicted error (R2 score)', fontsize=16)
	plt.legend(fontsize=16)

	if savefig == True:
		plt.savefig('Results/R2_%sdeg_%s.png' %(p_degree, dataset))

	plt.figure(2)
	plt.plot(complexity, bias, label="Bias")
	plt.plot(complexity, variance, label="Variance")
	plt.title('Bias-variance tradeoff for %s (%s)' %(method, dataset), fontsize=16)
	plt.xlabel('Complexity (polynomial degree)', fontsize=16)
	plt.ylabel('Predicted error (MSE)', fontsize=16)
	plt.legend(fontsize=16)

	if savefig == True:
		plt.savefig('Results/BV_%sdeg_%s.png' %(p_degree, dataset))

	plt.figure(3)
	plt.plot(complexity, MSE_test, label='MSE Test')
	plt.plot(complexity, MSE_train, label='MSE Train')
	plt.title('MSE for %s (%s)' %(method, dataset), fontsize=16)
	plt.xlabel('Complexity (polynomial degree)', fontsize=16)
	plt.ylabel('Predicted error (MSE)', fontsize=16)
	plt.legend(fontsize=16)

	if savefig == True:
		plt.savefig('Results/MSE_%sdeg_%s.png' %(p_degree, dataset))


def plot_3D(x, y, z, p, file_name, func="OLS", savefig=False):
	"""
	Create a 3D plot
	input file_name		| Insert a text string, without '.png'
	input func  		| Insert "Ridge" or "Lasso"
	"""
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	if len(z.shape) < 2:
		z = np.reshape(z, (len(x[:,0]), len(y[0,:])))

	surf = ax.plot_surface(x, y, z, alpha=0.5, cmap=cm.coolwarm,linewidth=0, antialiased=False)

	fig.colorbar(surf, shrink=0.5, aspect=7.5)
	ax.view_init(azim=65, elev=57)
	ax.set_xlabel("x",fontsize=16)
	ax.set_ylabel("y",fontsize=16)
	ax.set_zlabel("z",fontsize=16)

	# Make title
	if func == "OLS":
		plt.title('Model with OLS regression \n Polynomial of degree $p = %g $' %p,fontsize=16)
	elif func == "Ridge":
		plt.title('Model with Ridge regression \n Polynomial of degree $p = %g $' %p,fontsize=16)
	elif func == "Lasso":
		plt.title('Model with Lasso regression \n Polynomial of degree $p = %g $' %p,fontsize=16)
	elif func == "Final_terrain":
		plt.title("Unnamed crater in Utopia Planitia, Mars",fontsize=16)

	if savefig == True:
		plt.savefig('Results/' + file_name + '.png')

def Plot_3D_Franke(x, y, z, model, p, file_name, func="OLS", scatter=False, savefig=False, l=0):
	"""
	Function that plots the Franke model
	input p			| Polynomial degree
	input func  	| OLS, Ridge or Lasso
	"""
	# Reshape model if the shape is < 2
	if len(model.shape) < 2:
		model = np.reshape(model, (len(x[:,0]), len(y[0,:])))

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	if scatter == False:
		# Plotting the model surface
		surf_model = ax.plot_surface(x, y, model, alpha=0.5, cmap=cm.coolwarm,linewidth=0, antialiased=False)

	else:
		# Plotting the model surface with the data points as dots
		surf_model = ax.plot_surface(x, y, model, alpha=0.5, cmap=cm.coolwarm,linewidth=0, antialiased=False)
		scatter = ax.scatter(x, y, z, alpha=0.5, s=0.5, color='black')
		plt.gcf().text(0.0, 0.02, 'The black dots represent the true Franke function', fontsize=14) # Adding text

	ax.view_init(azim=52, elev=18)
	ax.set_xlabel("x",fontsize=16)
	ax.set_ylabel("y",fontsize=16)
	ax.set_zlabel("z",fontsize=16)
	# Add a color bar which maps values to colors.
	fig.colorbar(surf_model, shrink=0.5, aspect=7.5)

	# Make title
	if func == "OLS":
		plt.title('Franke function with OLS regression \n Polynomial of degree $p = %g $ and noise' %p,fontsize=16)
	elif func == 'True':
		plt.title('The Franke function',fontsize=16)
	elif func == "Ridge":
		plt.title('Franke function with Ridge regression \n Polynomial of degree $p = %g $ and $\\lambda=%s$' %(p,l),fontsize=16)
	elif func == "Lasso":
		plt.title('Franke function with Lasso regression \n Polynomial of degree $p = %g $ and $\\lambda=%s$' %(p,l),fontsize=16)

	if savefig == True:
		plt.savefig('Results/' + file_name + '.png')



def plot_MSE_lambda(x, y, data, k, degree_start, degree_stop, lambdas, file_name, method=None, shuffle=False, savefig=False):
	"""
	Function to plot MSE vs Lambda, for varoius polynomial degrees
	"""
	if method == None:
		print('You need to pass a method, "Ridge" og "Lasso"')
		sys.exit()

	Deg       = []
	Best_lamb = []
	Min_MSE   = []

	for degree in range(degree_start, degree_stop+1):
		print(degree)
		index = np.arange(len(np.ravel(data)))

		if shuffle:
			np.random.shuffle(index)

		train_MSE = np.zeros(len(lambdas))
		test_MSE  = np.zeros(len(lambdas))
		train_R2  = np.zeros(len(lambdas))
		test_R2   = np.zeros(len(lambdas))
		bias      = np.zeros(len(lambdas))
		variance  = np.zeros(len(lambdas))

		for l in range(0, len(lambdas)):

			X = project1_func.CreateDesignMatrix(x,y, n=degree)
			if method == 'Ridge':
				method_name = "Ridge"
				train_MSE[l], test_MSE[l], bias[l], variance[l], train_R2[l], test_R2[l] = project1_func.k_fold(data, X, k, index, method='Ridge', l=lambdas[l])
			elif method == 'Lasso':
				method_name = "Lasso"
				train_MSE[l], test_MSE[l], bias[l], variance[l], train_R2[l], test_R2[l] = project1_func.k_fold(data, X, k, index, method='Lasso', l=lambdas[l])

		Min_MSE.append(min(test_MSE))
		lambda_min = np.argmin(test_MSE)
		plt.plot(lambdas, test_MSE, label='p = %g' %degree)
		plt.plot(lambdas[lambda_min], test_MSE[lambda_min], 'ro')
		Deg.append(degree)
		Best_lamb.append(lambdas[lambda_min])

	print("The best lambda value for each degree:")
	for i in range(len(Deg)):
		print('For degree=%g - Best alpha=%g - Min MSE=%g' %(Deg[i], Best_lamb[i], Min_MSE[i]))

	plt.title('MSE Test for %s' % method_name, fontsize=16)
	plt.xlabel('$\\lambda$', fontsize=16)
	plt.ylabel('Predicted error (MSE)', fontsize=16)
	plt.legend(fontsize=16)

	if savefig == True:
		plt.savefig('Results/MSEvsLambda_' + file_name + '.png')


	return Deg, Best_lamb, Min_MSE

def plot_both(x, y, model, file_name, string='', savefig=False):
	"""
	Function to plot both a 3D plot and terrain 2D plot
	"""
	fig = plt.figure() #(figsize=plt.figaspect(0.5))
	plt.suptitle('Model %s' %(string), fontsize=16)

	# PLot 2D
	ax = fig.add_subplot(1,2,1, projection='3d')
	plt.title('Terrain', fontsize=14)
	plt.imshow(plot_model, cmap='gray', alpha=1)
	plt.colorbar(shrink=0.5, aspect=7.5)
	ax.set_xlabel('x', fontsize=16)
	ax.set_ylabel('y', fontsize=16)


	if model.size < 1:
		model = np.reshape(model, (len(x), len(y)))

	# PLot 3D
	ax = fig.add_subplot(1,2,2, projection='3d')
	surf = ax.plot_surface(x, y, model, alpha=0.5, cmap=cm.coolwarm,linewidth=0, antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=7.5)
	ax.view_init(azim=65, elev=57)
	plt.title('3D terrain', fontsize=14)
	ax.set_xlabel('x', fontsize=16)
	ax.set_ylabel('y', fontsize=16)

	if savefig == True:
		plt.savefig('Results/' + file_name + '.png')


def plot_terrain(x, y, terrain, deg, lamb, file_name, func="OLS", string='', savefig=False):
	"""
	Function to plot a 2D image of the terrain, with a colorbar
	"""
	# Reshape terrain input if the shape is < 2
	if len(terrain.shape) < 2:
		terrain = np.reshape(terrain, (len(x[:,0]), len(y[0,:])))

	# Make title
	if func == "OLS":
		plt.title("%s \n OLS, k-fold CV, p=%g" %(string, deg), fontsize=16)
	elif func == "Ridge":
		plt.title("%s \n Ridge, k-fold CV, p=%g, $\\lambda$=%g" %(string, deg, lamb), fontsize=16)
	elif func == "Lasso":
		plt.title("%s \n Lasso, k-fold CV, p=%g, $\\lambda$=%g" %(string, deg, lamb), fontsize=16)
	elif func == "Original":
		plt.title("%s" %string, fontsize=16)

	plt.imshow(terrain, cmap='afmhot', alpha=1)
	plt.colorbar(shrink=0.5, aspect=7.5)
	plt.xlabel('x', fontsize=16)
	plt.ylabel('y', fontsize=16)

	if savefig == True:
		plt.savefig('Results/Terrain_' + file_name + '.png')
