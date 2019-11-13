"""
Main program for project 2 in FYS-STK4155
"""
import sys, os, random, xlsxwriter
import pandas            		 as pd
import numpy 	         		 as np
import matplotlib.pyplot 		 as plt
import scikitplot       		 as skplt

from sklearn.model_selection     import train_test_split
from sklearn.preprocessing 		 import OneHotEncoder, Normalizer
from sklearn.compose 			 import ColumnTransformer
from sklearn.preprocessing       import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.metrics 			 import confusion_matrix, accuracy_score, roc_auc_score, auc, roc_curve, recall_score, precision_score, f1_score
from sklearn.linear_model 		 import LogisticRegression, SGDRegressor, SGDClassifier

import credit_card       as CD
import plots             as P
import functions         as func
import nn_linreg_sklearn as NN_Franke_sklearn
from   neural_network    import NN
from   nn_linreg_new     import MLP

# -----------------------------------------------------------------------------
seed = 0
np.random.seed(seed)

eta_range   = [0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7]
gamma_range = [0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7]
eta         = 1e-4
gamma       = 0.001
thresholds  = np.linspace(0.1, 0.9, 100)
# -----------------------------------------------------------------------------

features, target = CD.CreditCard(Corr_matrix=True)
X, y 			 = CD.DesignMatrix(features, target)
# Calculating the beta values
#betas = func.next_beta(X, y, eta, gamma)

# Checking how many 1s and 0s we have
print('----------- Data information --------------')
print('Actual number of defaulters:    ', np.sum(y == 1))
print('Actual number of not defaulters:', np.sum(y == 0))
print('-------------------------------------------')

# Splitting X and y in a train and test set
X_train, X_test, y_train, y_test = func.splitting(X, y, TrainingShare=0.75, seed=seed)



if __name__ == '__main__':

	# check for input arguments
	if len(sys.argv) == 1:
		print('No arguments passed. Please specify method; "Log", "NN" or "linreg".')
		sys.exit()

	arg = sys.argv[1]

	if arg == "Log":

		# Calculating the beta values based of the training set:
		betas_train = func.steepest(X_train, y_train, gamma=gamma, iterations=1000)

		# Plot treshold plot:
		#threshold_plot = func.threshold_plot(X_train, X_test, y_train, y_test, gamma, thresholds)

		# Calculating ytilde and the model of logistic regression
		z 		    = X_test @ betas_train   # choosing best beta here?
		model       = func.logistic_function(z)
		model 		= func.IndicatorFunc(model, threshold=0.44)

		# Get AUC score and predict_proba_scikit. Used for plots and terminal print
		acc_scikit, TPR_scikit, precision_scikit, f1_score_scikit, AUC_scikit, predict_proba_scikit \
		= func.scikit(X_train, X_test, y_train, y_test, model)

		# Calculating the different metrics:
		print('\n-------------------------------------------')
		print('The accuracy is  : %.3f' % func.accuracy(model, y_test))
		print('The F1 score is  : %.3f' % func.F1_score(y_test, model))
		print('The precision is : %.3f' % func.precision(y_test, model))
		print('The recall is    : %.3f' % func.recall(y_test, model))
		print('The AUC is       : %.3f' % AUC_scikit)
		print('-------------------------------------------')

		# Make Cumulative gain and ROC plot
		P.Cumulative_gain_plot(y_test, model)
		P.ROC_plot(y_test, predict_proba_scikit)

		# Creating a Confusion matrix using pandas and pandas dataframe
		P.Confusion_matrix(y_test, model)


	elif arg == "NN":

	    X_train_sc = X_train
	    X_test_sc  = X_test

	    Y_train_onehot, Y_test_onehot = func.to_categorical_numpy(y_train), func.to_categorical_numpy(y_test)

	    epochs           = 100
	    batch_size       = 80
	    n_hidden_neurons = 50
	    n_categories 	 = 2
	    eta_vals         = np.logspace(-7, -4, 7)
	    lmbd_vals        = np.logspace(-7, -1, 7)

	    # Make heatmap of the accuracy score with eta_vals and lmbd_vals
		# Commented out to save time
	    #func.heatmap(eta_vals, lmbd_vals, X_test_sc, X_train_sc, Y_train_onehot, y_test, epochs, batch_size, n_hidden_neurons, n_categories)

	    # Use best values from heatmap
	    eta_final  = 1e-4
	    lmbd_final = 1e-2

	    dnn_f = NN(X_train_sc, Y_train_onehot, eta=eta_final, lmbd=lmbd_final, epochs=epochs, batch_size=batch_size, n_hidden_neurons=n_hidden_neurons, n_categories=n_categories)
	    dnn_f.train()

	    y_predict = dnn_f.predict(X_test_sc)
	    model     = y_predict

	    # Make Cumulative gain plot
	    P.Cumulative_gain_plot(y_test, model)

		# Creating a Confusion matrix using pandas and pandas dataframe
	    CM 			  = func.Create_ConfusionMatrix(model, y_test, plot=True)
	    CM_DataFrame  = func.ConfusionMatrix_DataFrame(CM, labels=['pay', 'default'])

	    acc_scikit, TPR_scikit, precision_scikit, f1_score_scikit, AUC_scikit, predict_proba_scikit = func.scikit(X_train, X_test, y_train, y_test, model)

	    # Calculating the different metrics
	    print('\n-------------------------------------------')
	    print('The accuracy is  : %.3f' % func.accuracy(model, y_test))
	    print('The F1 score is  : %.3f' % func.F1_score(y_test, model))
	    print('The precision is : %.3f' % func.precision(y_test, model))
	    print('The recall is    : %.3f' % func.recall(y_test, model))
	    print('The AUC is       : %.3f' % AUC_scikit)
	    print('-------------------------------------------')

    elif arg == "linreg":

		if __name__ == "__main__":

		    def FrankeFunction(x,y):
			    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
			    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
			    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
			    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
			    return term1 + term2 + term3 + term4

		    def create_X(x, y, n ):
			    if len(x.shape) > 1:
				    x = np.ravel(x)
				    y = np.ravel(y)
			    N = len(x)
			    l = int((n+1)*(n+2)/2)		# Number of elements in beta
			    X = np.ones((N,l))
			    for i in range(1,n+1):
				    q = int((i)*(i+1)/2)
				    for k in range(i+1):
					    X[:,q+k] = (x**(i-k))*(y**k)
			    return X

		    def relu(x):
		        """
		        the relu function
		        f(x) = max(0, x)
		        """
		        return abs(x)*(x > 0)

		    def idf(x):
		        """
		        the identity function
		        f(x) = x
		        """
		        return x

		    def d_relu(x):
		        return 1.*(x > 0)


			np.random.seed(5)

			n = 5  	 # Project 1
			N = 200  # Project 1
			k = 20   # Project 1

			x = np.sort(np.random.uniform(0, 1, N))
			y = np.sort(np.random.uniform(0, 1, N))
			X = func.create_X(x, y, n)

			XX, YY = np.meshgrid(x,y)
			ZZ     = func.FrankeFunction(XX, YY)
			z      = func.Create_data(XX, YY, ZZ, noise=True)
			X      = func.create_X(XX, YY, n)

			X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=1.0/k)

			epochs     = 500 # 1000
			batch_size = 100

		    X_train, X_test = X_train.T, X_test.T

		    eta_vals  = np.logspace(-6, -3, 4)
		    lmbd_vals = np.logspace(-6, -3, 4)
		    MSE_array = np.zeros((len(eta_vals), len(lmbd_vals)))  #keeping the MSE value for heatmap

		    n_features = len(X_train)  # nr of predictors
		    n_responses = 1
		    n_nodes = 50
		    hidden_activation = relu
		    response_activation = idf
		    hidden_activation_df = d_relu

		    MSE = np.zeros((len(eta_vals),len(lmbd_vals)))

		    nn_reg = MLP(n_features=n_features, n_responses=n_responses, n_nodes=n_nodes,
		                 hidden_activation=hidden_activation, response_activation=response_activation,
		                 hidden_activation_df=hidden_activation_df)

		    nn_reg.create_biases_and_weights(weight_spread = 0.1, bias_constant = 0.1)

		    def not_now():
		        for i,eta in enumerate(eta_vals):
		            for j,lmbd in enumerate(lmbd_vals):

		                print("step:    ", i, j)
		                print("eta    = ", eta)
		                print("lambda = ", lmbd)

		                nn_reg.train(X_train, y_train, learning_rate=eta, regularization=lmbd,
		                             n_epochs=epochs, batch_size=batch_size)
		                y_pred = nn_reg.predict(X_test).flatten()

		                MSE[i,j] = mean_squared_error(y_train, y_pred)

		                print("MSE score on test set: ", MSE[i,j])
		                print()

		    not_now()

		    np.save('acc_score', accuracy_array)
		    np.save('eta_values', eta_vals)
		    np.save('lambda_values', lmbd_vals)

			P.map()


			'''
			etas = ["{:0.2e}".format(i) for i in eta_vals]

			fig, ax = plt.subplots()
			sns.heatmap(MSE, annot=True, xticklabels=lmbd_vals, yticklabels=etas, ax=ax, linewidths=.3, linecolor="black")
			ax.set_title("MSE scores (sklearn)")
			ax.set_ylabel("$\eta$")
			ax.set_xlabel("$\lambda$")
			plt.show()

			fig, ax = plt.subplots()
			sns.heatmap(R2, annot=True, xticklabels=lmbd_vals, yticklabels=etas, ax=ax, linewidths=.3, linecolor="black")
			ax.set_title("Accuracy/R2 scores (sklearn)")
			ax.set_ylabel("$\eta$")
			ax.set_xlabel("$\lambda$")
			plt.show()
			'''

			# Find best values:

		    i_min, j_min = 0,0

		    nn_reg.train(X_train, y_train, learning_rate=eta_vals[-1], regularization=lmbd_vals[-1],
		                 n_epochs=epochs, batch_size=batch_size)
		    y_pred2 = nn_reg.predict(X_train).flatten()

		    print("MSE sco  re on test set: ", mean_squared_error(y_train, np.ravel(y_pred2)))


			fig  = plt.figure(); fig2 = plt.figure()
			#cmap = cm.PRGn; my_cmap_r = cm.get_cmap('PRGn_r')
			ax   = fig.gca(projection='3d'); ax2  = fig2.gca(projection='3d')

			ax.set_title("Franke's function", fontsize=16)
			ax2.set_title("Model (sklearn)", fontsize=16)

			# Plot the surface.
			surf = ax.plot_surface(XX, YY, ZZ, cmap=cm.inferno, #my_cmap_r
			                       linewidth=0, antialiased=False)

			surf2 = ax2.plot_surface(XX, YY, y_pred2, cmap=cm.inferno,
			                       linewidth=0, antialiased=False)

			# Customize the z axis.
			ax.view_init(azim=61, elev=15);  ax2.view_init(azim=61, elev=15)
			ax.set_xlabel('x',  fontsize=15); ax.set_ylabel('y',  fontsize=15); ax.set_zlabel('z',  fontsize=15)
			ax2.set_xlabel('x', fontsize=15); ax2.set_ylabel('y', fontsize=15); ax2.set_zlabel('z', fontsize=15)

			# Add a color bar which maps values to colors.
			fig.colorbar(surf,   shrink=0.5, aspect=5)
			fig2.colorbar(surf2, shrink=0.5, aspect=5)
			plt.show()


	elif arg == "linreg_sklearn":

		#NN_Franke_sklearn.Heatmap_MSE_R2()
		NN_Franke_sklearn.Best_model()

	else:
		print('Pass "Log", "NN" or "linreg" after main.py')
