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

		#NN_Franke_sklearn.Heatmap_MSE_R2()
		NN_Franke_sklearn.Best_model()

	else:
		print('Pass "Log", "NN" or "linreg" after main.py')
