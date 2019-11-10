"""
Main program for project 2 in FYS-STK4155
"""
import sys
import os
import random
import xlsxwriter
import pandas            		 as pd
import numpy 	         		 as np
import matplotlib.pyplot 		 as plt
import scikitplot       		 as skplt

from sklearn.model_selection     import train_test_split
from sklearn.preprocessing 		 import OneHotEncoder, Normalizer
from sklearn.compose 			 import ColumnTransformer
from sklearn.preprocessing       import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.metrics 			 import confusion_matrix, accuracy_score, roc_auc_score, auc, roc_curve, recall_score, precision_score, f1_score
from sklearn.linear_model 		 import LogisticRegression
from sklearn.linear_model 		 import SGDRegressor, SGDClassifier  # better than logistic ??
from sklearn.datasets 		     import load_breast_cancer

import credit_card     as CD
import plots           as P
import functions       as func
from   neural_network  import NN

# -----------------------------------------------------------------------------
seed = 0
np.random.seed(seed)
# -----------------------------------------------------------------------------

features, target = CD.CreditCard()
X, y 			 = CD.DesignMatrix(features, target)

print('')
print('The shape of X is:', X.shape)
print('The shape of y is:', y.shape)
print('')

# Checking how many 1s and 0s we have
print('Actual number of defaulters    :',     np.sum(y == 1))
print('Actual number of not defaulters:', np.sum(y == 0))
print('')

# Splitting X and y in a train and test set
X_train, X_test, y_train, y_test = func.splitting(X, y, TrainingShare=0.75, seed=seed)

eta = 1e-4
gamma = 0.001

thresholds = np.linspace(0.1,0.9,100)

arg = sys.argv[1]

if arg == "Log":

	# Calculating the beta values based og the training set
	betas_train = func.steepest(X_train, y_train, gamma=gamma, iterations=1000)
	#betas_train = func.SGD_beta(X_train, y_train, eta=1e-4, gamma=0.01)

	#threshold_plot = func.threshold_plot(X_train, X_test, y_train, y_test, gamma, thresholds)
	
	# Calculating ytilde and the model of logistic regression
	z 		    = X_test @ betas_train   # choosing best beta here?
	model       = func.logistic_function(z)
	model 		= func.IndicatorFunc(model, threshold=0.44)

	acc_scikit, TPR_scikit, precision_scikit, f1_score_scikit, AUC_scikit, predict_proba_scikit \
	= func.scikit(X_train, X_test, y_train, y_test, model)

	# Calculating the different metrics
	accuracy_test =  func.accuracy(model, y_test)
	TPR 	      = func.recall(y_test, model)
	precision     = func.precision(y_test, model)
	F1_score      = func.F1_score(y_test, model)

	print('\n-------------------------------------------')
	print('The accuracy is  :', accuracy_test)
	print('The F1 score is  :', F1_score)
	print('The precision is :', precision)
	print('The recall is    :', TPR)
	print('The AUC is       :', AUC_scikit)
	print('-------------------------------------------')

	#p = predict_proba_scikit[:,0]
	p = func.probabilities(model)
	notP = 1 - np.ravel(p)
	y_p = np.zeros((len(notP), 2))
	y_p[:,0] = np.ravel(p)
	y_p[:,1] = np.ravel(notP)

	x_plot, y_plot = func.bestCurve(y_test)

	skplt.metrics.plot_cumulative_gain(y_test, y_p, text_fontsize='medium')
	plt.plot(x_plot, y_plot, linewidth=4)
	plt.legend(["Pay", "Default", "Baseline", "Best curve"])
	plt.ylim(0, 1.05)
	plt.show()

	# Area Ratio?
	x_data, y_data = skplt.helpers.cumulative_gain_curve(y_test, predict_proba_scikit[:,0])
	plt.plot(x_data, y_data)
	plt.show()

	skplt.metrics.plot_roc(y_test, predict_proba_scikit)
	plt.show()

	# Creating a Confusion matrix using pandas and pandas dataframe
	CM 			 = func.Create_ConfusionMatrix(model, y_test, plot=True)
	CM_DataFrame = func.ConfusionMatrix_DataFrame(CM, labels=['pay', 'default'])

	print('')
	'-------------------------------------------'
	print('The Confusion Matrix')
	print('')
	print(CM_DataFrame)
	'-------------------------------------------'
	
elif arg == "NN":
    X_train_sc = X_train
    X_test_sc  = X_test

    def to_categorical_numpy(integer_vector):

	    n_inputs 	  = len(integer_vector)
	    n_categories  = np.max(integer_vector) + 1
	    onehot_vector = np.zeros((n_inputs, n_categories))
	    onehot_vector[range(n_inputs), integer_vector] = 1

	    return onehot_vector

    Y_train_onehot, Y_test_onehot = to_categorical_numpy(y_train), to_categorical_numpy(y_test)

    # 78 accuracy
    epochs     = 100 #60 #30
    batch_size = 80  #60 #500

    eta_vals = np.logspace(-7, -4, 7)
    lmbd_vals = np.logspace(-7, -1, 7)

    # store the models for later use
    DNN_numpy 		 = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
    accuracy_array 	 = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
    n_hidden_neurons = 50 #not sure about number???
    n_categories 	 = 2

    print(X_train.shape)
    print(Y_train_onehot.shape)


    make_files = True
    if make_files:
        # grid search
        for i, eta in enumerate(eta_vals):
            for j, lmbd in enumerate(lmbd_vals):
                dnn = NN(X_train_sc, Y_train_onehot, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size, n_hidden_neurons=n_hidden_neurons, n_categories=n_categories)
                dnn.train()
                
                DNN_numpy[i][j] = dnn
                test_predict    = dnn.predict(X_test_sc)
              
                accuracy_array[i][j] = accuracy_score(y_test, test_predict)
                
                print("Learning rate  = ", eta)
                print("Lambda = ", lmbd)
                print("Accuracy score on test set: ", accuracy_score(y_test, test_predict))
                print()


        np.save('acc_score', accuracy_array)
        np.save('eta_values', eta_vals)
        np.save('lambda_values', lmbd_vals)

    P.map()

'''
p = func.probabilities(model)
notP = 1 - np.ravel(p)
y_p = np.zeros((len(notP), 2))
y_p[:,0] = np.ravel(p)
y_p[:,1] = np.ravel(notP)

x_plot, y_plot = func.bestCurve(y_test)

skplt.metrics.plot_cumulative_gain(y_test, y_p)
plt.plot(x_plot, y_plot, label='best curve', linewidth=4)
plt.legend()
plt.show()

skplt.metrics.plot_roc(y_test, predict_proba_scikit)
plt.show()
'''