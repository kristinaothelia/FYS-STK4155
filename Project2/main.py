"""
Main program for project 2 in FYS-STK4155
"""
import sys
import os
import random
import xlsxwriter
import pandas            		 as pd
import numpy 	         		 as np
import functions         		 as func
import credit_card       		 as CD
import matplotlib.pyplot 		 as plt
import scikitplot       		 as skplt

from sklearn.model_selection     import train_test_split
from sklearn.preprocessing 		 import OneHotEncoder, Normalizer
from sklearn.compose 			 import ColumnTransformer
from sklearn.preprocessing       import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.metrics 			 import confusion_matrix, accuracy_score, roc_auc_score, auc, roc_curve
from sklearn.linear_model 		 import LogisticRegression
from sklearn.linear_model 		 import SGDRegressor, SGDClassifier  # better than logistic ??
from sklearn.datasets 		     import load_breast_cancer

from neural_network import NN
import plots        as P
import functions    as func
# -----------------------------------------------------------------------------
seed = 0
np.random.seed(seed)
# -----------------------------------------------------------------------------

arg = sys.argv[1]

# Setting the eta and gamma parameters
#eta = 0.01
#gamma = 0.0001  # learning rate?

eta = 0.01
gamma = 0.1  # learning rate?

eta_range = [0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7]
gamma_range = [0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7]



CreditCard = True

if CreditCard == True:
	features, target = CD.CreditCard()
	X, y = CD.DesignMatrix(features, target)
	# Calculating the beta values
	#betas = func.next_beta(X, y, eta, gamma)

	# Splitting X and y in a train and test set
	X_train, X_test, y_train, y_test = func.splitting(X, y, TrainingShare=0.75, seed=seed)

else:
	cancer = load_breast_cancer()
	X = cancer.data
	y = cancer.target

	X_train, X_test, y_train, y_test = func.splitting(X, y, TrainingShare=0.75, seed=seed)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	#logReg = LogisticRegression()
	#logReg.fit(X_train, y_train)

'-------------------------------------------'
print('The shape of X is:', X.shape)
print('The shape of y is:', y.shape)
'-------------------------------------------'
print('')

if arg == "Log":
	# Calculating the beta values based og the training set
	#betas_train = func.next_beta(X_train, y_train, eta, gamma)
	betas_train = func.steepest(X_train, y_train, gamma)

	# Calculating ytilde and the model of logistic regression
	z 		    = X_test @ betas_train   # choosing best beta here?
	model       = func.logistic_function(z)

	# Calculating the accuracy with our own function
	accuracy_test =  func.accuracy(model, y_test)
	exp_term = X_test
	Probabilities = func.probabilities(exp_term)   # ???

	# Creating a logistic regression model with scikit-learn
	# Calculating the corresponding accuracy
	logReg = LogisticRegression(random_state=seed, solver='sag', max_iter=1000, fit_intercept=False) # solver='lbfgs'
	logReg.fit(X_train, np.ravel(y_train))

	ypredict_scikit  		     = logReg.predict(X_test)
	predict_probabilities_scikit = logReg.predict_proba(X_test)  # Probability estimates
	score_scikit				 = logReg.score(X_test, y_test)  # ?? same as accuracy ??

	accuracy_scikit  = accuracy_score(y_pred=ypredict_scikit, y_true=y_test)

	# Comparing our own accuracy with scikit-learn
	print('')
	'-------------------------------------------'
	print('The accuracy with our function is  :', accuracy_test)
	print('The accuracy of scikit-learn is    :', accuracy_scikit)
	'-------------------------------------------'

	fpr, tpr, thresholds = roc_curve(y_test, predict_probabilities_scikit[:,1], pos_label=None)
	AUC_scikit 			 = auc(fpr, tpr)

	# The AUC scikit
	print('')
	'-------------------------------------------'
	print('The AUC is:', AUC_scikit)
	'-------------------------------------------'

	p = predict_probabilities_scikit[:,0]
	#p = func.probabilities(model)
	notP = 1 - np.ravel(p)
	y_p = np.zeros((len(notP), 2))
	y_p[:,0] = np.ravel(p)
	y_p[:,1] = np.ravel(notP)

	x_plot, y_plot = func.bestCurve(y_test)

	skplt.metrics.plot_cumulative_gain(y_test, y_p)
	plt.plot(x_plot, y_plot, label='best curve', linewidth=4)
	plt.legend()
	plt.show()

	# Creating a Confusion matrix using pandas and pandas dataframe
	CM 			 = func.Create_ConfusionMatrix(model, y_test, plot=False)
	CM_DataFrame = func.ConfusionMatrix_DataFrame(CM, labels=['pay', 'default'])

	print('')
	'-------------------------------------------'
	print('The Confusion Matrix')
	print('')
	print(CM_DataFrame)
	'-------------------------------------------'

elif arg == "NN":

    #scaler = RobustScaler()
    '''
    # Scale data
    scaler.fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc = scaler.transform(X_test)
    '''
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
    batch_size = 80 #60 #500

    eta_vals = np.logspace(-7, -4, 7)
    #eta_vals = np.linspace(0, 3.5, 7)
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
                
                #print(X_test_sc)
                test_predict = dnn.predict(X_test_sc)
                #print(test_predict)
                
                #print(test_predict)
                accuracy_array[i][j] = accuracy_score(y_test, test_predict)
                
                print("Learning rate  = ", eta)
                print("Lambda = ", lmbd)
                print("Accuracy score on test set: ", accuracy_score(y_test, test_predict))
                print()


        np.save('acc_score', accuracy_array)
        np.save('eta_values', eta_vals)
        np.save('lambda_values', lmbd_vals)

        P.map()





















#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#design_matrix = pd.DataFrame(X)
#design_matrix.to_excel(excel_writer = "DesignMatrix.xlsx", header=False, index=False)

#target = pd.DataFrame(X)
#design_matrix.to_excel(excel_writer = "DesignMatrix.xlsx")

#design_matrix_file = pd.read_excel("DesignMatrix.xlsx", header=None, skiprows=None, index_col=None)
#print(design_matrix_file)


'''

# Create an instance of the estimator
logReg = LogisticRegression() #n_jobs=-1, random_state=15

# Using the training data to train the estimator
logReg.fit(X_train, y_train)
betas = logReg.coef_

# Evaluating the model
y_pred_test = logReg.predict(X_test)

#accuracy = accuracy_score(y_pred=y_pred_test, y_true=y_test) # metrics.loc['accuracy', 'LogisticReg']


# Confusion matrix
#CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
#CMatrix(CM)
'''
