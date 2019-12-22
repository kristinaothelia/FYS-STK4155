"""
FYS-STK4155 - Project 3: Logistic regression
"""
import os
import numpy 				 as np
import matplotlib.pyplot	 as plt
import scikitplot        	 as skplt
import seaborn               as sns
import pandas 				 as pd
import functions 			 as func
import goldilock             as GL

from random 				 import random, seed
from sklearn.metrics 		 import classification_report, f1_score,		\
								    precision_score, recall_score,       \
                                    accuracy_score, mean_squared_error,  \
                                    confusion_matrix
from sklearn.linear_model 	 import LogisticRegression
from sklearn.model_selection import GridSearchCV
#------------------------------------------------------------------------------

def Best_params(seed, X_train, y_train):

	param_test = {"solver": ['lbfgs', 'liblinear', 'saga'],
				  "max_iter" :			[50,100,200,500],
				  }

	gsearch = GridSearchCV(LogisticRegression(random_state=seed), param_grid = param_test, cv=5)
	trained_model = gsearch.fit(X_train, y_train)
	print("Best parameters: ", trained_model.best_params_)


def LogReg(X_train, X_test, y_train, y_test, candidates, GoldiLock, seed, Goldilock_zone=False, plot_confuse_matrix=False,
			threshold=0.5):

	print("Exoplanet threshold = %g" % threshold)

	# Find best parameters
	#Best_params(seed, X_train, y_train)

	# Make Logistic regression analysis
	logreg = LogisticRegression(solver='liblinear', max_iter = 100, random_state=seed)
	logreg.fit(X_train, y_train)
	y_pred = logreg.predict(X_test)

	# Calculating different metrics
	accuracy 	= accuracy_score(y_test,y_pred)
	precision   = precision_score(y_test, y_pred, average="macro")
	recall      = recall_score(y_test, y_pred, average="macro")
	F1_score    = f1_score(y_test, y_pred, average="macro")
	# Calculate the absolute errors
	errors 		= abs(y_pred - y_test)

	# Printing the different metrics:
	func.Print_parameters(accuracy, F1_score, precision, recall, errors, name='Logistic regression')

	if plot_confuse_matrix == True:

		func.ConfusionMatrix_Plot(y_test, y_pred, 'Logistic regression (Candidates)', threshold)


	# Prediction with threshold
	pred_cand = np.array(logreg.predict_proba(candidates))

	pred_cand[:,0] = (pred_cand[:,0] < threshold).astype('int')
	pred_cand[:,1] = (pred_cand[:,1] >= threshold).astype('int')

	pred_FP   	   = (pred_cand[:,1] == 0).sum()
	pred_Conf 	   = (pred_cand[:,1] == 1).sum()

	# Information print to terminal
	print('\nThe LOG Classifier predicted')
	print('--------------------------------------')
	print('%-5g exoplanets      of %g candidates'  %(pred_Conf, len(pred_cand)))
	print('%-5g false positives of %g candidates'  %(pred_FP, len(pred_cand)))

	# Plotting a bar plot of candidates predicted as confirmed and false positives
	func.Histogram2(pred_cand[:,1], 'Logistic regression (Candidates)', threshold)

	# Make AUC curve?

	if Goldilock_zone:

		print("Goldilock zone calculations")

		# Prediction with threshold
		predict_goldilocks = np.array(logreg.predict_proba(GoldiLock))

		predict_goldilocks[:,0] = (predict_goldilocks[:,0] < threshold).astype('int')
		predict_goldilocks[:,1] = (predict_goldilocks[:,1] >= threshold).astype('int')

		predicted_false_positive_goldilocs  = (predict_goldilocks[:,1] == 0).sum()
		predicted_exoplanets_goldilocks     = (predict_goldilocks[:,1] == 1).sum()

		# Information print to terminal
		print('\nThe LOG Classifier predicted')
		print('--------------------------------------')
		print('%-3g exoplanets      of %g candidates'  %(predicted_exoplanets_goldilocks, len(predict_goldilocks)))
		print('%-3g false positives of %g candidates'  %(predicted_false_positive_goldilocs, len(predict_goldilocks)))

		# Plotting a bar plot of candidates predicted as confirmed and false positives
		# Need to fix input title, labels etc maybe?
		func.Histogram2(predict_goldilocks[:,1], 'Logistic regression (Goldilock)', threshold)

		GL.GoldilocksZone(predict_goldilocks[:,1], 'Logistic regression', threshold)
