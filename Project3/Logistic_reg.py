"""
FYS-STK4155 - Project 3: Logistic regression
"""
import numpy 			 	 	as np
import functions 				as func

from random 				 	import random, seed
from sklearn.metrics 		 	import classification_report, f1_score,		\
									   precision_score, recall_score,       \
                                       accuracy_score, mean_squared_error,  \
                                       confusion_matrix
#from sklearn.model_selection 	import train_test_split
from sklearn.linear_model 		import LogisticRegression #, SGDClassifier
#------------------------------------------------------------------------------

def LogReg(X_train, X_test, y_train, y_test, candidates):

	# Make Logistic regression analysis
	logreg = LogisticRegression()
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

	print(confusion_matrix(y_test, y_pred))

	pred_cand  = np.array(logreg.predict(candidates)) 	# Predict candidates

	# Divide into predicted false positives and confirmed exoplanets
	pred_FP    = (pred_cand == 0).sum() 	# Predicted false positives
	pred_Conf  = (pred_cand == 1).sum() 	# Predicted exoplanets/confirmed

	# Information print to terminal
	print('\nThe Neural Network Classifier predicted')
	print('--------------------------------------')
	print('%-5g exoplanets      of %g candidates'  %(pred_Conf, len(pred_cand)))
	print('%-5g false positives of %g candidates'  %(pred_FP, len(pred_cand)))


	# Make AUC curve?
