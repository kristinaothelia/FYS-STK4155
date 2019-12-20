"""
FYS-STK4155 - Project 3: Neural Network Classification
"""
import sys
import numpy 			 	 	as np
import seaborn 			 	 	as sns
import matplotlib.pyplot 	 	as plt
import scikitplot       		as skplt

from sklearn.preprocessing 		import StandardScaler
from sklearn.neural_network  	import MLPClassifier
from sklearn.metrics 		 	import classification_report, f1_score,		\
									   precision_score, recall_score,       \
                                       accuracy_score, mean_squared_error,  \
                                       mean_absolute_error, confusion_matrix
from sklearn.model_selection 	import train_test_split, GridSearchCV
from random 				 	import random, seed
from mpl_toolkits.mplot3d 	 	import Axes3D
from matplotlib 			 	import cm
from matplotlib.ticker 		 	import LinearLocator, FormatStrFormatter

import functions 				as func
import goldilock             	as GL
#------------------------------------------------------------------------------

def NeuralNetwork(X_train, X_test, y_train, y_test, candidates, GoldiLock, seed, Goldilock_zone=False, plot_confuse_matrix=False):

	model = MLPClassifier(max_iter=1000, random_state=seed)
	"""
	model = MLPClassifier(solver 			= 'lbfgs',	# 'adam'
						activation			= 'logistic',
						hidden_layer_sizes  = 100,		# (200,150,100)
						random_state=seed,
						max_iter			= 1000)		# 1500
	"""

	trained_model = model.fit(X_train, y_train)
	"""
	param_test = {"hidden_layer_sizes": [130, 140, 150],
				  "learning_rate_init": [ 0.125, 0.150, 0.175]}


	gsearch = GridSearchCV(MLPClassifier(solver='lbfgs', activation='logistic', max_iter=1500, random_state=seed), param_grid = param_test, cv=5)
	trained_model = gsearch.fit(X_train, y_train)
	print('aaaaaaaaaaaaaaa')
	print(trained_model.best_params_)
	print('aaaaaaaaaaaaaaa')
	"""

	# Calculating different metrics
	predict     = trained_model.predict(X_test)
	accuracy 	= accuracy_score(y_test, predict)
	precision   = precision_score(y_test, predict, average="macro")
	recall      = recall_score(y_test, predict, average="macro")
	F1_score    = f1_score(y_test, predict, average="macro")

	# Calculate the absolute errors
	errors = abs(predict - y_test)

	# Printing the different metrics:
	func.Print_parameters(accuracy, F1_score, precision, recall, errors, name='Neural Network classification')


	print(confusion_matrix(y_test, predict))

	predict_candidates       = np.array(trained_model.predict(candidates))

	predicted_false_positive = (predict_candidates == 0).sum()
	predicted_exoplanets     = (predict_candidates == 1).sum()

	# Information print to terminal
	print('\nThe Neural Network Classifier predicted')
	print('--------------------------------------')
	print('%-5g exoplanets      of %g candidates'  %(predicted_exoplanets, len(predict_candidates)))
	print('%-5g false positives of %g candidates'  %(predicted_false_positive, len(predict_candidates)))


	if Goldilock_zone:

		print("Goldilock zone calculations")

		predict_goldilocks = np.array(trained_model.predict(GoldiLock))
		np.save('GoldiLock_predicted', predict_goldilocks)

		predicted_false_positive_goldilocs  = (predict_goldilocks == 0).sum()
		predicted_exoplanets_goldilocks     = (predict_goldilocks == 1).sum()

		# Information print to terminal
		print('\nThe Neural Network Classifier predicted')
		print('--------------------------------------')
		print('%-5g exoplanets      of %g candidates' %(predicted_exoplanets_goldilocks, len(predict_goldilocks)))
		print('%-5g false positives of %g candidates' %(predicted_false_positive_goldilocs, len(predict_goldilocks)))

		# Plotting a bar plot of candidates predicted as confirmed and false positives
		# Need to fix input title, labels etc maybe?
		func.Histogram2(predict_goldilocks, method='NN')

		GL.GoldilocksZone()
