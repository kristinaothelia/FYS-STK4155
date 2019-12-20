"""
FYS-STK4155 - Project 3: Random Forest
"""
import os
import numpy 				 as np
import matplotlib.pyplot	 as plt
import scikitplot        	 as skplt
import functions         	 as func
import goldilock             as GL

from sklearn.impute 		 import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble 		 import RandomForestClassifier
#------------------------------------------------------------------------------

def Random_Forest(X_train, X_test, y_train, y_test, candidates, GoldiLock,	\
				  feature_list, header_names, seed=0, 						\
				  plot_confuse_matrix=False, Goldilock_zone=False):

	#from sklearn import tree
	#pd.DataFrame(X).fillna()
	# grid search

	# Plot error against number of trees?
	RF = RandomForestClassifier(n_estimators	= 100,
								max_depth		= None,
								random_state	= seed,
								criterion		= 'gini')
	RF.fit(X_train,y_train)

	# Calculating different metrics
	predict     = RF.predict(X_test)
	accuracy 	= RF.score(X_test,y_test)
	precision   = func.precision(y_test, predict)
	recall      = func.recall(y_test, predict)
	F1_score    = func.F1_score(y_test, predict)

	# Calculate the absolute errors
	errors = abs(predict - y_test)

	# Printing the different metrics:
	func.Print_parameters(accuracy, F1_score, precision, recall, errors, name='Random Forest')

	if plot_confuse_matrix == True:
		skplt.metrics.plot_confusion_matrix(y_test, predict)
		plt.show()

	#print(RF.decision_path(X_test))

	# Pull out one tree from the forest
	tree_nr = 5
	tree 	= RF.estimators_[tree_nr]

	#print(len(feature_list), len(header_names)) # 24 og 56
	func.PlotOneTree(tree, feature_list) # header_names?

	predict_candidates       = np.array(RF.predict(candidates))
	predicted_false_positive = (predict_candidates == 0).sum()
	predicted_exoplanets     = (predict_candidates == 1).sum()

	# Information print to terminal
	print('\nThe Random Forest Classifier predicted')
	print('--------------------------------------')
	print('%-5g exoplanets      of %g candidates'  %(predicted_exoplanets, len(predict_candidates)))
	print('%-5g false positives of %g candidates'  %(predicted_false_positive, len(predict_candidates)))

	if plot_confuse_matrix == True:
		# Plotting a bar plot of candidates predicted as confirmed and false positives
		# Need to fix input title, labels etc maybe?
		func.Histogram2(predict_candidates)


	if Goldilock_zone:

		print("Goldilock zone calculations")

		predict_goldilocks = np.array(RF.predict(GoldiLock))
		np.save('GoldiLock_predicted', predict_goldilocks)

		predicted_false_positive_goldilocs  = (predict_goldilocks == 0).sum()
		predicted_exoplanets_goldilocks     = (predict_goldilocks == 1).sum()

		# Information print to terminal
		print('\nThe Random Forest Classifier predicted')
		print('--------------------------------------')
		print('%-3g exoplanets      of %g candidates'  %(predicted_exoplanets_goldilocks, len(predict_goldilocks)))
		print('%-3g false positives of %g candidates'  %(predicted_false_positive_goldilocs, len(predict_goldilocks)))

		# Plotting a bar plot of candidates predicted as confirmed and false positives
		# Need to fix input title, labels etc maybe?
		func.Histogram2(predict_goldilocks)

		GL.GoldilocksZone()


	'''
	feature_importance = RF.feature_importances_
	print(feature_importance)
	print(len(feature_importance))


	#for i in range(len(feature_importance)):
		# Check the i in feature_importance
		# assign corresponding header name


	plt.hist(feature_importance, align='left', histtype='bar', orientation='horizontal', rwidth=0.3)
	plt.title('Feature Importance')
	plt.xlabel('--')
	plt.ylabel('--')
	#plt.xlim([lb-width/2, ub-width/2])
	plt.show()
	'''

	'''
	from matplotlib.ticker import MaxNLocator
	plt.barh(feature_importance, width=0.1,  align='center')
	plt.grid(True, linestyle='--', which='major',color='grey', alpha=.25)
	plt.show()
	'''