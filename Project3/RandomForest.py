"""
FYS-STK4155 - Project 3: Random Forest
"""
import os
import numpy 				 as np
import matplotlib.pyplot	 as plt
import scikitplot        	 as skplt
import functions         	 as func
import goldilock             as GL
import seaborn               as sns
import pandas 				 as pd

from sklearn.impute 		 import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble 		 import RandomForestClassifier
#------------------------------------------------------------------------------


def Best_params(seed, X_train, y_train):

	param_test = {"n_estimators": 	  [200, 300, 400, 500],
				  "max_features": 	  [None, 'auto', 'log2'],
				  "max_depth": 	 	  [7, 8, 9],
				  "min_samples_leaf": [1, 2, 5, 10]}
	'''
	param_test = {"n_estimators": 	  [200, 300, 400, 500],
				  "max_features": 	  ['auto', 'log2'],
				  "max_depth": 	 	  [7, 8, 9, None],
				  "min_samples_leaf": [1, 2, 3, 4]}
	'''
	# {'max_depth': 8, 'max_features': 'auto', 'min_samples_leaf': 1, 'n_estimators': 300}

	gsearch = GridSearchCV(RandomForestClassifier(), param_grid = param_test, cv=5)
	gsearch.fit(X_train, y_train)
	print(gsearch.best_params_)



def Random_Forest(X_train, X_test, y_train, y_test, candidates, GoldiLock,	\
				  feature_list, header_names, seed=0, threshold=0.5, 		\
				  plot_confuse_matrix=False, Goldilock_zone=False):
	"""
	Ha en input oversikt...?

	threshold		| 0.5 == RF.predict
					  0.7 --> 	Need 70$%$ probability to be an exoplanet, to
					  			be calssified as an exoplanet
	"""
	print("Exoplanet threshold = %g" % threshold)

	Best_params(seed, X_train, y_train)

	# Plot error against number of trees?

	RF = RandomForestClassifier(n_estimators	 = 300,
								max_features     = 'auto',
								max_depth		 = 8,
								min_samples_leaf = 1,
								random_state	 = seed,
								criterion		 = 'gini',   # 'entropy'
								bootstrap        = True
								)
	RF.fit(X_train,y_train)

	# https://github.com/erykml/medium_articles/blob/master/Machine%20Learning/feature_importance.ipynb

	header_names = np.load('feature_names.npy', allow_pickle=True)

	# function for creating a feature importance dataframe
	def feature_importance(column_names, importances):
		df = pd.DataFrame({'feature': column_names,'feature_importance': importances}) \
		.sort_values('feature_importance', ascending = False) \
		.reset_index(drop = True)
		return df

	# plotting a feature importance dataframe (horizontal barchart)
	def feature_importance_plot(feature_importances, title):
		feature_importances.columns = ['feature', 'feature_importance']
		sns.barplot(x = 'feature_importance', y = 'feature', data = feature_importances, orient = 'h', color = 'royalblue') \
		.set_title(title, fontsize = 15)
		plt.ylabel('feature', fontsize=15)
		plt.xlabel('feature importance', fontsize=15)
		plt.show()

	feature_imp = feature_importance(header_names[1:], RF.feature_importances_)
	feature_importance_plot(feature_imp[:11], "Feature Importance")

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

	#predict_candidates       = np.array(RF.predict(candidates))
	#predicted_false_positive = (predict_candidates == 0).sum()
	#predicted_exoplanets     = (predict_candidates == 1).sum()

	predict_candidates = np.array(RF.predict_proba(candidates))

	predict_candidates[:,0] = (predict_candidates[:,0] < threshold).astype('int')
	predict_candidates[:,1] = (predict_candidates[:,1] >= threshold).astype('int')

	predicted_false_positive = (predict_candidates[:,1] == 0).sum()
	predicted_exoplanets     = (predict_candidates[:,1] == 1).sum()


	# Information print to terminal
	print('\nThe Random Forest Classifier predicted')
	print('--------------------------------------')
	print('%-5g exoplanets      of %g candidates'  %(predicted_exoplanets, len(predict_candidates)))
	print('%-5g false positives of %g candidates'  %(predicted_false_positive, len(predict_candidates)))

	if plot_confuse_matrix == True:
		# Plotting a bar plot of candidates predicted as confirmed and false positives
		# Need to fix input title, labels etc maybe?
		func.Histogram2(predict_candidates, 'Random Forest (Candidates)')

		# func.Histogram2(g=df.loc[:, (df.columns == 'koi_disposition')].values)

	if Goldilock_zone:

		print("")
		print("Goldilock zone calculations")

		predict_goldilocks = np.array(RF.predict_proba(GoldiLock))

		predict_goldilocks[:,0] = (predict_goldilocks[:,0] < threshold).astype('int')
		predict_goldilocks[:,1] = (predict_goldilocks[:,1] >= threshold).astype('int')
		#np.save('GoldiLock_predicted', predict_goldilocks)

		predicted_false_positive_goldilocs  = (predict_goldilocks[:,1] == 0).sum()
		predicted_exoplanets_goldilocks     = (predict_goldilocks[:,1] == 1).sum()

		# Information print to terminal
		print('\nThe Random Forest Classifier predicted')
		print('--------------------------------------')
		print('%-3g exoplanets      of %g GL candidates' %(predicted_exoplanets_goldilocks, len(predict_goldilocks)))
		print('%-3g false positives of %g GL candidates' %(predicted_false_positive_goldilocs, len(predict_goldilocks)))

		# Plotting a bar plot of candidates predicted as confirmed and false positives
		func.Histogram2(predict_goldilocks[:,1], 'Random Forest (Goldilock)')

		GL.GoldilocksZone(predict_goldilocks[:,1])


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
