"""
FYS-STK4155 - Project 3:
A file that contains various functions used in the project
"""
import seaborn             as sns
import numpy               as np
import pandas              as pd
import matplotlib.pyplot   as plt
import pydot

from sklearn.tree import export_graphviz
#------------------------------------------------------------------------------

def TRUE_FALSE_PREDICTIONS(y, model):
	"""
	Calculates the proportion of the predictions that are true and false
	"""

	TP = 0  # True  Positive
	FP = 0  # False Positive
	TN = 0  # True  Negative
	FN = 0  # False Negative

	for i in range(len(model)):

		# Negative: False positive (not exoplanet)
		if model[i] == 0:
			if y[i] == 0:
				TN += 1
			else:
				FN +=1

		# Positive: Exoplanet
		elif model[i] == 1:
			if y[i] == 1:
				TP +=1
			else:
				FP += 1

	return TP, FP, TN, FN

def precision(y, model):
	"""
	The proportion of positive predictions that are actually correct
	Often used to: limit the number of false positives (FP)
	"""

	TP, FP, TN, FN = TRUE_FALSE_PREDICTIONS(y, model)
	precision = TP/(TP+FP)
	return precision

def recall(y, model):
	"""
	The proportion of actual defaulters that the model will correctly predict as such
	TPR: True positive rate (also called recall or sensitivity)
	"""

	TP, FP, TN, FN = TRUE_FALSE_PREDICTIONS(y, model)
	TPR = TP/(TP+FN)
	return TPR

def F1_score(y, model):
	"""
	Calculates the F1_score using the precision and recall
	"""
	p = precision(y, model)
	r = recall(y, model)
	f = 2*((p*r)/(p+r))
	return f

def Print_parameters(accuracy, f1_score, precision, recall, errors, name=''):
	print('\nMethod: %s' %name)
	print('-------------------------------------------')
	print('The accuracy is    : %.3f' % accuracy)
	print('The F1 score is    : %.3f' % f1_score)
	print('The precision is   : %.3f' % precision)
	print('The recall is      : %.3f' % recall)
	print('The absolute error : %.3f' % np.mean(errors))
	print('-------------------------------------------')

# Plotting functions
#------------------------------------------------------------------------------

def Histogram2(g):

	labels, counts = np.unique(g, return_counts=True)

	#plt.style.use('dark_background')
	plt.bar(labels, counts, align='center', color='purple')
	plt.gca().set_xticks(labels)
	plt.ylabel("Observations count")
	plt.title("Kepler's objects of interest")
	plt.show()


def PlotOneTree(tree, feature_list):

	# Export the image to a dot file
	export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)

	# Use dot file to create a graph
	(graph, ) = pydot.graph_from_dot_file('tree.dot')

	# Write graph to a png file
	graph.write_png('tree.png')
