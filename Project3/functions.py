


def TRUE_FALSE_PREDICTIONS(y, model):
	"""
	Calculates the proportion of the predictions that are true and false
	"""

	TP = 0  # True  Positive
	FP = 0  # False Positive
	TN = 0  # True  Negative
	FN = 0  # False Negative

	for i in range(len(model)):

		# Negative: pay
		if model[i] == 0:
			if y[i] == 0:
				TN += 1
			else:
				FN +=1

		# Positive: default
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