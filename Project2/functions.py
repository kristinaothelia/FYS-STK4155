"""
Functions used for project 2 in FYS-STK4155
"""
import sys
import numpy 				 as np
import pandas 		 		 as pd
import scikitplot    		 as skplt
import matplotlib.pyplot 	 as plt

from sklearn.preprocessing   import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.metrics 		 import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.special 			 import expit
from sklearn.metrics 	     import confusion_matrix, accuracy_score, roc_auc_score, auc, roc_curve, recall_score, precision_score, f1_score
from sklearn.linear_model    import LogisticRegression
# -----------------------------------------------------------------------------
seed = 0
np.random.seed(seed)
# -----------------------------------------------------------------------------

def logistic_function(z):
	"""
	The Logistic Regression function
	The probability that a data point x_i belongs to a category y_i={0,1} is
	given by the Sigmoid function (gives the likelihood for an event)
	It is useful in neural networks for assigning weights on a relative scale.
	The value z is the weighted sum of parameters involved in the learning algorithm
	"""
	g = 1/(1+np.exp(-z))
	#g = np.exp(z)/(1+np.exp(z))
	return g


def IndicatorFunc(model, threshold=0.5):
	model[model < threshold] = 0
	model[model >= threshold] = 1
	return np.ravel(model)


def accuracy(model, t):
	"""
	Accuracy: The proportion of the total number of predictions that are correct
	Takes in the output of the Logistic Regression code (t), and the predicted values (model)
	"""
	t        = np.ravel(t) # target

	accuracy = np.mean(t == model)
	#TP, FP, TN, FN = TRUE_FALSE_PREDICTIONS(t, model)
	#accuracy = (TP+TN)/(TP+TN+FP+FN)
	return accuracy

def AUC_ROC(model, y, tpr, fpr):
	"""
	Not working
	"""

	model = np.ravel(model)
	auc = 0.0
	height = 0.0

	#auc += (p1[0] - p0[0]) * ((p1[1] + p0[1]))/ 2  #if trapezoid else p0[1])

	'''
	for i in range(len(model)):
		if model[i] == 1.0:
			height = height + fpr
		else:
			auc = auc + height * tpr
	'''
	return np.mean(auc)

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

	#print(TP, FP, TN, FN)
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


def probabilities(ytilde):
	"""
	Returning probabilities?
	"""
	exp_term = np.exp(ytilde)
	#p = exp_term/np.sum(exp_term, axis=1, keepdims=True)
	p = np.exp(-ytilde)/(np.exp(-ytilde)+1)
	return p


def cost(X, y, beta):
	"""
	The cost function
	"""

	z = X @ beta
	sigmoid = logistic_function(z)
	#c = -np.sum(y_train.T).dot(np.log(sigmoid)) - (1-y_train).T.dot(np.log(1-sigmoid)) # minus or plus on second term???
	# fix (1-sigmoid)
	#cost = -np.sum(np.transpose(y)@np.log(sigmoid) - np.transpose(1-y)@np.log(1+sigmoid))
	cost = -np.sum(np.transpose(y)@np.log(sigmoid) - np.transpose(1-y)@np.log(logistic_function(-z)))
	return cost


def beta_gradients(X, y, beta):
	'''
	Calculating the gradients
	'''
	#grad_beta_C = 2*X.T.dot((X.dot(beta)-y))  linear???
	#sigmoid = 1/(1+np.exp(-(X)@(beta)))
	z = X @ beta
	sigmoid = logistic_function(z)
	grad_beta_C = -(np.transpose(X)@(sigmoid-y))

	#der_sig = logistic_function(z)*(1-logistic_function(z))
	#grad_beta_C = 2*(y-sigmoid) *der_sig*X.T

	return grad_beta_C

def steepest(Xf, yf, gamma=0.001, iterations=1000):   # DONT WORK, be happy
    """
    # Steepest ??
    n = len(X[0])
    #epsilon = 1e-8

    beta = np.random.randn(len(X[0]), 1)

    for i in range(iterations):
	    grad_beta_C = beta_gradients(X, y, beta)
	    beta -= beta - gamma * grad_beta_C
    """
    K = len(Xf[0,:])
    beta = np.random.randn(K, 1)
    for i in range(iterations):
        t = Xf@beta
        sigmoid = expit(t)
        #print(sigmoid)
        #siggy = 1./(1 + np.exp(t))
        #loss = yf - sigmoid
        #print("iteration %g, cost: %f" % (i, loss))
        grad = 2/K*Xf.T@(sigmoid - yf)
        beta = beta - gamma*grad
        #cost = -np.sum(np.transpose(yf)@np.log(1 + siggy) - np.transpose(1-yf)@np.log(siggy))
        #print(cost)
        #print(i)
        #break
    return beta

def learning_schedule(t, t0=5, t1=50):
	ls = t0/(t+t1)
	return ls

def SGD_beta(X, y, eta, gamma):
	"""
	Calculating the beta values
	"""

	# Stochastic Gradient Descent, shuffle?
	beta = np.random.randn(len(X[0]), 1)
	n = len(X)
	M = 80 #0.05*n  	         # Size of each minibatch, should be smaller than n
	m = int(n/M)   	             # Number of minibatches
	n_epochs = 100      		 # Nmber of epochs

	acc = np.zeros(n_epochs+1)
	epoch_list = np.zeros(n_epochs+1)

	#z_i = np.zeros(m)
	#model_i = np.zeros(m)
	#y_i = np.zeros(m)

	for epoch in range(1,n_epochs+1):
		for i in range(m):

			random_index = np.random.randint(m)    #Pick the k-th minibatch at random
			xi = X[random_index:random_index+1]
			yi = y[random_index:random_index+1]

			#Compute the gradient using the data in minibatch Bk
			grad_beta_C = beta_gradients(xi, yi, beta)
			beta -= eta - gamma * grad_beta_C

			#y_i[i] = yi
			#z_i[i] = xi@beta
			#model_i[i] = logistic_function(z_i[i])

		#acc[epoch] = accuracy(model_i, y_i)
		#epoch_list[epoch] = epoch

	return beta

def cost_minimized():  ## same as above??
	pass

def Create_ConfusionMatrix(y_pred_test, y_test, plot=False):
	"""
	Using scikit to create confusion matrix
	"""
	CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
	if plot == True:
		skplt.metrics.plot_confusion_matrix(y_test, y_pred_test)
		plt.show()
	return CM

def ConfusionMatrix_DataFrame(CM, labels=['pay', 'default']):
	"""
	Confusion Matrix with pandas dataframe
	"""
	df = pd.DataFrame(data=CM, index=labels, columns=labels)

	df.index.name   = 'True'
	df.columns.name = 'Prediction'
	df.loc['Total'] = df.sum()
	df['Total']     = df.sum(axis=1)

	return df

def metrics():
	"""
	Creating a pandas dataframe for evaluation metrics
	"""
	metrics = pd.DataFrame(index=['accuracy', 'precision', 'recall'],
						   columns=['LogisticReg', 'NeuralNetwork'])
	return metrics

def bestCurve(y):
	'''
	Calculating the best curve
	'''

	defaults = sum(y == 1)
	total = len(y)

	x = np.linspace(0, 1, total)
	y1 = np.linspace(0, 1, defaults)
	y2 = np.ones(total-defaults)
	y3 = np.concatenate([y1,y2])

	return x, y3

def splitting(X, y, TrainingShare=0.5, seed=0):
	"""
	Splitting the data in a train set and a test set
	"""

	#target_name = 'default'
	#X = default.drop('default', axis=1)
	#robust_scaler = RobustScaler()
	#sc = StandardScaler()
	#X = robust_scaler.fit_transform(X)
	#y = default[target_name]

	# Train-test split
	X_train, X_test, y_train, y_test=train_test_split(X, y, train_size=TrainingShare, test_size = 1-TrainingShare, random_state=seed)
	return X_train, X_test, y_train, y_test

def MeanSquaredError(y_data, y_model):
	"""
	Function to calculate the mean squared error (MSE) for our model
	Input y_data	| Function array
	Input y_model	| Predicted function array
	"""
	n   = np.size(y_model)
	MSE = (1/n)*np.sum((y_data-y_model)**2)

	return MSE

def R2_ScoreFunction(y_data, y_model):
	"""
	Function to calculate the R2 score for our model
	Input y_data	| Function array
	Input y_model	| Predicted function array
	"""
	counter     = np.sum((y_data-y_model)**2)
	denominator = np.sum((y_data-np.mean(y_data))**2)
	R_2          = 1 - (counter/denominator)

	return R_2


def scikit(X_train, X_test, y_train, y_test, model):
	# A logistic regression model with scikit-learn
	logReg = LogisticRegression(random_state=seed, solver='sag', max_iter=1000, fit_intercept=False) # solver='lbfgs'
	logReg.fit(X_train, np.ravel(y_train))

	ypredict_scikit      = logReg.predict(X_test)
	predict_proba_scikit = logReg.predict_proba(X_test)  # Probability estimates
	acc_score_scikit	 = logReg.score(X_test, y_test)  # Accuracy
	acc_scikit 			 = accuracy_score(y_pred=ypredict_scikit, y_true=y_test)
	fpr, tpr, thresholds = roc_curve(y_test, predict_proba_scikit[:,1], pos_label=None)
	AUC_scikit 			 = auc(fpr, tpr)
	#AUC_scikit2 		 = roc_auc_score(y_test, predict_probabilities_scikit[:,1])
	TPR_scikit 			 = recall_score(y_test, model)
	precision_scikit     = precision_score(y_test, model)
	f1_score_scikit      = f1_score(y_test, np.ravel(model).astype(np.int64))

	return acc_scikit, TPR_scikit, precision_scikit, f1_score_scikit, AUC_scikit, predict_proba_scikit


def activation_function(X):
	"""
	Activation function
	Inputs x_i are the outputs of the neurons in the preceding layer
	"""
	z = np.sum(w*x+b)
	return z
