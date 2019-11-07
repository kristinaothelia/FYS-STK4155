"""
Functions used for project 2 in FYS-STK4155
"""

import sys 

import numpy 		 as np
import pandas 		 as pd
import scikitplot    as skplt

import matplotlib.pyplot as plt

from sklearn.preprocessing       import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.metrics 			 import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection     import train_test_split
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


def IndicatorFunc():
	pass


def accuracy(model, y):
	"""
	Takes in the output of the Logistic Regression code (y),
	and use them as targets in the Indicator function

	Accuracy: The proportion of the total number of predictions that are correct

	"""

	#model[model < 0.5] = 0
	#model[model >= 0.5] = 1 

	model = np.ravel(model)
	y = np.ravel(y)

	
	for i in range(len(model)):
		if model[i] < 0.5:
			model[i] = 0
		else:
			model[i] = 1

	accuracy = np.mean(y == model)
	#accuracy = np.sum(y == model)/len(y)
	return accuracy

def precision():
	"""
	The proportion of positive predictions that are actually correct
	"""
	pass

def recall():
	"""
	The proportion of actual defaulters that the model will correctly predict as such
	"""
	pass

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

def steepest(X, y, gamma, iterations=1000):   # DONT WORK 
	# Steepest ??
	n = len(X[0])
	#epsilon = 1e-8

	beta = np.random.randn(len(X[0]), 1)

	for i in range(iterations):
		grad_beta_C = beta_gradients(X, y, beta)
		beta -= beta - gamma * grad_beta_C
	return beta

def learning_schedule(t, t0=5, t1=50):
	ls = t0/(t+t1)
	return ls

def next_beta(X, y, eta, gamma):
	"""
	Calculating the beta values
	"""

	# Stochastic Gradient Descent, shuffle?
	beta = np.random.randn(len(X[0]), 1)
	n = len(X)
	M = 80 #0.05*n  	         # Size of each minibatch, should be smaller than n
	m = int(n/M)   	         # Number of minibatches
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


def feed_forward_train(X):
	"""
	From lecture PP on neural networks, from Morten
	"""

	# Weighted sum of inputs to the hidden layer
	z_h = np.matmul(X, hidden_weights) + hidden_bias

	# Activation in the hidden layer
	a_h = logistic_function(z_h)

	# Weighted sum of inputs to the output layer
	z_o = np.matmul(a_h, output_weights) + output_bias

	# Softmax output ??
	# Axis 0 holds each input and axis 1 the probabilities of each category
	exp_term = np.exp(z_o)
	probabilities = probabilities(exp_term)

	# For backpropagation need activations in hidden and output layers
	return a_h, probabilities


def BackPropagation(lamb):
	"""
	Back propagation code for a multilayer perceptron model, from Morten
	"""

	output_weights, output_bias, hidden_weights, hidden_bias = weights_bias(eta, lamb)

	a_h, p = feed_forward_train(X)

	# Error in the output layer
	error_output = p - y

	# Error in the hidden layer
	error_hidden = np.matmul(error_output, output_weights.T)*a_h*(1-a_h)

	# Gradients for the output layer
	output_weights_gradient = np.matmul(X.T, error_hidden)
	output_bias_gradient = np.sum(error_hidden, axis=0)

	# Gradients for the hidden layer
	hidden_weights_gradient = np.matmul(X.T, error_hidden)
	hidden_bias_gradient = np.sum(error_hidden, axis=0)

	return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient



def weights_bias(eta, lamb):

	#eta = 0.01
	#lmbd = 0.01

	for i in range(1000):

		# Calculating the gradients
		dWo, dBo, dWh, dBh = BackPropagation(X_train, Y_train_onehot)

		# Calculating regularization term gradients
		dWo += lmbd * output_weights
		dWh += lmbd * hidden_weights


		# Update weights and biases
		output_weights -= eta * dWo
		output_bias -= eta * dBo
		hidden_weights -= eta * dWh
		hidden_bias -= eta * dBh

	#print("New accuracy on training data: " + str(accuracy_score(predict(X_train), Y_train)))
	return output_weights, output_bias, hidden_weights, hidden_bias


def activation_function(X):
	"""
	Activation function
	Inputs x_i are the outputs of the neurons in the preceding layer
	"""
	z = np.sum(w*x+b)
	return z

