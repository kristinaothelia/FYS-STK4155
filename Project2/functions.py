"""
Functions used for project 2 in FYS-STK4155
"""
import numpy as np

from sklearn.preprocessing       import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.metrics 			 import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection     import train_test_split
# -----------------------------------------------------------------------------
np.random.seed(0)
seed  = 1
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
	#t = y
	#n = len(t)
	#indicator = IndicatorFunc(t)
	#accuracy  = np.sum(indicator)/n

	for i in range(len(model)):
		if model[i] < 50:
			model[i] = 0
		else:
			model[i] = 1

	accuracy = np.mean(y == model)
	return accuracy

def precision():
	"""
	The proportion of positive predictions that are actually correct
	"""
	pass

def recall():
	"""
	The proportion of actual defalters that the model will correctly predict as such
	"""
	pass

def probabilities(exp_term):
	"""
	Returning probabilities?
	"""
	p = exp_term/np.sum(exp_term, axis=1, keepdims=True)
	return p


def cost(y_train, z):
	"""
	The cost function
	"""

	print(y_train.dtype)
	print(z.dtype)

	sigmoid = logistic_function(z)
	#c = -np.sum(y_train.T).dot(np.log(sigmoid)) - (1-y_train).T.dot(np.log(1-sigmoid)) # minus or plus on second term???
	# fix (1-sigmoid)
	#cost = -np.sum(np.transpose(y_train)@np.log(sigmoid) - np.transpose(1-y_train)@np.log(1-sigmoid))
	cost = -np.sum(np.transpose(y_train)@np.log(sigmoid) - np.transpose(1-y_train)@np.log(1-sigmoid))
	return c


def derivative_cost_loss(X, y, beta):
	#grad_beta_C = 2*X.T.dot((X.dot(beta)-y))  linear???
	sigmoid = 1/(1+np.exp(-(X)@(beta)))
	grad_beta_C = -(np.transpose(X)@(sigmoid-y))
	return grad_beta_C


def steepest():
	# Steepest ??
	n = len(X[0])
	epsilon = 1e-8
	iterations = 1000

	beta = np.random.randn(n, 1)
	print(len(beta))

	for i in range(iterations):
		grad_beta_C = derivative_cost_loss(X, y, beta)
		beta -= eta - gamma * grad_beta_C
	return beta

def next_beta(X, y, eta, gamma):
	"""
	"""

	# Stochastic Gradient Descent, shuffle
	beta = np.random.randn(len(X[0]), 1)
	n = len(X)    #100 datapoints
	M = 100       #size of each minibatch
	m = int(n/M)  #number of minibatches
	n_epochs = 10 #number of epochs

	j = 0

	for epoch in range(1,n_epochs+1):
	    for i in range(m):
	        k = np.random.randint(m) #Pick the k-th minibatch at random
	        #Compute the gradient using the data in minibatch Bk
	        grad_beta_C = derivative_cost_loss(X, y, beta)
	        #print(grad_beta_C)
	        beta -= eta - gamma * grad_beta_C
	        #Compute new suggestion for
	        j += 1

	return beta


def cost_minimized():  ## same as above??
	pass

def Create_ConfusionMatrix(y_pred_test, y_test):
	"""
	Using scikit to create confusion matrix
	"""
	CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
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


"""
def ConfusionMatrix():

	pass
"""
