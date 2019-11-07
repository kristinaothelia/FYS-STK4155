"""
Neural Network code for project 2 in FYS-STK4155
"""
import numpy as np
import functions as func

from sklearn.model_selection import train_test_split
# -----------------------------------------------------------------------------
#np.random.seed(0)
#seed  = 1
# -----------------------------------------------------------------------------

# Inputs : X, y, eta, lamb, minibatch_size, epochs, n_boots, nodes
# hidden_layers = len(nodes)
# n_boots for bootstrap

#def weights_biases():
#    pass

def create_biases_and_weights(n_features, n_hidden_neurons, n_categories):
	"""

	"""

	hidden_weights = np.random.randn(n_features, n_hidden_neurons)
	hidden_bias    = np.zeros(n_hidden_neurons) + 0.01

	output_weights = np.random.randn(n_hidden_neurons, n_categories)
	output_bias    = np.zeros(n_categories) + 0.01

	return hidden_weights, hidden_bias, output_weights, output_bias


def feed_forward_train(X, n_features, n_hidden_neurons, n_categories):
	"""
	From lecture PP on neural networks, from Morten
    X : features
	"""
	hidden_weights, hidden_bias, output_weights, output_bias = create_biases_and_weights(n_features, n_hidden_neurons, n_categories)
	#Make  z_h and a_h lists??

	z_h = np.matmul(X, hidden_weights) + hidden_bias	    # Weighted sum of inputs to the hidden layer
	a_h = func.logistic_function(z_h)						# Activation in the hidden layer

	# Weighted sum of inputs to the output layer
	z_o = np.matmul(a_h, output_weights) + output_bias

	# Softmax output ??
	# Axis 0 holds each input and axis 1 the probabilities of each category
	exp_term = np.exp(z_o)
	probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

	# For backpropagation need activations in hidden and output layers
	return a_h, probabilities

def feed_forward_out(X, y, eta, lamb, n_features, n_hidden_neurons, n_categories):
	output_weights, output_bias, hidden_weights, hidden_bias = back_propagation(X, y, eta, lamb, n_features, n_hidden_neurons, n_categories)
	#create_biases_and_weights(n_features, n_hidden_neurons, n_categories)
	# feed-forward for output
	z_h = np.matmul(X, hidden_weights) + hidden_bias
	a_h = func.logistic_function(z_h)

	z_o = np.matmul(a_h, output_weights) + output_bias

	exp_term = np.exp(z_o)
	probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
	return probabilities

def back_propagation(X, y, eta, lamb, n_features, n_hidden_neurons, n_categories):
	"""
	Back propagation code for a multilayer perceptron model, from Morten
	"""

	output_weights, output_bias, hidden_weights, hidden_bias = create_biases_and_weights(n_features, n_hidden_neurons, n_categories)

	a_h, p = feed_forward_train(X, n_features, n_hidden_neurons, n_categories)

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

	
	if lamb > 0.0:
		output_weights_gradient += lamb * output_weights
		hidden_weights_gradient += lamb * hidden_weights

	output_weights -= eta * output_weights_gradient
	output_bias -= eta * output_bias_gradient
	hidden_weights -= eta * hidden_weights_gradient
	hidden_bias -= eta * hidden_bias_gradient
	

	return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient

def predict(X, y, eta, lamb, n_features, n_hidden_neurons, n_categories):
        probabilities = feed_forward_out(X, y, eta, lamb, n_features, n_hidden_neurons, n_categories)
        return np.argmax(probabilities, axis=1)

def predict_probabilities(X, y, eta, lamb, n_features, n_hidden_neurons, n_categories):
    probabilities = feed_forward_out(X, y, eta, lamb, n_features, n_hidden_neurons, n_categories)
    return probabilities

def train(X, y, eta, lamb, n_inputs, epochs, iterations, batch_size, n_features, n_hidden_neurons, n_categories):
	data_indices = np.arange(n_inputs)
	X_data_full = X
	Y_data_full = y
	for i in range(epochs):
		for j in range(iterations):
			# pick datapoints with replacement
			chosen_datapoints = np.random.choice(data_indices, size=batch_size, replace=False)

			# minibatch training data
			X_data = X_data_full[chosen_datapoints]
			Y_data = Y_data_full[chosen_datapoints]

			# Tror dette blir helt feil...
			probabilities = feed_forward_train(X_data, n_features, n_hidden_neurons, n_categories)
			output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient = back_propagation(X_data, Y_data, eta, lamb, n_features, n_hidden_neurons, n_categories)

			return probabilities, output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient


	def cost_derivative(self, output_activations, labels):
		return output_activations-labels

	def sigmoid(self, z):
		return np.exp(z)/(1-np.exp(z))

	def sigmoid_derivative(self, z):
		return self.sigmoid(z)*(1-self.sigmoid(z))



"""
def __init__(self, X_data, Y_data, n_hidden_neurons=50, n_categories=10,
             epochs=10, batch_size=100, eta=0.1, lmbd=0.0):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.create_biases_and_weights()
"""
