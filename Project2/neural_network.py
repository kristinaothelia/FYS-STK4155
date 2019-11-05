"""
Neural Network code for project 2 in FYS-STK4155
"""
import numpy as np

from sklearn.model_selection import train_test_split
# -----------------------------------------------------------------------------
#np.random.seed(0)
#seed  = 1
# -----------------------------------------------------------------------------

# Inputs : X, y, eta, lamb, minibatch_size, epochs, n_boots, nodes
# hidden_layers = len(nodes)..?
# n_boots for bootstrap

#def weights_biases():
#    pass


# One hidden layer. Only a few nodes. Compare with sklearn

def create_biases_and_weights(n_features, n_hidden_neurons, n_categories):

    '''
    weights = []
    biases  = []

    for n in range(hidden_layers):
        if n == 0:
            input_to_node = n_features
        else:
            input_to_node = w.shape[1]

        # w = np.random.randn(input_to_node,self.nodes[n]) * np.sqrt(1./input_to_node)
        w = (2/np.sqrt(input_to_node)) * np.random.random_sample((input_to_node, nodes[n])) - (1/np.sqrt(input_to_node))
        # w = np.random.randn(input_to_node,self.nodes[n]) * np.sqrt(2/(input_to_node+self.nodes[n]))
        weights.append(np.array(w))

        b = np.zeros(nodes[n]) #+ 0.01
        biases.append(b[:,np.newaxis])
    '''

    # define output weights and biases
    w_out = np.random.rand(w.shape[1],self.y.shape[1])
    self.weights.append(np.array(w_out))

    b_out = np.zeros(w_out.shape[1]) #+ 0.01
    self.biases.append(b_out[:,np.newaxis])


    hidden_weights = np.random.randn(n_features, n_hidden_neurons)
    hidden_bias = np.zeros(n_hidden_neurons) + 0.01

    output_weights = np.random.randn(n_hidden_neurons, n_categories)
    output_bias = np.zeros(n_categories) + 0.01

	return hidden_weights, hidden_bias, output_weights, output_bias

def feed_forward_train(X):
	"""
	From lecture PP on neural networks, from Morten
    X : features
	"""
	hidden_weights, hidden_bias, output_weights, output_bias = (n_features, n_hidden_neurons, n_categories)
	#Make  z_h and a_h lists??

	z_h = np.matmul(X, hidden_weights) + hidden_bias	# Weighted sum of inputs to the hidden layer
	a_h = logistic_function(z_h)						# Activation in the hidden layer

	# Weighted sum of inputs to the output layer
	z_o = np.matmul(a_h, output_weights) + output_bias

	# Softmax output ??
	# Axis 0 holds each input and axis 1 the probabilities of each category
	exp_term = np.exp(z_o)
	probabilities = probabilities(exp_term)

	# For backpropagation need activations in hidden and output layers
	return a_h, probabilities

def feed_forward_out(X):

    hidden_weights, hidden_bias, output_weights, output_bias = (n_features, n_hidden_neurons, n_categories)

    # feed-forward for output
    z_h = np.matmul(X, hidden_weights) + hidden_bias
    a_h = sigmoid(z_h)

    z_o = np.matmul(a_h, output_weights) + output_bias

    exp_term = np.exp(z_o)
    probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
    return probabilities

def back_propagation(lamb):
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

def predict(X):
        probabilities = feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

def predict_probabilities(X):
    probabilities = feed_forward_out(X)
    return probabilities

def train(n_inputs, epochs, iterations, batch_size):
    data_indices = np.arange(n_inputs)

    for i in range(epochs):
        for j in range(iterations):
            # pick datapoints with replacement
            chosen_datapoints = np.random.choice(data_indices, size=batch_size, replace=False)

            # minibatch training data
            X_data = X_data_full[chosen_datapoints]
            Y_data = Y_data_full[chosen_datapoints]

			# Tror dette blir helt feil...
            probabilities = feed_forward()
            output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient = backpropagation()

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
