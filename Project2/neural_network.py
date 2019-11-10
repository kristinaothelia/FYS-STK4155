"""
Neural Network class for project 2 in FYS-STK4155
"""
import matplotlib.pyplot     as plt
import numpy                 as np
import scikitplot            as skplt

from sklearn.model_selection import train_test_split
from sklearn.datasets        import load_breast_cancer
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler, RobustScaler
from sklearn.metrics         import accuracy_score
from scipy.special           import expit


class NN:
    def __init__(self, X_data, Y_data,
                n_hidden_neurons=50,
                n_categories=10,
                epochs=10,
                batch_size=100,
                eta=0.1,
                lmbd=0.0):

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

    def sigmoid(self, x):
        #A = 1./(1. + np.exp(-x))
        A = expit(x)
        #print(A)
        return A

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        # feed-forward for training
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = self.sigmoid(self.z_h)

        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias

        exp_term = np.exp(self.z_o)
        #print(exp_term)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        #print("probb:", self.probabilities)

    def feed_forward_out(self, X):
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = self.sigmoid(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias

        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities

    def backpropagation(self):
        #self.Y_data = self.Y_data[:,np.newaxis] #???
        #print('prob', self.probabilities.shape)
        #print('a_h', self.a_h.shape)
        #print('y_data', self.Y_data.shape)
        error_output = self.probabilities - self.Y_data
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)

        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False)

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()


'''
 Slette?

def safe():

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
'''
