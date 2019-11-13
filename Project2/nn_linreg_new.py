import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plots        as P

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

class MLP:
    def __init__(self, n_features, n_responses, n_nodes,
                 hidden_activation, response_activation,
                 hidden_activation_df):
        """
        n_features           | number of features
        n_responses          | number of response variables
        n_nodes              | number of nodes in hidden layer
        hidden_activation    | activation function for hidden layer
        response_activation  | activation function for ouput layer
        """
        # store attributes
        # important: store activation functions as attributes
        # setup weights and biases
        self.n_features = n_features
        self.n_responses = n_responses
        self.n_nodes = n_nodes
        self.hidden_activation = hidden_activation
        self.response_activation = response_activation
        self.hidden_activation_df = hidden_activation_df

    def create_biases_and_weights(self, weight_spread=0.1, bias_constant=0.1):
        """
        create inital biaes and weights
        Gaussian distribution, mean=?, std=?
        """
        self.hidden_weights = weight_spread*np.random.randn(self.n_features, self.n_nodes).T
        self.hidden_bias = np.zeros((self.n_nodes,1)) + bias_constant
        self.output_weights = weight_spread*np.random.randn(self.n_nodes, self.n_responses).T
        self.output_bias = np.zeros((self.n_responses,1)) + bias_constant

    def train(self, X, Y, learning_rate, regularization, n_epochs, batch_size):

        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        if Y.shape[0] != self.n_responses:
            Y = Y.T if len(Y.shape) == 2 else Y[:,None].T

        n_inputs = X.shape[1]
        data_indices = np.arange(n_inputs)
        n_batches = int((X.shape[1]/self.batch_size))
        #print(self.batch_size)
        #print(n_batches)
        for i in range(self.n_epochs):
            for j in range(n_batches):
                # prepare minibatch
                print(i/self.n_epochs)
                chosen_datapoints = np.random.choice(data_indices, size=self.batch_size, replace=False)

                idx = chosen_datapoints
                X_batch, Y_batch = X[:,idx], Y[:,idx]
                #print(X_batch.shape)
                # learning algorithm
                Y_predict = self.predict(X_batch)
                self.backpropagation(Y_batch, Y_predict, X_batch)
                if np.any(np.isnan(self.hidden_weights)):
                    sys.exit(1)
                else:
                    pass#print(self.hidden_weights)

    def predict(self, X):
        # hidden layer matrix multiplication
        #print(X.shape)
        #print(self.hidden_weights.shape, X.shape)
      
        self.z_h = np.matmul(self.hidden_weights, X) + self.hidden_bias
        
        """if np.any(self.z_h > 10):
            sys.exit(1)
        else:
            print(self.z_h)"""
        # hidden layer activation ( e.g. A_hidden = self.hidden_activation(z_h) )
        self.a_h = self.hidden_activation(self.z_h)

        # output layer matrix multiplication
        z_o = np.matmul(self.output_weights, self.a_h) + self.output_bias

        # output layer activation
        #a_o = self.response_activation(z_o)
        #prediction = a_o
        return z_o#prediction

    def backpropagation(self, Y_true, Y_predict, X):
        # backpropagate prediction error
        error_output = Y_true - Y_predict
        error_hidden = np.matmul(self.output_weights.T, error_output) * self.hidden_activation_df(self.z_h)

        # compute weight gradients
        output_weights_gradient = np.matmul(error_output, self.a_h.T)
        output_bias_gradient = np.sum(error_output, axis=1)[:,None]

        hidden_weights_gradient = np.matmul(error_hidden, X.T)
        hidden_bias_gradient = np.sum(error_hidden, axis=1)[:,None]

        if self.learning_rate > 0.0:
            output_weights_gradient -= self.regularization * self.output_weights
            hidden_weights_gradient -= self.regularization * self.hidden_weights
        self.output_weights += self.learning_rate * output_weights_gradient
        self.output_bias    += self.learning_rate * output_bias_gradient
        self.hidden_weights += self.learning_rate * hidden_weights_gradient
        self.hidden_bias    += self.learning_rate * hidden_bias_gradient
