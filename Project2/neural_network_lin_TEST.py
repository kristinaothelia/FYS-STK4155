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
import sys


class NN:
    def __init__(self, X_data, Y_data,
            n_hidden_neurons=50,
            n_categories=10,
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0,
            cost_f = 'sigmoid'):

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
        self.cost_f = cost_f
        print(self.cost_f)

    

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        # feed-forward for training
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = self.f(self.z_h)

        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias
        #self.a_o = self.f(self.z_o)
        self.a_o = self.z_o


    def feed_forward_out(self, X):
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = self.f(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        a_o = z_o  #f(x) = x

        return(a_o)

    def f(self, x):
        return abs(x) * (x > 0)
        #return self.relu(x, False)
        #return 1/(1+exp(-x))
        
        return L

    def df(self, x):
        return 1.*(x > 0)

    def backpropagation(self):
        
        #error_output = 2*(self.a_o.reshape((np.size(self.a_o), 1)) - self.Y_data.reshape((np.size(self.a_o), 1)))
        #error_output = self.df(self.z_o)* -2*(self.a_o.reshape((np.size(self.a_o), 1)) - self.Y_data.reshape((np.size(self.a_o), 1)))
        #error_output = (self.Y_data.reshape((np.size(self.a_o), 1)) - self.a_o.reshape((np.size(self.a_o), 1))  )
        #error_hidden = np.matmul(error_output, self.output_weights.T) * self.df(self.z_o)

        Y_data = self.Y_data.reshape(len(self.Y_data), 1)
        a_o = self.a_o.reshape(len(self.a_o), 1) 
        #print(Y_data.shape)

        error_output = Y_data - a_o
        #print(self.output_weights.T.shape)
        #print(error_output.shape)

        error_hidden = np.matmul(error_output, self.output_weights.T) * self.df(self.z_o)

        #sys.exit()




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
        return(probabilities)
        #return np.argmax(probabilities, axis=1)

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
