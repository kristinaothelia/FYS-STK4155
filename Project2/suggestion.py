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
        learning_rate        | backpropagation learning rate
        regularization       | weight gradient regularization strength
        n_epochs             | number of training cycles
        n_batches            | number of batches per training cycle
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
                chosen_datapoints = np.random.choice(data_indices, size=self.batch_size, replace=False)

                idx = chosen_datapoints
                X_batch, Y_batch = X[:,idx], Y[:,idx]

                # learning algorithm
                Y_predict = self.predict(X_batch)
                self.backpropagation(Y_batch, Y_predict, X_batch)
                if np.any(np.isnan(self.hidden_weights)):
                    sys.exit(1)
                else:
                    pass#print(self.hidden_weights)

    def predict(self, X):
        # hidden layer matrix multiplication
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


class MLPSimpleClassifier(MLP):
    """
    MLP Classifier for simple (single variable) respones with n categories
    """
    def __init__(self, n_features, n_categories, n_nodes,
                 hidden_activation,
                 learning_rate, regularization,
                 n_epochs, n_batches ):
        """
        See MLP
        """
        # call MLP __init__ using output activation = sigmoid = expit
        # and n_responses = n_categories to circumvent OneHotEncoding.
        super().__init__(n_features, n_categories, n_nodes,
                         hidden_activation, expit,
                         learning_rate, regularization,
                         n_epochs, n_batches)

    # need to adjust for one-hot-encoding
    def train(self, X, Y, encoded=False):
        Y_OHE = OneHotEncoding(Y) if not encoded else Y
        super().train(X, Y_OHE)


    def classify(self, X):
        Y_prob = self.predict(X)
        return np.argmax(Y_prob, axis=1)





if __name__ == "__main__":
    np.random.seed(0)

    def FrankeFunction(x,y):
	    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	    return term1 + term2 + term3 + term4

    def create_X(x, y, n ):
	    if len(x.shape) > 1:
		    x = np.ravel(x)
		    y = np.ravel(y)
	    N = len(x)
	    l = int((n+1)*(n+2)/2)		# Number of elements in beta
	    X = np.ones((N,l))
	    for i in range(1,n+1):
		    q = int((i)*(i+1)/2)
		    for k in range(i+1):
			    X[:,q+k] = (x**(i-k))*(y**k)
	    return X

    def relu(x):
        """
        the relu function
        f(x) = max(0, x)
        """
        return abs(x)*(x > 0)

    def idf(x):
        """
        the identity function
        f(x) = x
        """
        return x

    def d_relu(x):
        return 1.*(x > 0)


    # Making meshgrid of datapoints and compute Franke's function
    n = 4
    N = 100
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))
    X = create_X(x, y, n=n)

    XX, YY = np.meshgrid(x,y)
    ZZ = FrankeFunction(XX, YY)
    z=np.ravel(ZZ)
    #print(z.shape)

    X = create_X(XX, YY, n=n)

    """
    train_size = 0.75
    test_size = 1. - train_size
    X_train, X_test, y_train, y_test = train_test_split(X, z, train_size=train_size, test_size=test_size)
    X_train, X_test = X_train.T, X_test.T
    """
    X_train = X.T
    y_train = z.T
    
    
    epochs     = 100
    batch_size = 500

    eta_vals  = np.logspace(-6, -3, 4)
    lmbd_vals = np.logspace(-6, -3, 4)
    MSE_array 	 = np.zeros((len(eta_vals), len(lmbd_vals)))  #keeping the MSE value for heatmap

    n_features = len(X_train)  #nr of predictors
    n_responses = 1 #len(y_train)
    n_nodes = 50  #why this number?
    hidden_activation = relu
    response_activation = idf
    hidden_activation_df = d_relu

    
    accuracy_array = np.zeros((len(eta_vals),len(lmbd_vals)))
    
    nn_reg = MLP(n_features=n_features, n_responses=n_responses, n_nodes=n_nodes,
                 hidden_activation=hidden_activation, response_activation=response_activation,
                 hidden_activation_df=hidden_activation_df)
    
    nn_reg.create_biases_and_weights(weight_spread = 0.1, bias_constant = 0.1)
    
    def not_now():
        for i,eta in enumerate(eta_vals):
            for j,lmbd in enumerate(lmbd_vals):
            
                print("step: ", i, j)
                print("eta  = ", eta)
                print("lambda = ", lmbd)
                
                nn_reg.train(X_train, y_train, learning_rate=eta, regularization=lmbd,
                             n_epochs=epochs, batch_size=batch_size)
                y_pred = nn_reg.predict(X_train).flatten()

                            
                #dnn = NN(X_train, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                        #cost_f = 'mse', n_hidden_neurons=n_hidden_neurons, n_categories=n_categories)
                #dnn.train()
    
                #print(X_test_sc)
                #y_pred = dnn.predict(X_test)
                #print(test_predict)
                #print(np.sum(y_pred))

                #print(test_predict)
                #accuracy_array[i][j] = accuracy_score(y_train, test_predict)
                accuracy_array[i,j] = mean_squared_error(y_train, y_pred)

                print("MSE score on test set: ", accuracy_array[i,j])
                print()
                #break
                
    #not_now()
    
    np.save('acc_score', accuracy_array)
    np.save('eta_values', eta_vals)
    np.save('lambda_values', lmbd_vals)
    
    #P.map()
    #print(accuracy_array)
    i_min, j_min = 0,0
    
    nn_reg.train(X_train, y_train, learning_rate=eta_vals[-1], regularization=lmbd_vals[-1],
                 n_epochs=epochs, batch_size=batch_size)
    y_pred2 = nn_reg.predict(X_train).flatten()
    #y_pred2 = dnn.predict(X).reshape(XX.shape)

    print("MSE sco  re on test set: ", mean_squared_error(y_train, np.ravel(y_pred2)))


    fig = plt.figure()
    fig2 = plt.figure()
    ax = fig.gca(projection='3d')
    ax2 = fig2.gca(projection='3d')

    #HAK = int(len(y_pred)/2)
    #y_pred = y_pred.reshape(70, 70)

    # Plot the surface.
    surf = ax.plot_surface(XX, YY, ZZ, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)


    y_pred2 = y_pred2.reshape(XX.shape)
    #print(y_pred2)
    surf2 = ax2.plot_surface(XX, YY, y_pred2, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig2.colorbar(surf2, shrink=0.5, aspect=5)


    plt.figure()
    plt.imshow(ZZ - y_pred2)
    plt.colorbar(cmap='coolwarm')
    plt.show()



