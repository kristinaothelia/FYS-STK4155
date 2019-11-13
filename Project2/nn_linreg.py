"""
NN to do linear regression on Franke Function
temporary file for development
set up similar to Morten's mplfranke.py for comparison
"""

# Common imports
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from neural_network_lin_TEST import NN
import plots        as P
import sys

np.random.seed(42)

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


# Making meshgrid of datapoints and compute Franke's function
n = 4
N = 10
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
#z = FrankeFunction(x, y)
X = create_X(x, y, n=n)

XX, YY = np.meshgrid(x,y)
ZZ = FrankeFunction(XX, YY)
z=np.ravel(ZZ)

X = create_X(XX, YY, n=n)


#Morten used no train-test split, so default is split = False
split = True
if split:
    train_size = 0.5
    test_size = 1.0 - train_size
    X_train, X_test, y_train, y_test = train_test_split(X, z, train_size=train_size, test_size=test_size)
    #print("X_train.shape = " + str(X_train.shape))
    #print("X_test.shape  = " + str(X_test.shape))
else:
    X_train, y_train = X, z
    #print("X_train.shape = " + str(X_train.shape))
    #print("y_train.shape = " + str(y_train.shape))


epochs     = 200 #60 #30
batch_size = 10 #60 #500

eta_vals = np.logspace(-7, -4, 7)
lmbd_vals = np.logspace(-7, -1, 7)

accuracy_array 	 = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
n_hidden_neurons = 50 #not sure about number???
n_categories 	 = 1


def to_categorical_numpy(integer_vector):

    n_inputs 	  = len(integer_vector)
    n_categories  = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector

#Y_train_onehot, Y_test_onehot = to_categorical_numpy(y_train), to_categorical_numpy(y_test)

#print(X_train.shape)
#print(X_test)

def not_now():
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            dnn = NN(X_train, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    cost_f = 'mse', n_hidden_neurons=n_hidden_neurons, n_categories=n_categories)
            dnn.train()

            #print(X_test_sc)
            y_pred = dnn.predict(X_test)
            #print(test_predict)
            #print(np.sum(y_pred))

            #print(test_predict)
            #accuracy_array[i][j] = accuracy_score(y_train, test_predict)
            accuracy_array[i][j] = mean_squared_error(y_test, y_pred)

            print("Learning rate  = ", eta)
            print("Lambda = ", lmbd)
            print("MSE score on test set: ", mean_squared_error(y_test, y_pred))
            print()
            #break

    np.save('acc_score', accuracy_array)
    np.save('eta_values', eta_vals)
    np.save('lambda_values', lmbd_vals)
not_now()
P.map()

##plt.imshow(accuracy_array)
##plt.show()

#print(np.where(accuracy_array == np.min(accuracy_array)))


dnn = NN(X_train, y_train, eta=1e-6, lmbd=0.1, epochs=epochs, batch_size=batch_size,
         cost_f = 'mse', n_hidden_neurons=n_hidden_neurons, n_categories=n_categories)
dnn.train()
y_pred = dnn.predict(X_test)
#y_pred2 = dnn.predict(X).reshape(XX.shape)

print("MSE sco  re on test set: ", mean_squared_error(y_test, y_pred))

sys.exit()


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

fig = plt.figure()
fig2 = plt.figure()
ax = fig.gca(projection='3d')
ax2 = fig2.gca(projection='3d')

#HAK = int(len(y_pred)/2)
#y_pred = y_pred.reshape(70, 70)

# Plot the surface.
surf = ax.plot_surface(XX, YY, ZZ, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)



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


plt.show()
