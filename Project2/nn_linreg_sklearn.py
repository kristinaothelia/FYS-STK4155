"""
Linear regression on Frankes's function, using sklearn NN
"""
import sys
import numpy 			 	 	as np
import seaborn 			 	 	as sns
import matplotlib.pyplot 	 	as plt

from sklearn.neural_network  	import MLPRegressor
from sklearn.metrics 		 	import mean_squared_error, classification_report, confusion_matrix
from sklearn.model_selection 	import train_test_split
from random 				 	import random, seed
from mpl_toolkits.mplot3d 	 	import Axes3D
from matplotlib 			 	import cm
from matplotlib.ticker 		 	import LinearLocator, FormatStrFormatter

import plots     				as P
import functions 				as func
from   neural_network_lin_TEST 	import NN

np.random.seed(5)

n = 5  	 # Project 1
N = 200  # Project 1
k = 20   # Project 1

x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
X = func.create_X(x, y, n)

XX, YY = np.meshgrid(x,y)
ZZ     = func.FrankeFunction(XX, YY)
z      = func.Create_data(XX, YY, ZZ, noise=True)
X      = func.create_X(XX, YY, n)

X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=1.0/k) # Ikke splitte?

epochs     = 500 #1000
batch_size = 50  #60 #500

eta_vals   = np.logspace(-7, -3, 5)
lmbd_vals  = np.logspace(-7, -3, 5)

MSE        = np.zeros((len(eta_vals), len(lmbd_vals)))
R2         = np.zeros((len(eta_vals), len(lmbd_vals)))
sns.set()

def not_now():
	for i, eta in enumerate(eta_vals):
		for j, lmbd in enumerate(lmbd_vals):

			reg = MLPRegressor(	activation="relu", # Eller en annen?
			    				solver="sgd",
								alpha=lmbd,
			    				learning_rate_init=eta,
			    				max_iter=epochs,
			    				tol=1e-5 )

			reg.fit(X_train, y_train)
			y_pred    = reg.predict(X_test)  # data
			MSE[i][j] = mean_squared_error(y_test, y_pred)
			R2[i][j]  = reg.score(X_test, y_test) # reg.score(X_test, y_pred)

			#y_pred = np.reshape(y_pred, (N, N))

			print("Learning rate = ", eta)
			print("Lambda =        ", lmbd)
			print("MSE score:      ", mean_squared_error(y_test, y_pred) )
			print("R2 score:       ", reg.score(X_test, y_test) ) #reg.score(X_test, y_pred))
			#print("R2 score:       ", reg.score(X_test, y_pred))  # Blir alltid 1.0
			print()

	etas = ["{:0.2e}".format(i) for i in eta_vals]

	fig, ax = plt.subplots()
	sns.heatmap(MSE, annot=True, xticklabels=lmbd_vals, yticklabels=etas, ax=ax, linewidths=.3, linecolor="black")
	ax.set_title("MSE scores (sklearn)")
	ax.set_ylabel("$\eta$")
	ax.set_xlabel("$\lambda$")
	plt.show()

	fig, ax = plt.subplots()
	sns.heatmap(R2, annot=True, xticklabels=lmbd_vals, yticklabels=etas, ax=ax, linewidths=.3, linecolor="black")
	ax.set_title("Accuracy/R2 scores (sklearn)")
	ax.set_ylabel("$\eta$")
	ax.set_xlabel("$\lambda$")
	plt.show()

#not_now()


# Plot real function and model
#--------------------------------------------------------------------------
lamb  = 0.0001
alpha = lamb
#eta   = 0.001  # Default
eta   = 0.01

reg = MLPRegressor(	activation="relu", # Eller en annen?
    				solver="sgd",
    				learning_rate='constant',
    				alpha=alpha,
					learning_rate_init=eta,
    				max_iter=1000,
    				tol=1e-5 ) # hidden_layer_sizes=(100,20) Bruke 1?

reg  = reg.fit(X_train, y_train)
pred = reg.predict(X_test)
print("MSE score on test set: ", mean_squared_error(y_test, pred))
print("R2  score on test set: ", reg.score(X_test, y_test))  # Accuracy? Feil? Hva skal sendes inn?
#MSE score on test set:  0.9901514153804305
#R2  score on test set:  0.06982857708324741

pred = reg.predict(X).reshape(N,N)

fig  = plt.figure(); fig2 = plt.figure()
#cmap = cm.PRGn; my_cmap_r = cm.get_cmap('PRGn_r')
ax   = fig.gca(projection='3d'); ax2  = fig2.gca(projection='3d')

ax.set_title("Franke's function", fontsize=16)
ax2.set_title("Model (sklearn)", fontsize=16)

# Plot the surface.
surf = ax.plot_surface(XX, YY, ZZ, cmap=cm.inferno, #my_cmap_r
                       linewidth=0, antialiased=False)

surf2 = ax2.plot_surface(XX, YY, pred, cmap=cm.inferno,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.view_init(azim=61, elev=15);  ax2.view_init(azim=61, elev=15)
ax.set_xlabel('x',  fontsize=15); ax.set_ylabel('y',  fontsize=15); ax.set_zlabel('z',  fontsize=15)
ax2.set_xlabel('x', fontsize=15); ax2.set_ylabel('y', fontsize=15); ax2.set_zlabel('z', fontsize=15)

# Add a color bar which maps values to colors.
fig.colorbar(surf,   shrink=0.5, aspect=5)
fig2.colorbar(surf2, shrink=0.5, aspect=5)
plt.show()
