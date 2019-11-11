"""
Linear regression on Frankes's function, using sklearn NN
"""
import sys
import numpy 			 	 	as np
import seaborn 			 	 	as sns
import matplotlib.pyplot 	 	as plt
import scikitplot       		as skplt

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

epochs     = 1000
batch_size = 100

eta_vals   = np.logspace(-5, -1, 5)
lmbd_vals  = np.logspace(-5, -1, 5)

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
			model     = reg.predict(X).reshape(N,N)

			MSE[i][j] = func.MeanSquaredError(ZZ, model)
			R2[i][j]  = func.R2_ScoreFunction(ZZ, model)

			print("Learning rate = ", eta)
			print("Lambda =        ", lmbd)
			print("MSE score:      ", func.MeanSquaredError(ZZ, model))
			print("R2 score:       ", func.R2_ScoreFunction(ZZ, model))
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


# Plot real function and model, with best lamb and eta
#--------------------------------------------------------------------------
lamb  = 1e-4
eta   = 1e-2

reg = MLPRegressor(	activation="relu", # Eller en annen?
    				solver="sgd",
    				learning_rate='constant',
    				alpha=lamb,
					learning_rate_init=eta,
    				max_iter=1000,
    				tol=1e-5 )

reg  = reg.fit(X_train, y_train)
pred = reg.predict(X_test)
pred_ = reg.predict(X).reshape(N,N)

print("MSE score on test set: ", mean_squared_error(ZZ, pred_))
print("R2  score on test set: ", func.R2_ScoreFunction(ZZ, pred_))

fig  = plt.figure(); fig2 = plt.figure()
#cmap = cm.PRGn; my_cmap_r = cm.get_cmap('PRGn_r')
ax   = fig.gca(projection='3d'); ax2  = fig2.gca(projection='3d')

ax.set_title("Franke's function", fontsize=16)
ax2.set_title("Model (sklearn)", fontsize=16)

# Plot the surface.
surf = ax.plot_surface(XX, YY, ZZ, cmap=cm.inferno, #my_cmap_r
                       linewidth=0, antialiased=False)

surf2 = ax2.plot_surface(XX, YY, pred_, cmap=cm.inferno,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.view_init(azim=61, elev=15);  ax2.view_init(azim=61, elev=15)
ax.set_xlabel('x',  fontsize=15); ax.set_ylabel('y',  fontsize=15); ax.set_zlabel('z',  fontsize=15)
ax2.set_xlabel('x', fontsize=15); ax2.set_ylabel('y', fontsize=15); ax2.set_zlabel('z', fontsize=15)

# Add a color bar which maps values to colors.
fig.colorbar(surf,   shrink=0.5, aspect=5)
fig2.colorbar(surf2, shrink=0.5, aspect=5)
plt.show()
