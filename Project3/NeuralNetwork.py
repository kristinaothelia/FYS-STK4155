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

import functions 				as F
#------------------------------------------------------------------------------
X = np.load('features.npy', allow_pickle=True)
y = np.load('targets.npy',  allow_pickle=True)

candidates   = np.load('candidates.npy',    allow_pickle=True)
header_names = np.load('feature_names.npy', allow_pickle=True)
feature_list = header_names[1:]

print(X); print(y)

y = y.astype('int')
y = np.ravel(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TrainingShare, test_size = 1-TrainingShare, random_state=seed)


epochs     = 500 # 1000
batch_size = 100

# From project 2:

def Heatmap_MSE_R2():

	eta_vals   = np.logspace(-5, -1, 5)
	lmbd_vals  = np.logspace(-5, -1, 5)

	MSE        = np.zeros((len(eta_vals), len(lmbd_vals)))
	R2         = np.zeros((len(eta_vals), len(lmbd_vals)))
	sns.set()

	for i, eta in enumerate(eta_vals):
		for j, lmbd in enumerate(lmbd_vals):

			### ENDRE!!!
			reg = MLPRegressor(	activation="relu", # Eller en annen?
			    				solver="sgd",
								alpha=lmbd,
			    				learning_rate_init=eta,
			    				max_iter=epochs,
			    				tol=1e-5 )

			reg.fit(X_train, y_train)
			y_pred    = reg.predict(X_test)
			model     = reg.predict(X).reshape(N,N)

			MSE[i][j] = func.MeanSquaredError(data, model)
			R2[i][j]  = func.R2_ScoreFunction(data, model)

			print("Learning rate = ", eta)
			print("Lambda =        ", lmbd)
			print("MSE score:      ", F.MeanSquaredError(data, model))
			print("R2 score:       ", F.R2_ScoreFunction(data, model))
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

def Best_model():

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

	print("MSE score: ", mean_squared_error(data, pred_))
	print("R2  score: ", func.R2_ScoreFunction(data, pred_))
