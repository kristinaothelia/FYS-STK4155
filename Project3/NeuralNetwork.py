"""
FYS-STK4155 - Project 3: Neural Network Classification
"""
import sys
import numpy 			 	 	as np
import seaborn 			 	 	as sns
import matplotlib.pyplot 	 	as plt
import scikitplot       		as skplt

from sklearn.preprocessing 		import StandardScaler
from sklearn.neural_network  	import MLPClassifier
from sklearn.metrics 		 	import classification_report, f1_score,		\
									   precision_score, recall_score,       \
                                       accuracy_score, mean_squared_error,  \
                                       mean_absolute_error, confusion_matrix
from sklearn.model_selection 	import train_test_split
from random 				 	import random, seed
from mpl_toolkits.mplot3d 	 	import Axes3D
from matplotlib 			 	import cm
from matplotlib.ticker 		 	import LinearLocator, FormatStrFormatter

import functions 				as func
#------------------------------------------------------------------------------
TrainingShare = 0.70
seed 		  = 0

X = np.load('features.npy', allow_pickle=True)
y = np.load('targets.npy',  allow_pickle=True)

candidates   = np.load('candidates.npy',    allow_pickle=True)
header_names = np.load('feature_names.npy', allow_pickle=True)
GoldiLock    = np.load('GoldiLock.npy', allow_pickle=True)
feature_list = header_names[1:]

y = y.astype('int')
y = np.ravel(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TrainingShare, test_size = 1-TrainingShare, random_state=seed)

#epochs     		 = 500 # 1000
#batch_size 		 = 100
#n_hidden_neurons = 50
#n_categories 	 = 2

def to_categorical_numpy(integer_vector):

	n_inputs 	  = len(integer_vector)
	n_categories  = np.max(integer_vector) + 1
	onehot_vector = np.zeros((n_inputs, n_categories))
	onehot_vector[range(n_inputs), integer_vector] = 1

	return onehot_vector

Y_train_onehot, Y_test_onehot = to_categorical_numpy(y_train), to_categorical_numpy(y_test)

# Make heatmap of the accuracy score with eta_vals and lmbd_vals? See P2
#eta_vals         = np.logspace(-6, -1, 6)
#lmbd_vals        = np.logspace(-6, -1, 6)
# Call heatmap function
# Use best values from heatmap
#eps_final = 1e-4
#alp_final = 1e-2

mlp = MLPClassifier(solver 				= 'lbfgs',
					activation			= 'logistic',
					hidden_layer_sizes  = (200,150,100),
					max_iter			= 1500)
mlp.fit(X_train, y_train)

# Calculating different metrics
predict     = mlp.predict(X_test)
accuracy 	= accuracy_score(y_test, predict)
precision   = precision_score(y_test, predict, average="macro")
recall      = recall_score(y_test, predict, average="macro")
F1_score    = f1_score(y_test, predict, average="macro")

# Calculate the absolute errors
errors = abs(predict - y_test)

# Printing the different metrics:
func.Print_parameters(accuracy, F1_score, precision, recall, errors, name='Neural Network classification')

print(confusion_matrix(y_test, predict))

predict_candidates       = np.array(mlp.predict(candidates))

predicted_false_positive = (predict_candidates == 0).sum()
predicted_exoplanets     = (predict_candidates == 1).sum()

# Information print to terminal
print('\nThe Neural Network Classifier predicted')
print('--------------------------------------')
print('%-5g exoplanets      of %g candidates'  %(predicted_exoplanets, len(predict_candidates)))
print('%-5g false positives of %g candidates'  %(predicted_false_positive, len(predict_candidates)))


"""
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
	ax.set_ylabel("$\\eta$")
	ax.set_xlabel("$\\lambda$")
	plt.show()

	fig, ax = plt.subplots()
	sns.heatmap(R2, annot=True, xticklabels=lmbd_vals, yticklabels=etas, ax=ax, linewidths=.3, linecolor="black")
	ax.set_title("Accuracy/R2 scores (sklearn)")
	ax.set_ylabel("$\\eta$")
	ax.set_xlabel("$\\lambda$")
	plt.show()

def Best_model():

	lamb  = 1e-4
	eta   = 1e-2

	reg = MLPRegressor(	activation="relu", 				# Eller en annen?
	    				solver="sgd",
	    				learning_rate='constant',
	    				alpha=lamb,
						learning_rate_init=eta,
	    				max_iter=1000,
	    				tol=1e-5 )

	reg  = reg.fit(X_train, y_train)
	pred = reg.predict(X_test)
	pred_ = reg.predict(X)#.reshape(N,N)

	print("MSE score: ", mean_squared_error(y, pred_))
	print("R2  score: ", func.R2_ScoreFunction(y, pred_))

Best_model()
"""
