from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
import project1_func
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import sys

def Lasso_CV(X, data, alphas):
	X_train, X_test, data_train, data_test = train_test_split(X, data, shuffle=True, test_size=0.2)

	lasso = Lasso(max_iter = 1e3, normalize = True)
	coefs = []


	for a in alphas:
		lasso.set_params(alpha=a)
		lasso.fit(X_train, data_train)
		coefs.append(lasso.coef_)
		

	lassocv = LassoCV(alphas = None, cv = 10, max_iter = 1e4, normalize = True)
	lassocv.fit(X_train, data_train)

	lasso.set_params(alpha=lassocv.alpha_)
	lasso.fit(X_train, data_train)

	#ytilde = lasso.predict(X_train)
	ypredict = lasso.predict(X)
	print(ypredict.shape)
	MSE_test = mean_squared_error(data_test, ypredict)
	print(X_test.shape, X_train.shape, data_train.shape)
	#MSE_train = mean_squared_error(data_train, ytilde)

	return ypredict, lassocv.alpha_, MSE_test

# Load the terrain
terrain_image = imread('Oslo.tif')

p_degree = 0
		
terrain_arr = np.array(terrain_image)

x = np.arange(0, terrain_arr.shape[1])/(terrain_arr.shape[1]-1)
y = np.arange(0, terrain_arr.shape[0])/(terrain_arr.shape[0]-1)

x,y = np.meshgrid(x,y)

X = project1_func.CreateDesignMatrix(x,y, n=p_degree)
terrain_arr = np.ravel(terrain_arr)

alphas = np.logspace(-6, -3, 100)

model, alpha, MSE = Lasso_CV(X, terrain_arr, alphas)