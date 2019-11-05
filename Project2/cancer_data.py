#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets 		 import load_breast_cancer
from sklearn.model_selection import KFold, cross_val_score, validation_curve
from sklearn 				 import linear_model

df = pd.read_csv('breast-cancer-wisconsin.data.csv', header=None)
df = df.replace({'B':0, 'M':1})

X = df.iloc[:,2:] # features 
y = df.iloc[:,1]  # targets 

print(X)
print(y)
print(X.shape, y.shape)

X_mean = X.mean()
X_std = X.std()

#X_norm = (X - X_mean)/X_std
#print(X_norm.shape)


logReg = linear_model.LogisticRegression()  # estimator 
kfold = KFold(n_splits=5,random_state=7)
cv_results = cross_val_score(logReg, X, y, cv=kfold)
#print(cv_results.mean()*100, "%")



'''
	# Setting the eta and gamma parameters
	eta = 0.001
	gamma = 0.0001  # learning rate? 
	cancer = load_breast_cancer()
	X = cancer.data
	y = cancer.target

	X_train, X_test, y_train, y_test = func.splitting(X, y, TrainingShare=0.75, seed=0)

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	logReg = LogisticRegression()
	logReg.fit(X_train, y_train)
'''