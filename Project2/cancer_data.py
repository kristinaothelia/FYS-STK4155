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