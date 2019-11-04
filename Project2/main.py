"""
Main program for prohect 2 in FYS-STK4155
"""
import os
import random
import xlsxwriter
import pandas            as pd
import numpy 	         as np
import functions         as func
import credit_card       as CD
import matplotlib.pyplot as plt

from sklearn.model_selection     import train_test_split
from sklearn.preprocessing 		 import OneHotEncoder, Normalizer
from sklearn.compose 			 import ColumnTransformer
from sklearn.preprocessing       import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.metrics 			 import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.linear_model 		 import LogisticRegression
from sklearn.linear_model 		 import SGDRegressor, SGDClassifier
# -----------------------------------------------------------------------------
seed = 0
np.random.seed(seed)
# -----------------------------------------------------------------------------

features, target = CD.CreditCard()

X, y = CD.DesignMatrix(features, target)

#sc = StandardScaler()
#robust_scaler = RobustScaler() transform

eta = 0.001
gamma = 0.0001
betas = func.next_beta(X, y, eta, gamma)
#print("betas")
#print(betas)

X_train, X_test, y_train, y_test = func.splitting(X, y, TrainingShare=0.7, seed=0)

eta = 0.001
gamma = 0.0001
betas_train = func.next_beta(X_train, y_train, eta, gamma)

#print(betas_train.shape)
#print(X_test.shape)
ytilde = X_test @ betas_train
model = func.logistic_function(ytilde)


logReg = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000)
#logReg= SGDClassifier(loss='log', random_state=seed, eta0=eta, learning_rate='constant', fit_intercept=False, max_iter=1000)
logReg.fit(X_train, np.ravel(y_train))
ypredict_test_scikit =logReg.predict(X_test)

a_scikit = accuracy_score(y_pred=ypredict_test_scikit, y_true=y_test)
print(a_scikit)


a_ = func.accuracy(model, y_test)
print(a_)

cost = func.cost(y_train, model)   # ytilde or model??
print(cost)
plt.plot(cost)
plot.show()

# Should we use Regressor or Classifier ??

'''
sgdreg = SGDRegressor(random_state=seed, eta0=eta, epsilon=1e-8)
#sgdreg = SGDRegressor(max_iter = 1000, eta0=eta, alpha=gamma, random_state=seed) # average=?
sgdreg.fit(X,np.ravel(y))
#print(sgdreg.intercept_, sgdreg.coef_)
print(sgdreg.coef_)
print('-------')
'''

#sgd_clas = SGDClassifier(loss='log', random_state=seed, eta0=eta, learning_rate='constant', fit_intercept=False, max_iter=1000)
#sgd_clas.fit(X,np.ravel(y))
#print(sgd_clas.coef_)
#print('-------')


#beta_linreg = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
#print(beta_linreg)

'''
# Create an instance of the estimator
#sc = StandardScaler()
logReg = LogisticRegression(solver='lbfgs', max_iter=1000) #n_jobs=-1, random_state=15 # multi_class='multinomial'
# Using the training data to train the estimator
logReg.fit(X, np.ravel(y))
betas_logreg = logReg.coef_
print(betas_logreg)
'''












































#design_matrix = pd.DataFrame(X)
#design_matrix.to_excel(excel_writer = "DesignMatrix.xlsx", header=False, index=False)

#target = pd.DataFrame(X)
#design_matrix.to_excel(excel_writer = "DesignMatrix.xlsx")

#design_matrix_file = pd.read_excel("DesignMatrix.xlsx", header=None, skiprows=None, index_col=None)
#print(design_matrix_file)


'''

# Create an instance of the estimator
logReg = LogisticRegression() #n_jobs=-1, random_state=15

# Using the training data to train the estimator
logReg.fit(X_train, y_train)
betas = logReg.coef_

# Evaluating the model
y_pred_test = logReg.predict(X_test)

#accuracy = accuracy_score(y_pred=y_pred_test, y_true=y_test) # metrics.loc['accuracy', 'LogisticReg']


# Confusion matrix
#CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
#CMatrix(CM)
'''
