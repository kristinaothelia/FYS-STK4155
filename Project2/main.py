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
import scikitplot        as skplt

from sklearn.model_selection     import train_test_split
from sklearn.preprocessing 		 import OneHotEncoder, Normalizer
from sklearn.compose 			 import ColumnTransformer
from sklearn.preprocessing       import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.metrics 			 import confusion_matrix, accuracy_score, roc_auc_score, auc, roc_curve
from sklearn.linear_model 		 import LogisticRegression
from sklearn.linear_model 		 import SGDRegressor, SGDClassifier  # better than logistic ??
from sklearn.datasets 		     import load_breast_cancer
# -----------------------------------------------------------------------------
seed = 0
np.random.seed(seed)
# -----------------------------------------------------------------------------

# Setting the eta and gamma parameters
#eta = 0.001
#gamma = 0.0001  # learning rate? 

eta = 0.01
gamma = 0.1  # learning rate? 

eta_range = [0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7]
gamma_range = [0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7]



CreditCard = True

if CreditCard == True:
	features, target = CD.CreditCard()
	X, y = CD.DesignMatrix(features, target)
	# Calculating the beta values
	betas = func.next_beta(X, y, eta, gamma)

	# Splitting X and y in a train and test set
	X_train, X_test, y_train, y_test = func.splitting(X, y, TrainingShare=0.75, seed=0)

else: 
	cancer = load_breast_cancer()
	X = cancer.data
	y = cancer.target

	X_train, X_test, y_train, y_test = func.splitting(X, y, TrainingShare=0.75, seed=0)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	
	#logReg = LogisticRegression()
	#logReg.fit(X_train, y_train)

'-------------------------------------------'
print('The shape of X is:', X.shape)
print('The shape of y is:', y.shape)
'-------------------------------------------'
print('')

# Calculating the beta values based og the training set
betas_train = func.next_beta(X_train, y_train, eta, gamma)
#betas_train = func.steepest(X_train, y_train, gamma)

# Calculating ytilde and the model of logistic regression 
z 		    = X_test @ betas_train   # choosing best beta here? 
model       = func.logistic_function(z)

# Calculating the accuracy with our own function
accuracy_test =  func.accuracy(model, y_test)
exp_term = X_test
Probabilities = func.probabilities(exp_term)   # ??? 

# Creating a logistic regression model with scikit-learn 
# Calculating the corresponding accuracy
logReg = LogisticRegression(random_state=0, solver='sag', max_iter=1000, fit_intercept=False) # solver='lbfgs'
logReg.fit(X_train, np.ravel(y_train))

ypredict_scikit  		     = logReg.predict(X_test)
predict_probabilities_scikit = logReg.predict_proba(X_test)  # Probability estimates
score_scikit				 = logReg.score(X_test, y_test)  # ?? same as accuracy ?? 

accuracy_scikit  = accuracy_score(y_pred=ypredict_scikit, y_true=y_test)

# Comparing our own accuracy with scikit-learn
print('')
'-------------------------------------------'
print('The accuracy with our function is  :', accuracy_test)
print('The accuracy of scikit-learn is    :', accuracy_scikit)
'-------------------------------------------'

fpr, tpr, thresholds = roc_curve(y_test, predict_probabilities_scikit[:,1], pos_label=None)
AUC_scikit 			 = auc(fpr, tpr)

# The AUC scikit
print('')
'-------------------------------------------'
print('The AUC is:', AUC_scikit)
'-------------------------------------------'

#p = model
p = predict_probabilities_scikit[:,0]
#p = func.probabilities(model)
notP = 1 - np.ravel(p)
y_p = np.zeros((len(notP), 2))
y_p[:,0] = np.ravel(p)
y_p[:,1] = np.ravel(notP)

skplt.metrics.plot_cumulative_gain(y_test, y_p)
plt.show()

# Creating a Confusion matrix using pandas and pandas dataframe
CM 			 = func.Create_ConfusionMatrix(model, y_test, plot=False)
CM_DataFrame = func.ConfusionMatrix_DataFrame(CM, labels=['pay', 'default'])

print('')
'-------------------------------------------'
print('The Confusion Matrix')
print('')
print(CM_DataFrame)
'-------------------------------------------'

'''
cost = func.cost(X_test, y_train, betas_train)   # ytilde or model??
print(cost)
plt.plot(cost)
plt.show()
'''


'''
plt.plot(epoch_list, acc_train, label='train')
plt.plot(epoch_list, acc_test, label='test')
plt.legend()
plt.show()
'''




'''
for e in range(len(eta_range)):
	for g in range(len(gamma_range)):

		print('----------')
		print(eta_range[e])
		print(gamma_range[g])
		print('----------')

		# Calculating the beta values based og the training set
		betas_train = func.next_beta(X_train, y_train, eta_range[e], gamma_range[g])
		#betas_train = func.steepest(X_train, y_train, gamma)

		# Calculating ytilde and the model of logistic regression 
		z 		    = X_test @ betas_train   # choosing best beta here? 
		model       = func.logistic_function(z)

		# Calculating the accuracy with our own function
		accuracy_test =  func.accuracy(model, y_test)
		exp_term = X_test
		Probabilities = func.probabilities(exp_term)   # ??? 

		# Creating a logistic regression model with scikit-learn 
		# Calculating the corresponding accuracy
		logReg = LogisticRegression(random_state=0, solver='sag', max_iter=1000, fit_intercept=False) # solver='lbfgs'
		logReg.fit(X_train, np.ravel(y_train))

		ypredict_scikit  		     = logReg.predict(X_test)
		predict_probabilities_scikit = logReg.predict_proba(X_test)  # Probability estimates
		score_scikit				 = logReg.score(X_test, y_test)  # ?? same as accuracy ?? 

		accuracy_scikit  = accuracy_score(y_pred=ypredict_scikit, y_true=y_test)

		# Comparing our own accuracy with scikit-learn
		print('')
		'-------------------------------------------'
		print('The accuracy with our function is  :', accuracy_test)
		print('The accuracy of scikit-learn is    :', accuracy_scikit)
		'-------------------------------------------'

		fpr, tpr, thresholds = roc_curve(y_test, predict_probabilities_scikit[:,1], pos_label=None)
		AUC_scikit 			 = auc(fpr, tpr)

		# The AUC scikit
		print('')
		'-------------------------------------------'
		print('The AUC is:', AUC_scikit)
		'-------------------------------------------'
'''










#logReg= SGDClassifier(loss='log', random_state=seed, eta0=eta, learning_rate='constant', fit_intercept=False, max_iter=1000)

#sc = StandardScaler()
#robust_scaler = RobustScaler() transform


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
