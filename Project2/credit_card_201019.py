import pandas as pd
import os
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing 	 import OneHotEncoder
from sklearn.compose 		 import ColumnTransformer
from sklearn.preprocessing   import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.metrics 		 import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.linear_model 	 import LogisticRegression


# Trying to set the seed
np.random.seed(0)
random.seed(0)

seed  = 1

# Reading file into data frame
cwd = os.getcwd()
filename = cwd + '/default of credit card clients.xls'
nanDict = {}
df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)

#print(df)

df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

print(df.shape)

df = df.drop(df[(df.MARRIAGE < 1)].index)  
df = df.drop(df[(df.MARRIAGE > 3)].index)

df = df.drop(df[(df.EDUCATION > 4)].index)
df = df.drop(df[(df.EDUCATION < 1)].index)

df = df.drop(df[(df.BILL_AMT1 == 0) &
                (df.BILL_AMT2 == 0) &
                (df.BILL_AMT3 == 0) &
                (df.BILL_AMT4 == 0) &
                (df.BILL_AMT5 == 0) &
                (df.BILL_AMT6 == 0)].index)

df = df.drop(df[(df.PAY_AMT1 == 0) &
                (df.PAY_AMT2 == 0) &
                (df.PAY_AMT3 == 0) &
                (df.PAY_AMT4 == 0) &
                (df.PAY_AMT5 == 0) &
                (df.PAY_AMT6 == 0)].index)

df = df.drop(df[(df.PAY_0 == 0)].index)
df = df.drop(df[(df.PAY_2 == 0)].index)
df = df.drop(df[(df.PAY_3 == 0)].index)
df = df.drop(df[(df.PAY_4 == 0)].index)
df = df.drop(df[(df.PAY_5 == 0)].index)
df = df.drop(df[(df.PAY_6 == 0)].index)

df = df.drop(df[(df.PAY_0 < -1)].index)
df = df.drop(df[(df.PAY_2 < -1)].index)
df = df.drop(df[(df.PAY_3 < -1)].index)
df = df.drop(df[(df.PAY_4 < -1)].index)
df = df.drop(df[(df.PAY_5 < -1)].index)
df = df.drop(df[(df.PAY_6 < -1)].index)

'''
df = df.drop(df[(df.BILL_AMT1 == 0)].index)
df = df.drop(df[(df.BILL_AMT2 == 0)].index)
df = df.drop(df[(df.BILL_AMT3 == 0)].index)
df = df.drop(df[(df.BILL_AMT4 == 0)].index)
df = df.drop(df[(df.BILL_AMT5 == 0)].index)
df = df.drop(df[(df.BILL_AMT6 == 0)].index)

df = df.drop(df[(df.PAY_AMT1 == 0)].index)
df = df.drop(df[(df.PAY_AMT2 == 0)].index)
df = df.drop(df[(df.PAY_AMT3 == 0)].index)
df = df.drop(df[(df.PAY_AMT4 == 0)].index)
df = df.drop(df[(df.PAY_AMT5 == 0)].index)
df = df.drop(df[(df.PAY_AMT6 == 0)].index)
'''

'''
df = df.drop(df[(df.PAY_0 == 0) &
				(df.PAY_2 == 0) &
				(df.PAY_3 == 0) &
				(df.PAY_4 == 0) &
				(df.PAY_5 == 0) &
				(df.PAY_6 == 0)].index)

df = df.drop(df[(df.PAY_0 < -1) &
				(df.PAY_2 < -1) &
				(df.PAY_3 < -1) &
				(df.PAY_4 < -1) &
				(df.PAY_5 < -1) &
				(df.PAY_6 < -1)].index)

df = df.drop(df[(df.BILL_AMT1 == 0) &
                (df.BILL_AMT2 == 0) &
                (df.BILL_AMT3 == 0) &
                (df.BILL_AMT4 == 0) &
                (df.BILL_AMT5 == 0) &
                (df.BILL_AMT6 == 0)].index)

df = df.drop(df[(df.PAY_AMT1 == 0) &
                (df.PAY_AMT2 == 0) &
                (df.PAY_AMT3 == 0) &
                (df.PAY_AMT4 == 0) &
                (df.PAY_AMT5 == 0) &
                (df.PAY_AMT6 == 0)].index)
'''

print(df.shape)
print(np.max(df.PAY_0))
print(np.min(df.PAY_0))

# Features and targets 
X = df.loc[:, (df.columns != 'defaultPaymentNextMonth') & (df.columns != "ID")].values # Features # & (df.columns != "ID")
y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values # Targets 

#print(X)
#print(X.shape)

'''
pay_index = [0, 2, 3, 4, 5, 6]
for i in range(len(pay_index)):
	p = '%s' %(pay_index[i])
	p = int(p)
	if df.PAY_p == 0:
		print('hei')
'''


'''
df = df.drop(df[(df.PAY_0 == 0) &
				(df.PAY_2 == 0) &
				(df.PAY_3 == 0) &
				(df.PAY_4 == 0) &
				(df.PAY_5 == 0) &
				(df.PAY_6 == 0)].index)

print(df.shape)
print(np.max(df.PAY_0))
print(np.min(df.PAY_0))
'''

#print(X)
#print(X.shape)

sc = StandardScaler()
#robust_scaler = RobustScaler()

# Train-test split
trainingShare = 0.5 
X_train, X_test, y_train, y_test=train_test_split(X, y, train_size=trainingShare, test_size = 1-trainingShare, random_state=seed)

'''
print(X.shape)
# Categorical variables to one-hot's
onehotencoder = OneHotEncoder(categories="auto")

X = ColumnTransformer([("", onehotencoder, [3]),],remainder="passthrough").fit_transform(X)
print(X.shape)
'''


'''
# Create an instance of the estimator 
logReg = LogisticRegression() #n_jobs=-1, random_state=15

# Using the training data to train the estimator 
logReg.fit(X_train, y_train)

# Evaluating the model 
y_pred_test = logReg.predict(X_test)

accuracy = accuracy_score(y_pred=y_pred_test, y_true=y_test) # metrics.loc['accuracy', 'LogisticReg'] 
'''

# Confusion matrix
#CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
#CMatrix(CM)
