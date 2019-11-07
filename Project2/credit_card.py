"""
Program to handle the credit card data for project 2 in FYS-STK4155
"""
import os
import random
import xlsxwriter
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing 	 import OneHotEncoder, Normalizer, normalize
from sklearn.compose 		 import ColumnTransformer
from sklearn.preprocessing   import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.metrics 		 import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.linear_model 	 import LogisticRegression

import plots     as P
import functions as func
# -----------------------------------------------------------------------------
'''
# Trying to set the seed
np.random.seed(0)
random.seed(0)
seed  = 1
'''
# -----------------------------------------------------------------------------
def CreditCard(plot_hist=False):
    # Reading file into data frame
    cwd      = os.getcwd()
    filename = cwd + '/default of credit card clients.xls'
    nanDict  = {}
    df       = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict) # index_col = 1???

    # Renames the target column
    df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

    # Reducing the data
    df = df.drop(df[(df.MARRIAGE  < 1)].index)
    df = df.drop(df[(df.MARRIAGE  > 3)].index)

    df = df.drop(df[(df.EDUCATION > 4)].index)
    df = df.drop(df[(df.EDUCATION < 1)].index)

    '''
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

    if plot_hist:

        #P.Hist_Sex_Marriage_Education(df.SEX, "SEX")
        #P.Hist_Sex_Marriage_Education(df.MARRIAGE, "Marriage")
        #plt.show()
        """
        P.Hist_Sex_Marriage_Education(df.EDUCATION, "Education")
        plt.show()
        P.Histogram(df.AGE, "AGE", "Age [yr]")
        P.Histogram(df.LIMIT_BAL, "LIMIT_BAL", "Amount of given credit in NT dollars")
        """
        list_BILL_AMT = [df.BILL_AMT1, df.BILL_AMT2, df.BILL_AMT3, df.BILL_AMT4, df.BILL_AMT5, df.BILL_AMT6]
        list_PAY_AMT  = [df.PAY_AMT1, df.PAY_AMT2, df.PAY_AMT3, df.PAY_AMT4, df.PAY_AMT5, df.PAY_AMT6]
        list_PAY      = [df.PAY_0, df.PAY_2, df.PAY_3, df.PAY_4, df.PAY_5, df.PAY_6]
        list_PAY_name = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

        P.Multi_hist(list_=list_BILL_AMT, name="", label="Bill statement (NT dollar)",  title_="BILL_AMT")
        #P.Multi_hist(list_=list_PAY_AMT, name="", label="Previous payment (NT dollar)", title_="PAY_AMT")
        #P.Multi_hist(list_=list_PAY, name=list_PAY_name, label="Repayment status PAY_", title_=list_PAY_name, diff=True)

    # Features and targets
    features = df.loc[:, (df.columns != 'defaultPaymentNextMonth') & (df.columns != "ID")].values # Features # & (df.columns != "ID")
    target   = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values # Targets

    return features, target

def DesignMatrix(features, target):

    # One Hot
    onehotencoder = OneHotEncoder(categories="auto")  # sparse=False (from matrix to array)

    encoded_columns = ColumnTransformer([('onehotencoder', onehotencoder, [1,2,3])]).fit_transform(features) #onehotencoder.fit_transform(X[1:4])

    column_0 = features[:,0][:,np.newaxis]
    column_4_22 = features[:,4:]

    col_scale = np.concatenate((column_0, column_4_22), axis=1)

    scaled_columns  = RobustScaler().fit_transform(col_scale)

    X = np.concatenate([encoded_columns, scaled_columns], axis=1)
    y = target

    return X, y


def Make_histograms(X):

    list_BILL_AMT = [X[:,17], X[:,18], X[:,19], X[:,20], X[:,21], X[:,22]]
    list_PAY_AMT  = [X[:,23], X[:,24], X[:,25], X[:,26], X[:,27], X[:,28]]
    list_PAY      = [X[:,11], X[:,12], X[:,13], X[:,14], X[:,15], X[:,16]]
    list_PAY_name = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    P.Multi_hist(list_=list_BILL_AMT, name="", label="Bill statement (NT dollar)",  title_="BILL_AMT")
    P.Multi_hist(list_=list_PAY_AMT, name="", label="Previous payment (NT dollar)", title_="PAY_AMT")
    P.Multi_hist(list_=list_PAY, name=list_PAY_name, label="Repayment status PAY_", title_=list_PAY_name, diff=True)
