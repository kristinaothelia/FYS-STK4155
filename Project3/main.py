"""
Main program for FYS-STK4155 - Project 3: Exoplanet classification
"""
import sys, os
import numpy                 as np

from random 				 import random, seed
from sklearn.model_selection import train_test_split

# Import python programs
#import NeuralNetwork         as NN
import RandomForest          as RF
import goldilock             as GL
import Logistic_reg          as LR
import xgboost_test          as XG
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


if __name__ == '__main__':

    # check for input arguments
    if len(sys.argv) == 1:
	    print("No arguments passed. Please specify method; 'LOG', 'NN', 'RF', 'XG' or 'Hab'")
	    sys.exit()

    arg = sys.argv[1]

    if arg == "LOG":

        print("Logistic regression classification on NASA's KOI data")
        print("--"*55)

        LR.LogReg(X_train, X_test, y_train, y_test, candidates, seed)

    elif arg == "NN":

        print("Neural Network classification on NASA's KOI data")
        print("--"*55)

    elif arg == "RF":

        print("Random Forest classification on NASA's KOI data")
        print("--"*55)

        RF.Random_Forest(X_train, X_test, y_train, y_test, candidates, feature_list, header_names, seed)

    elif arg == "XG":

        print("XGBoost classification on NASA's KOI data")
        print("--"*55)

        XG.XG_Boost(X_train, X_test, y_train, y_test, candidates, feature_list, header_names, seed)

    elif arg == "Hab":

        # Ha muligheten for aa regne Goldilock for alle metodene?
        GL.GoldilocksZone()

    else:
	    print("Pass method 'LOG', 'NN', 'RF', 'XG' or 'Hab'")
