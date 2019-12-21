"""
Main program for FYS-STK4155 - Project 3: Exoplanet classification
"""
import sys, os
import argparse
import numpy                 as np

from random 				 import random, seed
from sklearn.model_selection import train_test_split

# Import python programs
import NeuralNetwork         as NN
import RandomForest          as RF
import Logistic_reg          as LR
import xgboost_test          as XG
#------------------------------------------------------------------------------

seed 	  = 0
Training  = 0.7
Threshold = 0.9

X = np.load('features.npy', allow_pickle=True)
y = np.load('targets.npy',  allow_pickle=True)

candidates   = np.load('candidates.npy',    allow_pickle=True)
header_names = np.load('feature_names.npy', allow_pickle=True)
GoldiLock    = np.load('GoldiLock.npy',     allow_pickle=True)
feature_list = header_names[1:]

y = y.astype('int')
y = np.ravel(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=Training, test_size = 1-Training, random_state=seed)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Exoplanet classification")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-1', '--LOG', action="store_true", help="Logistic classification")
    group.add_argument('-2', '--NN',  action="store_true", help="Neural network classification")
    group.add_argument('-3', '--RF',  action="store_true", help="Random Forest classification")
    group.add_argument('-4', '--XG',  action="store_true", help="XGBoost classification")

    # Optional argument for habitable zone calculations
    parser.add_argument('-X', '--hab', action='store_true', help="Habitable zone calculations", required=False)

    if len(sys.argv) <= 1:
        sys.argv.append('--help')

    args       = parser.parse_args()
    LOG_method = args.LOG
    NN_method  = args.NN
    RF_method  = args.RF
    XG_method  = args.XG
    hab_zone   = args.hab

    if LOG_method == True:

        print("Logistic regression classification on NASA's KOI data")
        print("--"*55)

        if hab_zone == True:
            LR.LogReg(X_train, X_test,
                      y_train, y_test,
                      candidates, GoldiLock, seed,
                      threshold = Threshold,
                      Goldilock_zone=True,
                      plot_confuse_matrix=True)

        else:
            LR.LogReg(X_train, X_test,
                      y_train, y_test,
                      candidates, GoldiLock, seed,
                      threshold = Threshold,
                      Goldilock_zone=False,
                      plot_confuse_matrix=True)

    elif NN_method == True:

        print("Neural Network classification on NASA's KOI data")
        print("--"*55)

        if hab_zone == True:
            NN.NeuralNetwork(X_train, X_test,
                             y_train, y_test,
                             candidates, GoldiLock, seed,
                             threshold = Threshold,
                             Goldilock_zone=True,
                             plot_confuse_matrix=True)

        else:
            NN.NeuralNetwork(X_train, X_test,
                             y_train, y_test,
                             candidates, GoldiLock, seed,
                             threshold = Threshold,
                             Goldilock_zone=False,
                             plot_confuse_matrix=True)


    elif RF_method == True:

        print("Random Forest classification on NASA's KOI data")
        print("--"*55)

        if hab_zone == True:
            RF.Random_Forest(X_train, X_test,
                             y_train, y_test,
                             candidates, GoldiLock,
                             feature_list, header_names, seed,
                             threshold = Threshold,
                             plot_confuse_matrix=True,
                             plot_feauture_importance=False,
                             Goldilock_zone=True)

        else:
            RF.Random_Forest(X_train, X_test,
                             y_train, y_test,
                             candidates, GoldiLock,
                             feature_list, header_names, seed,
                             threshold = Threshold,
                             plot_confuse_matrix=True,
                             plot_feauture_importance=False,
                             Goldilock_zone=False)

    elif XG_method == True:

        print("XGBoost classification on NASA's KOI data")
        print("--"*55)

        if hab_zone == True:
            XG.XG_Boost(X_train, X_test,
                        y_train, y_test,
                        candidates, GoldiLock,
                        feature_list, header_names, seed,
                        Goldilock_zone=True,
                        plot_confuse_matrix=True)

        else:
            XG.XG_Boost(X_train, X_test,
                        y_train, y_test,
                        candidates, GoldiLock,
                        feature_list, header_names, seed,
                        Goldilock_zone=False,
                        plot_confuse_matrix=True)
