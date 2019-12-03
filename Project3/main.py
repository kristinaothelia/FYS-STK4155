"""
Main program for FYS-STK4155 project 3
"""
import sys, os
import numpy        as np

import functions    as P
import RandomForest as RF

if __name__ == '__main__':

    # check for input arguments
    if len(sys.argv) == 1:
	    #print('No arguments passed. Please specify method;')
	    sys.exit()

    arg = sys.argv[1]

    if arg == "LOG":

        print('test LOG')

    elif arg == "NN":

        print('test NN')

    elif arg == "RF":

        print('test RF')

    elif arg == "GX":

        print('test GX')

    else:
	    print('Pass method')
