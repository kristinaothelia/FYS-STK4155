"""
FYS-STK4155 - Project 3: XGBoost
"""
import numpy     as np
import xgboost   as xgb
import functions as func

from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics         import precision_score, recall_score,          \
                                    accuracy_score, confusion_matrix,       \
                                    mean_absolute_error, f1_score
#------------------------------------------------------------------------------
def XG_Boost(X_train, X_test, y_train, y_test, candidates, feature_list, header_names, seed):

    D_train    = xgb.DMatrix(X_train, label=y_train)
    D_test     = xgb.DMatrix(X_test, label=y_test)

    param      = {  "eta": 0.3,
                    "max_depth": 3,
                    "objective": "multi:softprob",
                    "num_class": 2  }
    steps      = 20  #the number of training iterations
    model      = xgb.train(param, D_train, steps)
    y_pred     = model.predict(D_test)
    best_preds = np.asarray([np.argmax(line) for line in y_pred])

    #model.dump_model('test.txt')

    # Calculating different metrics
    accuracy 	= accuracy_score(y_test, best_preds)
    precision   = precision_score(y_test, best_preds, average="macro")
    recall      = recall_score(y_test, best_preds, average="macro")
    F1_score    = f1_score(y_test, best_preds, average="macro") # SE PAA DENNE!!! Feil input..
    # Calculate the absolute errors
    errors      = abs(best_preds - y_test) # SE PAA DENNE!!! Blir veldig hoy verdi. Feil input..

    # Printing the different metrics
    func.Print_parameters(accuracy, F1_score, precision, recall, errors, name='XGBoost')

    # Confusion matrix?


    # Hmmm... resultatet blir veldig rart... 90% eksoplaneter

    D_test     = xgb.DMatrix(candidates, label=candidates)
    y_pred     = model.predict(D_test)
    pred_cand  = np.asarray([np.argmax(line) for line in y_pred])

    # Divide into predicted false positives and confirmed exoplanets
    pred_FP    = (pred_cand == 0).sum() 	# Predicted false positives
    pred_Conf  = (pred_cand == 1).sum() 	# Predicted exoplanets/confirmed

    # Information print to terminal
    print('\nThe Neural Network Classifier predicted')
    print('--------------------------------------')
    print('%-5g exoplanets      of %g candidates'  %(pred_Conf, len(pred_cand)))
    print('%-5g false positives of %g candidates'  %(pred_FP, len(pred_cand)))
