"""
FYS-STK4155 - Project 3: XGBoost
"""
import numpy                 as np
import scikitplot            as skplt
import matplotlib.pyplot	 as plt
import xgboost               as xgb
import functions             as func
import goldilock             as GL

from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost 				 import plot_importance
from matplotlib				 import pyplot

from sklearn.metrics         import precision_score, recall_score, accuracy_score, mean_squared_error, mean_absolute_error, f1_score

from sklearn.metrics         import precision_score, recall_score,          \
                                    accuracy_score, confusion_matrix,       \
                                    mean_absolute_error, f1_score


#------------------------------------------------------------------------------
def XG_Boost(X_train, X_test, y_train, y_test, candidates, GoldiLock,   \
             feature_list, header_names, seed, Goldilock_zone=False, plot_confuse_matrix=False):


    param_test = {"max_depth":      [6,7,8,9],
                  "n_estimator":    [50, 100],  # 100
                  "subsample":      [0.6,0.8,1.0],
                  "learning_rate":  [0.1, 0.2, 0.3]
                  }

    gsearch = GridSearchCV(xgb.XGBClassifier(), param_grid = param_test, cv=5)
    model2 = gsearch.fit(X_train, y_train)

    # Print best parameters
    print(gsearch.best_params_)

    best_preds = model2.predict(X_test)

    # Calculating different metrics
    accuracy 	= accuracy_score(y_test, best_preds)
    precision   = precision_score(y_test, best_preds, average="macro")
    recall      = recall_score(y_test, best_preds, average="macro")
    F1_score    = f1_score(y_test, best_preds, average="macro") # SE PAA DENNE!!! Feil input..
    # Calculate the absolute errors
    errors      = abs(best_preds - y_test) # SE PAA DENNE!!! Blir veldig hoy verdi. Feil input..

    # Printing the different metri
    func.Print_parameters(accuracy, F1_score, precision, recall, errors, name='XGBoost')

    if plot_confuse_matrix == True:
        skplt.metrics.plot_confusion_matrix(y_test, best_preds)
        plt.show()


    pred_cand  = model2.predict(candidates)
    print(pred_cand)

    # Divide into predicted false positives and confirmed exoplanets
    pred_FP    = (pred_cand == 0).sum() 	# Predicted false positives
    pred_Conf  = (pred_cand == 1).sum() 	# Predicted exoplanets/confirmed

    # Information print to terminal
    print('\nThe XGBoost method predicted')
    print('--------------------------------------')
    print('%-5g exoplanets      of %g candidates'  %(pred_Conf, len(pred_cand)))
    print('%-5g false positives of %g candidates'  %(pred_FP, len(pred_cand)))

    #plot_importance(y_pred)
    #pyplot.show()

    if plot_confuse_matrix == True:
        # Plotting a bar plot of candidates predicted as confirmed and false positives
        func.Histogram2(pred_cand, method='XGBoost Classification')


    if Goldilock_zone:

        print("Goldilock zone calculations")

        D_test             = GoldiLock
        predict_goldilocks = model2.predict(D_test)
        np.save('GoldiLock_predicted', predict_goldilocks)

        predicted_false_positive_goldilocs  = (predict_goldilocks == 0).sum()
        predicted_exoplanets_goldilocks     = (predict_goldilocks == 1).sum()

        # Information print to terminal
        print('\nThe XGBoost method predicted')
        print('--------------------------------------')
        print('%-3g exoplanets      of %g candidates'  %(predicted_exoplanets_goldilocks, len(predict_goldilocks)))
        print('%-3g false positives of %g candidates'  %(predicted_false_positive_goldilocs, len(predict_goldilocks)))

        # Plotting a bar plot of candidates predicted as confirmed and false positives
        func.Histogram2(predict_goldilocks, method='XGBoost Classification')

        GL.GoldilocksZone()
